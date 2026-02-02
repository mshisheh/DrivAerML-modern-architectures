"""
Visual Architecture Diagram: Fourier Neural Operator for Surface Pressure Prediction
=====================================================================================

This diagram shows how the FNO processes voxelized geometry for pressure field prediction.
Reference: Li, Z. et al. Fourier neural operator for parametric partial differential equations.
           arXiv preprint arXiv:2010.08895 (2020).
"""

ARCHITECTURE_DIAGRAM = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║              Fourier Neural Operator (FNO) for DrivAerNet++                   ║
║                    Surface Pressure Field Prediction                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

STAGE 1: VOXELIZATION (Preprocessing)
═══════════════════════════════════════════════════════════════════════════════
INPUT: Surface Point Cloud (STL/VTK mesh)
┌─────────────────────────────────────────────┐
│  • Automotive geometry (car surface)        │
│  • N_points: ~50,000 - 100,000 vertices     │  Point Cloud
│  • Pressure values: p ∈ [-300, 100] Pa      │  (.vtk file)
│  • Bounding box computed with 10% padding   │
└─────────────────────────────────────────────┘
                    │
                    ▼ [Voxelization Process]
┌─────────────────────────────────────────────────────────────────────────┐
│                     VOXEL GRID CONSTRUCTION (32³)                       │
│                                                                         │
│  Step 1: Create regular 3D grid                                        │
│    • X-axis: linspace(x_min, x_max, 32)                               │
│    • Y-axis: linspace(y_min, y_max, 32)                               │
│    • Z-axis: linspace(z_min, z_max, 32)                               │
│    • Total voxels: 32,768                                              │
│                                                                         │
│  Step 2: Compute occupancy field                                       │
│    • occupancy[i,j,k] = 1 if voxel contains surface                   │
│    • occupancy[i,j,k] = 0 if voxel is empty                           │
│                                                                         │
│  Step 3: Add normalized coordinates                                    │
│    • x_norm = (x - x_min) / (x_max - x_min) ∈ [0, 1]                 │
│    • y_norm = (y - y_min) / (y_max - y_min) ∈ [0, 1]                 │
│    • z_norm = (z - z_min) / (z_max - z_min) ∈ [0, 1]                 │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
INPUT TO FNO: Voxel Grid (4 channels)
┌─────────────────────────────────────────────┐
│  Channel 0: Occupancy field [0 or 1]        │
│  Channel 1: x_normalized [0, 1]             │  Shape: (B, 4, 32, 32, 32)
│  Channel 2: y_normalized [0, 1]             │  Size: 131,072 values
│  Channel 3: z_normalized [0, 1]             │  Memory: ~0.5 MB per sample
└─────────────────────────────────────────────┘
                    │
                    ▼

STAGE 2: FOURIER NEURAL OPERATOR
═══════════════════════════════════════════════════════════════════════════════
┌─────────────────────────────────────────────────────────────────────────┐
│                         LIFTING LAYER                                   │
│  Linear: (4 → 16 channels)                                             │
│    • Applied point-wise to each voxel                                  │
│    • Projects input to hidden dimension                                │
│    • Params: 4 × 16 = 64                                               │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼ Shape: (B, 16, 32, 32, 32)
┌─────────────────────────────────────────────────────────────────────────┐
│                    FOURIER LAYER 1 (Spectral)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────┐      │
│  │              SPECTRAL CONVOLUTION BRANCH                    │      │
│  │                                                              │      │
│  │  Step 1: FFT (Real → Frequency Domain)                      │      │
│  │    x_ft = FFT(x) ──────► (B, 16, 32, 32, 17)              │ ─┐   │
│  │    [Using rfftn for real input]                            │  │   │
│  │                                                              │  │   │
│  │  Step 2: Spectral Multiplication (Learned in Fourier)      │  │   │
│  │    Keep only first 8 modes in each dimension               │  │   │
│  │    ┌─────────────────────────────────────────────┐         │  │   │
│  │    │  Weight Tensor 1: (16, 16, 8, 8, 8)         │         │  │   │
│  │    │    Upper octant (positive freqs)            │         │  │   │
│  │    │    Params: 16×16×8³ = 131,072 (complex)    │         │  │   │
│  │    │                                              │         │  │   │
│  │    │  Weight Tensor 2: (16, 16, 8, 8, 8)         │         │  │   │
│  │    │    Lower octant (negative freq dim 1)       │         │  │   │
│  │    │    Params: 131,072 (complex)                │         │  │   │
│  │    │                                              │         │  │   │
│  │    │  Weight Tensor 3: (16, 16, 8, 8, 8)         │         │  │   │
│  │    │    Negative freq dim 2                      │         │  │   │
│  │    │    Params: 131,072 (complex)                │         │  │   │
│  │    │                                              │         │  │   │
│  │    │  Weight Tensor 4: (16, 16, 8, 8, 8)         │         │  │   │
│  │    │    Negative freq both dims                  │         │  │   │
│  │    │    Params: 131,072 (complex)                │         │  │   │
│  │    └─────────────────────────────────────────────┘         │  │   │
│  │    Total: 1,048,576 parameters (complex = 2M real)         │  │   │
│  │                                                              │  │   │
│  │  Step 3: IFFT (Frequency → Real Domain)                    │  │   │
│  │    x1 = IFFT(x_ft * W) ──────► (B, 16, 32, 32, 32)        │  │   │
│  └──────────────────────────────────────────────────────────────┘  │   │
│                                                                     │   │
│  ┌─────────────────────────────────────────────────────────┐       │   │
│  │              SKIP CONNECTION BRANCH                     │       │   │
│  │    Conv3D (1×1×1): 16 → 16 channels                     │       │   │
│  │    x2 = Conv(x) ──────► (B, 16, 32, 32, 32)            │       │   │
│  │    Params: 16 × 16 + 16 = 272                           │ ──────┘   │
│  └─────────────────────────────────────────────────────────┘           │
│                                                                         │
│  Combine: x = x1 + x2 (element-wise addition)                         │
│  Activation: x = GELU(x)                                               │
│                                                                         │
│  Layer Params: 2,097,424 (spectral) + 272 (skip) = 2,097,696         │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼ Same shape: (B, 16, 32, 32, 32)
┌─────────────────────────────────────────────────────────────────────────┐
│                    FOURIER LAYER 2 (Spectral)                          │
│  [Identical structure to Layer 1]                                      │
│  Params: 2,097,696                                                     │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    FOURIER LAYER 3 (Spectral)                          │
│  [Identical structure to Layer 1]                                      │
│  Params: 2,097,696                                                     │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    FOURIER LAYER 4 (Spectral)                          │
│  [Identical structure to Layer 1]                                      │
│  No activation after this layer                                        │
│  Params: 2,097,696                                                     │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼ Shape: (B, 16, 32, 32, 32)
┌─────────────────────────────────────────────────────────────────────────┐
│                      PROJECTION LAYERS                                  │
│  Conv3D (1×1×1): 16 → 8 channels                                      │
│    Params: 16 × 8 + 8 = 136                                           │
│  GELU activation                                                       │
│  Conv3D (1×1×1): 8 → 16 channels (output features)                   │
│    Params: 8 × 16 + 16 = 144                                          │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
FNO OUTPUT: Volume Features (16 channels)
┌─────────────────────────────────────────────┐
│  • Shape: (B, 16, 32, 32, 32)               │
│  • Encoded pressure information in volume   │
│  • Global context via spectral convolutions │
└─────────────────────────────────────────────┘
                    │
                    ▼

STAGE 3: SURFACE INTERPOLATION & REFINEMENT
═══════════════════════════════════════════════════════════════════════════════
┌─────────────────────────────────────────────────────────────────────────┐
│                  TRILINEAR INTERPOLATION                                │
│                                                                         │
│  Input:                                                                 │
│    • Volume features: (B, 16, 32, 32, 32)                             │
│    • Query points: (N_points, 3) - surface positions                  │
│    • Bounding box: for coordinate normalization                       │
│                                                                         │
│  Process:                                                               │
│    1. Normalize query points to [-1, 1]³                              │
│       p_norm = 2 × (p - bbox_min)/(bbox_max - bbox_min) - 1          │
│                                                                         │
│    2. F.grid_sample with mode='bilinear'                              │
│       • Samples volume at arbitrary 3D positions                       │
│       • Trilinear interpolation between 8 nearest voxels              │
│       • align_corners=True for proper boundary handling               │
│                                                                         │
│    3. Extract features at each surface point                          │
│       features[i] = interpolate(volume, position[i])                  │
│                                                                         │
│  Output: (N_points, 16) - interpolated features per surface point     │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    POINT-WISE REFINEMENT MLP                            │
│                                                                         │
│  Input Features: Concatenate [interpolated_features, normalized_pos]   │
│    • Interpolated: 16 channels                                         │
│    • Position: 3 channels (x_norm, y_norm, z_norm)                    │
│    • Combined: 19 channels                                             │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────┐          │
│  │  Layer 1: Linear(19 → 64)                               │          │
│  │    Params: 19 × 64 + 64 = 1,280                         │          │
│  │  GELU activation                                         │          │
│  │                                                          │          │
│  │  Layer 2: Linear(64 → 64)                               │          │
│  │    Params: 64 × 64 + 64 = 4,160                         │          │
│  │  GELU activation                                         │          │
│  │                                                          │          │
│  │  Layer 3: Linear(64 → 1)                                │          │
│  │    Params: 64 × 1 + 1 = 65                              │          │
│  │  No activation (regression)                             │          │
│  └─────────────────────────────────────────────────────────┘          │
│                                                                         │
│  Total Refinement Params: 5,505                                        │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
OUTPUT: Surface Pressure Predictions
┌─────────────────────────────────────────────┐
│  • Shape: (N_points, 1)                      │
│  • One pressure value per surface point      │
│  • Denormalized: p_real = p_pred × σ + μ    │
│    where μ = -94.5 Pa, σ = 117.25 Pa        │
└─────────────────────────────────────────────┘


PARAMETER BREAKDOWN:
═══════════════════════════════════════════════════════════════════════════════
┌─────────────────────────────────────────────────────────────────────────┐
│  Component                    │ Parameters  │ Percentage │ Memory       │
│───────────────────────────────┼─────────────┼────────────┼──────────────│
│  Lifting Layer                │ 64          │ 0.003%     │ 256 B        │
│  Fourier Layer 1 (spectral)   │ 2,097,424   │ 99.68%     │ 8.39 MB      │
│  Fourier Layer 1 (skip)       │ 272         │ 0.013%     │ 1.09 KB      │
│  Fourier Layer 2              │ 2,097,696   │ 99.69%     │ 8.39 MB      │
│  Fourier Layer 3              │ 2,097,696   │ 99.69%     │ 8.39 MB      │
│  Fourier Layer 4              │ 2,097,696   │ 99.69%     │ 8.39 MB      │
│  Projection Layers            │ 280         │ 0.013%     │ 1.12 KB      │
│  Refinement MLP               │ 5,505       │ 0.26%      │ 22.0 KB      │
│───────────────────────────────┼─────────────┼────────────┼──────────────│
│  TOTAL                        │ 2,104,105   │ 100%       │ 8.41 MB      │
└─────────────────────────────────────────────────────────────────────────┘

Note: Spectral convolution parameters are complex-valued (2× memory)


COMPUTATIONAL COMPLEXITY:
═══════════════════════════════════════════════════════════════════════════════

Forward Pass Operations (per sample):
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage                    │ Operation                │ Complexity       │
│───────────────────────────┼──────────────────────────┼──────────────────│
│  Voxelization             │ Point → Grid mapping     │ O(N × 32³)       │
│  Lifting                  │ Point-wise linear        │ O(4 × 16 × 32³)  │
│  FFT (×4 layers)          │ 3D Real FFT              │ O(4 × 32³log32)  │
│  Spectral Conv (×4)       │ Complex multiplication   │ O(4 × 16² × 8³)  │
│  IFFT (×4 layers)         │ 3D Inverse FFT           │ O(4 × 32³log32)  │
│  Skip Conv (×4)           │ 1×1×1 convolutions       │ O(4 × 16² × 32³) │
│  Interpolation            │ Grid sampling            │ O(N × 8)         │
│  Refinement               │ MLP per point            │ O(N × 64²)       │
│───────────────────────────┴──────────────────────────┴──────────────────│
│  Dominant: FFT/IFFT = O(32³ log 32) ≈ 164K operations per layer        │
│  Total FFT operations: 8 × 164K ≈ 1.3M (8 FFTs: 4 forward + 4 inverse) │
└─────────────────────────────────────────────────────────────────────────┘

Training: Forward + Backward ≈ 3× forward pass complexity


MEMORY USAGE (Batch Size = 16):
═══════════════════════════════════════════════════════════════════════════════
┌─────────────────────────────────────────────────────────────────────────┐
│  Component               │ Shape              │ Size/Sample │ Total(B=16)│
│──────────────────────────┼────────────────────┼─────────────┼────────────│
│  Input Voxel Grid        │ (4, 32, 32, 32)    │ 512 KB      │ 8.2 MB     │
│  Hidden Features (×4)    │ (16, 32, 32, 32)   │ 2.0 MB      │ 128 MB     │
│  FFT Buffers (complex)   │ (16, 32, 32, 17)   │ 2.2 MB      │ 141 MB     │
│  Surface Points          │ (10000, 3)         │ 117 KB      │ 1.9 MB     │
│  Interpolated Features   │ (10000, 16)        │ 625 KB      │ 10 MB      │
│  Gradients (training)    │ [Same as forward]  │ ~5 MB       │ ~280 MB    │
│  Model Parameters        │ 2.1M params        │ 8.4 MB      │ 8.4 MB     │
│──────────────────────────┴────────────────────┴─────────────┴────────────│
│  Peak Training Memory:                                       ~580 MB     │
│  Peak Inference Memory:                                      ~290 MB     │
└─────────────────────────────────────────────────────────────────────────┘


KEY DESIGN CHOICES:
═══════════════════════════════════════════════════════════════════════════════

1. SPECTRAL CONVOLUTION vs SPATIAL CONVOLUTION:
   ┌────────────────────────────────────────────────────────────────┐
   │  Spatial (U-Net):  Local receptive field, explicit hierarchy   │
   │  Spectral (FNO):   Global receptive field from layer 1         │
   └────────────────────────────────────────────────────────────────┘
   
   Advantage: FFT provides global context immediately
   Trade-off: Fixed grid resolution, less flexible for varying sizes

2. FOURIER MODES (8 out of 32):
   • Low frequencies: 8³ = 512 modes (kept)
   • High frequencies: 24³ ≈ 13,800 modes (discarded)
   • Rationale: Pressure fields are smooth, dominated by low frequencies
   • Effect: 96% reduction in spectral parameters

3. GRID RESOLUTION (32³):
   • Higher: Better accuracy, more memory
   • Lower: Faster, less accurate
   • 32³: Sweet spot for 2.1M param budget
   
   Alternative: 64³ grid → 16M params (8× increase)

4. REFINEMENT NETWORK:
   • Corrects interpolation errors
   • Adds local geometric details
   • Only 0.26% of parameters
   • Critical for surface-to-surface prediction


COMPARISON WITH OTHER ARCHITECTURES:
═══════════════════════════════════════════════════════════════════════════════
┌──────────────────┬──────────┬──────────┬──────────┬─────────────────────┐
│ Model            │ Params   │ R²       │ MSE      │ Key Feature         │
├──────────────────┼──────────┼──────────┼──────────┼─────────────────────┤
│ PointNet         │ 1.67M    │ 0.7639   │ 1,048    │ Point-wise MLP      │
│ NeuralOperator   │ 2.10M    │ 0.8503   │ 559      │ Spectral learning   │
│ PointTransformer │ 3.05M    │ 0.9359   │ 285      │ Self-attention      │
│ AB-UPT           │ 6.01M    │ 0.9675   │ 144      │ Hierarchical attn   │
│ TransolverLarge  │ 7.58M    │ 0.9595   │ 180      │ Physics-informed    │
│ TripNet          │ 24.10M   │ 0.9590   │ 182      │ Triple encoders     │
└──────────────────┴──────────┴──────────┴──────────┴─────────────────────┘

FNO Position: Best efficiency (R²/param ratio) among grid-based methods


ADVANTAGES OF FNO:
═══════════════════════════════════════════════════════════════════════════════
✓ Global receptive field from first layer (via FFT)
✓ Resolution-invariant operators (can train on 32³, test on 64³)
✓ Efficient O(N log N) complexity for grid operations
✓ Mathematically principled (operator learning in Fourier space)
✓ Captures multi-scale patterns via different frequency modes
✓ No pooling/unpooling artifacts (unlike U-Net)


LIMITATIONS OF FNO:
═══════════════════════════════════════════════════════════════════════════════
✗ Requires regular grid input (voxelization needed)
✗ Fixed resolution during training (though can generalize)
✗ Memory scales cubically with resolution (32³ → 64³ = 8× memory)
✗ Voxelization can lose fine surface details
✗ Not suitable for point clouds directly (needs preprocessing)
✗ Periodic boundary assumption (less natural for open domains)


TRAINING CONFIGURATION:
═══════════════════════════════════════════════════════════════════════════════
Optimizer:      AdamW(lr=2e-3, weight_decay=1e-4)
Scheduler:      CosineAnnealingLR(T_max=100, eta_min=1e-6)
Loss:           MSE (normalized pressure)
Batch Size:     16 samples
Epochs:         100
Gradient Clip:  max_norm=1.0
Data:           ~5,500 train, ~700 val, ~1,900 test designs
Points/Sample:  10,000 surface points for supervision


INFERENCE PIPELINE:
═══════════════════════════════════════════════════════════════════════════════
1. Load VTK mesh                          [I/O bound]
2. Voxelize to 32³ grid                   [CPU: ~50ms]
3. FNO forward pass (volume)              [GPU: ~10ms]
4. Interpolate to 10K surface points      [GPU: ~5ms]
5. MLP refinement per point               [GPU: ~15ms]
   ────────────────────────────────────────────────────
   Total: ~80ms per design (on RTX 3090)
   
Throughput: ~12.5 designs/second (single GPU)
"""

IMPLEMENTATION_NOTES = """
IMPLEMENTATION HIGHLIGHTS:
═════════════════════════════════════════════════════════════════════════════

1. SPECTRAL CONVOLUTION (Core Innovation):
   ```python
   def forward(self, x):
       # Transform to frequency domain
       x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
       
       # Multiply low-frequency modes with learned weights
       out_ft[:, :, :modes1, :modes2, :modes3] = \
           self.compl_mul3d(x_ft[:, :, :modes1, :modes2, :modes3], 
                           self.weights1)
       
       # Handle negative frequencies (all 8 octants)
       # ... [weights2, weights3, weights4]
       
       # Transform back to spatial domain
       x = torch.fft.irfftn(out_ft, s=x.shape[-3:])
       return x
   ```

2. VOXELIZATION (Preprocessing):
   ```python
   def _voxelize_mesh(self, mesh, resolution=32):
       # Discretize points to voxel indices
       normalized_points = (points - bbox_min) / (bbox_max - bbox_min)
       voxel_indices = (normalized_points * (resolution - 1)).astype(int)
       
       # Mark occupied voxels
       occupancy[voxel_indices[:, 0], 
                voxel_indices[:, 1], 
                voxel_indices[:, 2]] = 1.0
       
       # Add coordinate channels
       voxel_grid = np.stack([occupancy, x_norm, y_norm, z_norm])
       return voxel_grid
   ```

3. TRILINEAR INTERPOLATION:
   ```python
   def interpolate_to_points(self, volume_features, positions, bbox):
       # Normalize positions to [-1, 1] for grid_sample
       normalized_pos = 2 * (pos - bbox_min) / (bbox_max - bbox_min) - 1
       
       # Interpolate (trilinear)
       features = F.grid_sample(
           volume_features,
           normalized_pos.unsqueeze(0).unsqueeze(0).unsqueeze(0),
           mode='bilinear',
           align_corners=True
       )
       return features.squeeze()
   ```

4. FLEXIBLE ARCHITECTURE:
   ```python
   model = FNOSurfaceFieldPredictor(
       grid_resolution=32,      # Can change: 16, 24, 32, 48, 64
       fno_modes=8,             # Frequency cutoff: 4, 8, 12, 16
       fno_width=16,            # Hidden channels: 8, 16, 24, 32
       fno_layers=4,            # Depth: 2, 4, 6, 8
       refine_hidden=64,        # MLP size: 32, 64, 128, 256
   )
   ```

5. MULTI-FIELD EXTENSION:
   ```python
   # For pressure + wall shear stress (4 outputs)
   self.fno = FNO3d(..., out_channels=32)  # More features
   self.refinement = nn.Sequential(
       nn.Linear(32 + 3, 128),
       nn.GELU(),
       nn.Linear(128, 4),  # [pressure, WSS_x, WSS_y, WSS_z]
   )
   ```


DEBUGGING TIPS:
═════════════════════════════════════════════════════════════════════════════
• NaN in FFT: Check for extreme values in input, add epsilon
• Poor accuracy: Increase fno_modes (more frequencies) or fno_width
• OOM errors: Reduce batch_size or grid_resolution
• Slow training: Profile FFT operations, ensure CUDA is used
• Bad interpolation: Verify bbox normalization is correct
• Mode aliasing: Ensure modes < resolution/2 (Nyquist limit)


MATHEMATICAL FOUNDATION:
═════════════════════════════════════════════════════════════════════════════
FNO learns a mapping between function spaces:

   G_θ : X → Y

where X = pressure coefficient distributions, Y = surface pressure fields

Key Idea: Parameterize G_θ as an integral kernel in Fourier space

   (K(x))(y) = ℱ⁻¹(ℱ(x) · W)

Properties:
1. Resolution-invariant (discretization-independent)
2. Captures long-range dependencies efficiently
3. Learns smooth operators (spectral bias helps regularization)


FUTURE EXTENSIONS:
═════════════════════════════════════════════════════════════════════════════
1. Adaptive Fourier modes (learnable frequency selection)
2. Hierarchical grid refinement (octree-based)
3. Attention in Fourier space (query/key/value in frequency domain)
4. Physics-informed loss (enforce Navier-Stokes in spectral form)
5. Multi-resolution training (curriculum from coarse to fine)
6. Uncertainty quantification (ensemble FNOs or Bayesian variants)
"""

if __name__ == '__main__':
    print(ARCHITECTURE_DIAGRAM)
    print("\n" + "="*80 + "\n")
    print(IMPLEMENTATION_NOTES)
