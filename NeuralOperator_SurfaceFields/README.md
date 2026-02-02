# Fourier Neural Operator for DrivAerNet++ Surface Pressure Prediction

This directory contains an implementation of the **Fourier Neural Operator (FNO)** for predicting surface pressure fields on automotive geometries from the DrivAerNet++ dataset.

## Reference

**Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A.** (2020). *Fourier neural operator for parametric partial differential equations.* arXiv preprint arXiv:2010.08895.

## Architecture Overview

The FNO processes aerodynamic data through three stages:

### 1. Voxelization
- Converts surface point clouds to regular **32³ voxel grids**
- Each voxel has 4 channels:
  - **Occupancy field**: 1 if geometry present, 0 otherwise
  - **Normalized coordinates**: (x, y, z) position in [0, 1]³

### 2. Fourier Neural Operator
- Operates in spectral domain using **3D FFT**
- **Spectral convolution**: Multiplies Fourier modes with learned weights
- Keeps only **first 8 modes** in each dimension for efficiency
- **4 Fourier layers** with skip connections
- Hidden dimension: **16 channels** (optimized for 2.1M params)

### 3. Surface Reconstruction
- **Trilinear interpolation**: Samples volume features at surface points
- **Point-wise refinement**: MLP with 64 hidden units
- Input: interpolated features (16D) + normalized position (3D)
- Output: pressure value at each surface point

## Model Statistics

- **Parameters**: ~2.10M
- **Performance** (from benchmark):
  - R²: 0.8503
  - MSE: 559 (normalized)
  
## Key Features

### Spectral Convolution
- Efficient global receptive field via FFT
- Learns in frequency domain
- Captures multi-scale patterns

### Advantages
- **Grid-based**: Regular structure enables FFT
- **Global context**: Full receptive field from first layer
- **Efficient**: O(N log N) complexity for FFT

### Limitations
- **Fixed resolution**: Requires consistent 32³ grid
- **Memory**: Stores full 3D volume
- **Interpolation loss**: Voxelization may lose fine details

## Directory Structure

```
NeuralOperator_SurfaceFields/
├── data_loader.py          # Voxelization and data loading
├── model.py                # FNO architecture
├── train.py                # Training script
└── README.md               # This file
```

## Installation

### Requirements
```bash
pip install torch torchvision
pip install pyvista numpy tqdm
```

### Data Preparation
Ensure VTK files are organized:
```
PressureVTK/
├── design_0001.vtk
├── design_0002.vtk
└── ...
```

Split files (train/val/test IDs):
```
train_val_test_splits/
├── train_design_ids.txt
├── val_design_ids.txt
└── test_design_ids.txt
```

## Usage

### Basic Training
```bash
python train.py \
    --dataset_path ../PressureVTK \
    --subset_dir ../train_val_test_splits \
    --batch_size 16 \
    --num_epochs 100
```

### Full Configuration
```bash
python train.py \
    --dataset_path ../PressureVTK \
    --subset_dir ../train_val_test_splits \
    --cache_dir ./cache_fno \
    --grid_resolution 32 \
    --fno_modes 8 \
    --fno_width 16 \
    --fno_layers 4 \
    --refine_hidden 64 \
    --num_points 10000 \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 2e-3 \
    --weight_decay 1e-4 \
    --save_dir ./checkpoints_fno \
    --seed 42
```

### Parameters Explained

**Voxelization:**
- `--grid_resolution 32`: Voxel grid size (32³ = 32,768 voxels)
- `--num_points 10000`: Surface points to sample for targets

**FNO Architecture:**
- `--fno_modes 8`: Fourier modes to keep (higher = more details, but slower)
- `--fno_width 16`: Hidden channels in FNO (calibrated for 2.1M params)
- `--fno_layers 4`: Number of spectral convolution layers
- `--refine_hidden 64`: Refinement network hidden size

**Training:**
- `--batch_size 16`: Process 16 geometries simultaneously
- `--learning_rate 2e-3`: AdamW learning rate
- `--weight_decay 1e-4`: L2 regularization
- `--num_epochs 100`: Total training iterations

## Implementation Details

### Data Processing

```python
from data_loader import VoxelGridDataset, get_dataloaders

# Create dataset
dataset = VoxelGridDataset(
    root_dir='../PressureVTK',
    grid_resolution=32,
    num_points=10000,
    preprocess=True,
)

# Get data loaders
train_loader, val_loader, test_loader = get_dataloaders(
    dataset_path='../PressureVTK',
    subset_dir='../train_val_test_splits',
    grid_resolution=32,
    batch_size=16,
)
```

### Model Creation

```python
from model import FNOSurfaceFieldPredictor

model = FNOSurfaceFieldPredictor(
    grid_resolution=32,
    fno_modes=8,
    fno_width=16,
    fno_layers=4,
    refine_hidden=64,
)

print(f"Parameters: {model.count_parameters():,}")
```

### Inference

```python
# Load trained model
checkpoint = torch.load('checkpoints_fno/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    predictions = model(voxel_grids, positions, bboxes)
    # predictions: list of (N_points, 1) tensors
```

## Comparison with Other Models

| Model | Parameters | R² Score | MSE | Notes |
|-------|-----------|----------|-----|-------|
| PointNet | 1.67M | 0.7639 | 1,048 | Point-based |
| **NeuralOperator** | **2.10M** | **0.8503** | **559** | **Grid-based, spectral** |
| PointTransformer | 3.05M | 0.9359 | 285 | Attention |
| AB-UPT | 6.01M | 0.9675 | 144 | Transformer |

**Insights:**
- FNO achieves competitive performance with few parameters
- Grid-based approach captures global structure efficiently
- Spectral learning is effective for smooth pressure fields
- Trade-off: resolution vs memory

## Training Tips

### Memory Management
- **Reduce batch size** if OOM: `--batch_size 8`
- **Lower resolution**: `--grid_resolution 24` (uses ~58% memory)
- **Smaller model**: `--fno_width 12 --fno_layers 3`

### Improving Performance
1. **More modes**: `--fno_modes 12` (captures finer details)
2. **Wider network**: `--fno_width 24` (increases capacity)
3. **Deeper network**: `--fno_layers 6`
4. **Larger refinement**: `--refine_hidden 128`
5. **Data augmentation**: Random rotations/scaling

### Convergence Issues
- **Lower LR**: `--learning_rate 1e-3`
- **Gradient clipping**: Already enabled (max_norm=1.0)
- **Check normalization**: Ensure pressure normalization is correct

## Technical Notes

### Spectral Convolution
```python
# Pseudocode
x_ft = FFT(x)  # Transform to frequency domain
x_ft = x_ft * learned_weights  # Multiply low frequencies
x = IFFT(x_ft)  # Back to spatial domain
```

### Why FFT Works
- Pressure fields are smooth → dominated by low frequencies
- FFT provides global receptive field immediately
- Computational efficiency: O(N log N)

### Voxelization Strategy
- **Occupancy**: Binary indicator of geometry presence
- **Coordinates**: Provide positional encoding
- **Interpolation**: Recovers surface from volume

## Output

### Checkpoints
- `best_model.pth`: Model with highest validation R²
- `checkpoint_epoch_X.pth`: Periodic saves
- `config.txt`: Hyperparameters used
- `test_results.txt`: Final test metrics

### Metrics Logged
- **MSE**: Mean squared error (main loss)
- **MAE**: Mean absolute error
- **R²**: Coefficient of determination (goodness of fit)
- **RMSE**: Root mean squared error

## Extending the Code

### Higher Resolution
```python
# 64³ grid (requires 8x memory)
model = FNOSurfaceFieldPredictor(grid_resolution=64, fno_modes=16, fno_width=16)
```

### Multi-Field Prediction
```python
# Predict pressure + WSS (4 outputs)
self.fno = FNO3d(..., out_channels=32)
self.refinement = nn.Linear(32 + 3, 4)
```

### Adaptive Modes
```python
# Learn which modes to use
self.mode_weights = nn.Parameter(torch.ones(modes1, modes2, modes3))
```

## Citation

If you use this implementation, please cite:

```bibtex
@article{li2020fourier,
  title={Fourier neural operator for parametric partial differential equations},
  author={Li, Zongyi and Kovachki, Nikola and Azizzadenesheli, Kamyar and Liu, Burigede and Bhattacharya, Kaushik and Stuart, Andrew and Anandkumar, Anima},
  journal={arXiv preprint arXiv:2010.08895},
  year={2020}
}
```

## Troubleshooting

### Issue: Slow voxelization
**Solution**: Enable caching with `--cache_dir ./cache_fno`. First run preprocesses, subsequent runs are fast.

### Issue: Poor accuracy
**Solution**: 
1. Increase resolution: `--grid_resolution 48`
2. More training: `--num_epochs 200`
3. Larger model: `--fno_width 24 --fno_modes 12`

### Issue: NaN loss
**Solution**:
1. Check data normalization
2. Reduce learning rate: `--learning_rate 1e-3`
3. Enable gradient clipping (already in code)

## Contact

For questions about this FNO implementation for DrivAerNet, please open an issue or refer to the original FNO paper.

---

**Last Updated**: 2024
**DrivAerNet++ Dataset**: [Link to dataset]
**FNO Paper**: https://arxiv.org/abs/2010.08895
