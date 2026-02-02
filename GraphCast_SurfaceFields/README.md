# GraphCast for DrivAerNet Surface Pressure Prediction

Self-contained implementation of GraphCast for DrivAerNet surface field prediction, based on the paper "GraphCast: Learning skillful medium-range global weather forecasting" (https://arxiv.org/abs/2212.12794).

## Overview

GraphCast is a state-of-the-art graph neural network architecture developed by Google DeepMind for weather forecasting. This implementation adapts GraphCast for automotive aerodynamics, specifically for predicting pressure coefficients on vehicle surfaces.

### Key Features

- **Multi-Scale Mesh Representation**: Uses a latent mesh to capture global flow patterns
- **Encoder-Processor-Decoder Architecture**: 
  - Encoder: Maps surface points to latent mesh
  - Processor: Multi-layer message passing on mesh
  - Decoder: Maps mesh back to surface predictions
- **No External Dependencies**: Standalone implementation without PhysicsNemo

## Architecture

```
Input Surface Points (N, 7)
    ↓
[Grid Embedder]
    ↓
Grid Features (N, hidden_dim)
    ↓
[Grid-to-Mesh Encoder] ←→ Mesh Features (M, hidden_dim)
    ↓
Grid Features (N, hidden_dim)
    ↓
[Mesh Processor] × L layers
    ↓
Mesh Features (M, hidden_dim)
    ↓
[Mesh-to-Grid Decoder] ←→ Grid Features (N, hidden_dim)
    ↓
[Output MLP]
    ↓
Pressure Predictions (N, 1)
```

### Components

1. **Embedders**: 
   - Grid embedder: Maps input features (x, y, z, nx, ny, nz, area) to hidden_dim
   - Mesh embedder: Maps mesh node positions (x, y, z) to hidden_dim
   - Edge embedder: Maps edge features (dx, dy, dz, distance) to hidden_dim

2. **Encoder (Grid-to-Mesh)**:
   - Bipartite graph connecting surface points to mesh nodes
   - Edge updates: f([edge_attr, src_feat, dst_feat])
   - Node updates: f([node_feat, aggregated_edge_feat])
   - k-NN connectivity (typically k=4)

3. **Processor (Mesh)**:
   - Multiple layers of message passing on mesh
   - Same edge/node update structure as encoder
   - Residual connections throughout
   - k-NN mesh connectivity (typically k=10)

4. **Decoder (Mesh-to-Grid)**:
   - Bipartite graph connecting mesh nodes back to surface points
   - Aggregates information from mesh to surface
   - k-NN connectivity (typically k=4)

5. **Output MLP**:
   - Maps final grid features to pressure predictions
   - Simple feedforward network

## Usage

### Training

```bash
python train.py \
    --data_dir /path/to/DrivAerNet/surface_field_data \
    --split_dir /path/to/DrivAerNet/train_val_test_splits \
    --hidden_dim 384 \
    --num_mesh_nodes 800 \
    --num_processor_layers 12 \
    --batch_size 1 \
    --num_epochs 100 \
    --lr 1e-4 \
    --save_dir ./checkpoints/graphcast \
    --log_dir ./logs/graphcast
```

### Parameter Configurations

Target different model sizes by adjusting hyperparameters:

| Config | hidden_dim | num_mesh_nodes | num_processor_layers | ~Params |
|--------|-----------|---------------|---------------------|---------|
| Small  | 256       | 600           | 8                   | ~1.5M   |
| Medium | 384       | 800           | 12                  | ~3.0M   |
| Large  | 512       | 1000          | 16                  | ~5.5M   |
| XLarge | 640       | 1200          | 20                  | ~9.0M   |

### Model Creation

```python
from model import create_graphcast, count_parameters

# Create model
model = create_graphcast(
    hidden_dim=384,
    num_mesh_nodes=800,
    num_processor_layers=12,
    num_mlp_layers=1,
    input_dim=7,  # x, y, z, nx, ny, nz, area
    output_dim=1, # pressure
)

# Count parameters
total_params = count_parameters(model)
print(f"Total parameters: {total_params:,}")
```

## Data Format

Expected input format:
- **Features** (7 dimensions): [x, y, z, nx, ny, nz, area]
  - x, y, z: Point coordinates
  - nx, ny, nz: Surface normals
  - area: Approximate area per point
- **Target** (1 dimension): Pressure coefficient (Cp)

Data files: `.npy` format, shape `[num_points, 8]`

## File Structure

```
GraphCast_SurfaceFields/
├── model.py           # GraphCast model implementation
├── data_loader.py     # Data loading and preprocessing
├── train.py           # Training script
├── README.md          # This file
└── architecture_diagram.py  # Detailed architecture visualization
```

## Key Differences from Original GraphCast

1. **Input Data**: 
   - Original: Lat-lon grid (weather data)
   - Ours: Irregular surface mesh (automotive surfaces)

2. **Mesh Construction**:
   - Original: Icosahedral mesh hierarchy
   - Ours: k-NN based mesh from surface sampling

3. **Scale**:
   - Original: Global weather (~10⁶ grid points, 36 layers)
   - Ours: Vehicle surfaces (~10⁵ points, 12-16 layers)

4. **Dependencies**:
   - Original: PhysicsNemo framework
   - Ours: Standalone (PyTorch + PyG only)

## Parameter Scaling

The total parameters scale approximately as:

```
Total ≈ (hidden_dim² × num_processor_layers × 10) + (hidden_dim × input_dim) + (hidden_dim × output_dim)
```

For `hidden_dim=384, num_processor_layers=12`:
- Embedders: ~1.5M params
- Encoder: ~0.3M params
- Processor: ~14M params (dominant)
- Decoder: ~0.15M params
- Output: ~0.4K params
- **Total: ~3.0M params**

## Performance Tips

1. **Memory**: GraphCast can be memory-intensive. Use batch_size=1 and gradient checkpointing if needed.
2. **Speed**: k-NN graph construction is the bottleneck. Consider caching graphs.
3. **Convergence**: Use learning rate warmup and ReduceLROnPlateau scheduler.
4. **Regularization**: Weight decay (1e-5) and gradient clipping (max_norm=1.0) are important.

## References

- Lam, R., et al. (2022). "GraphCast: Learning skillful medium-range global weather forecasting." arXiv:2212.12794
- Nature 2023: https://www.nature.com/articles/s41586-023-06185-3
- PhysicsNemo implementation: https://github.com/NVIDIA/modulus

## Citation

If using this implementation, please cite:
```bibtex
@article{lam2022graphcast,
  title={GraphCast: Learning skillful medium-range global weather forecasting},
  author={Lam, Remi and others},
  journal={arXiv preprint arXiv:2212.12794},
  year={2022}
}
```

## License

Based on NVIDIA PhysicsNemo (Apache 2.0). This implementation is provided for research purposes.
