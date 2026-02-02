# AB-UPT for Surface Field Prediction on DrivAerNet++

This directory contains an implementation of **AB-UPT (Anchored-Branched Universal Physics Transformers)** for predicting surface pressure and wall shear stress fields on car geometries from the DrivAerNet++ dataset.

## Overview

Unlike the scalar drag coefficient prediction task, this implementation uses AB-UPT for its **original purpose**: **field prediction**. The model predicts pressure and wall shear stress values at each point on the car surface.

### What This Predicts

- **Surface Pressure Field**: Scalar field (1D) at each surface point
- **Surface Wall Shear Stress Field** (optional): Vector field (3D) at each surface point

This is the perfect match for AB-UPT's architecture, which was designed for predicting field quantities from geometry.

## Architecture

```
Input: Surface Mesh from VTK files
    ↓
┌──────────────────────────────────────────────────┐
│ 1. Geometry Encoding                             │
│    - Sample geometry points from surface         │
│    - Supernode pooling for hierarchical features │
│    - Geometry transformer blocks                 │
└──────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────┐
│ 2. Surface Branch Processing                     │
│    - Surface anchor points                       │
│    - Cross-attention to geometry                 │
│    - Surface-specific transformer blocks         │
└──────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────┐
│ 3. Field Prediction                              │
│    - Pressure decoder: 1D output per point       │
│    - WSS decoder: 3D output per point (optional) │
└──────────────────────────────────────────────────┘
    ↓
Output: Pressure & WSS fields at query points
```

## Comparison with RegDGCNN

| Aspect | RegDGCNN | AB-UPT |
|--------|----------|--------|
| **Architecture** | Graph CNN with k-NN | Transformer with attention |
| **Input** | Point cloud (N, 3) | Point cloud (N, 3) |
| **Output** | Pressure field (N, 1) | Pressure (N, 1) + WSS (N, 3) |
| **Parameters** | ~5M | 8-30M (configurable) |
| **Training Time** | ~5-8 min/epoch | ~15-30 min/epoch |
| **Key Advantage** | Fast, efficient | Better long-range dependencies |

## Files

- **`data_loader.py`**: Dataset and dataloader for VTK surface mesh files
- **`collator.py`**: Batch collation for AB-UPT format
- **`model.py`**: AB-UPT model wrapper for field prediction
- **`train.py`**: Training script
- **`README.md`**: This file

## Installation

### Requirements

```bash
# Core dependencies
torch>=2.0.0
numpy
pyvista  # For loading VTK files
tqdm

# AB-UPT repository should be in parent directory
# C:\Learning\Scientific\CARBENCH\anchored-branched-universal-physics-transformers\
```

## Usage

### Quick Start

```bash
cd C:\Learning\Scientific\CARBENCH\DrivAerNet\ABUPT_SurfaceFields

# Train with lite model (quick test)
python train.py --model_size lite --epochs 30

# Train with base model (recommended)
python train.py --model_size base --epochs 150

# Train with WSS prediction
python train.py --model_size base --epochs 150 --predict_wss
```

### Full Training Command

```bash
python train.py \
    --model_size base \
    --epochs 150 \
    --batch_size 2 \
    --lr 1e-4 \
    --dataset_path ../PressureVTK \
    --predict_wss
```

### Configuration

Edit the `config` dictionary in `train.py`:

```python
config = {
    # Model architecture
    'model_size': 'base',          # 'lite', 'base', or 'large'
    'dim': 256,                    # Hidden dimension
    'geometry_depth': 2,           # Geometry encoding depth
    'num_surface_blocks': 6,       # Surface-specific blocks
    'predict_wss': False,          # Predict wall shear stress
    
    # Point sampling
    'num_geometry_points': 8192,   # For geometry encoding
    'num_surface_anchors': 4096,   # For field prediction
    'num_geometry_supernodes': 512,
    
    # Training
    'batch_size': 2,
    'epochs': 150,
    'lr': 1e-4,
    
    # Data
    'dataset_path': '../PressureVTK',
    'num_points': 10000,           # Points sampled from each VTK
}
```

## Data Format

### Input VTK Files

VTK files should contain:
- **Points**: Surface mesh vertices (N, 3)
- **Point Data**: 
  - `'p'` or `'pressure'`: Pressure values at each point
  - `'wallShearStress'` or `'WSS'`: Wall shear stress vectors (optional)

### Dataset Structure

```
PressureVTK/
├── Design_0001.vtk
├── Design_0002.vtk
└── ...

train_val_test_splits/
├── train_design_ids.txt
├── val_design_ids.txt
└── test_design_ids.txt
```

## Model Sizes

| Model | Parameters | Geometry Depth | Surface Blocks | Best For |
|-------|-----------|----------------|----------------|----------|
| **Lite** | ~2M | 1 | 2 | Quick experiments |
| **Base** | ~8M | 2 | 6 | Production use |
| **Large** | ~30M | 3 | 8 | Best accuracy |

## Training

### Expected Training Time (RTX 3090)

| Model | Time/Epoch | 150 Epochs |
|-------|-----------|------------|
| Lite | ~8-10 min | ~20 hours |
| Base | ~20-25 min | ~50 hours |
| Large | ~40-50 min | ~100 hours |

### Evaluation Metrics

The training script computes:
- **MSE**: Mean squared error
- **MAE**: Mean absolute error
- **R²**: Coefficient of determination
- **Relative Error**: Mean relative error

### Checkpoints

Saved to `./experiments/{exp_name}/`:
- `best_model.pth`: Best model by validation loss
- `checkpoint_epoch_X.pth`: Periodic checkpoints

## Advanced Usage

### Inference with Query Points

The model supports two modes:

1. **Training mode**: Predict at anchor positions
2. **Inference mode**: Predict at arbitrary query positions

```python
from model import ABUPTSurfaceFieldPredictor
import torch

model = ABUPTSurfaceFieldPredictor()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Batch with query positions
batch = {
    'geometry_position': ...,
    'geometry_supernode_idx': ...,
    'surface_anchor_position': ...,
    'surface_query_position': query_points,  # (B, M, 3)
}

with torch.no_grad():
    outputs = model(batch)
    predicted_pressure = outputs['surface_query_pressure']  # (B, M, 1)
```

### Visualizing Predictions

```python
import pyvista as pv

# Load mesh
mesh = pv.read('Design_0001.vtk')

# Get predictions
predicted_pressure = model.predict(mesh.points)

# Visualize
mesh['predicted_pressure'] = predicted_pressure.squeeze()
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='predicted_pressure', cmap='jet')
plotter.show()
```

## Key Differences from RegDGCNN Implementation

1. **Architecture**: Transformer-based vs Graph CNN
2. **Multi-field Output**: Can predict both pressure and WSS simultaneously
3. **Attention Mechanism**: Can identify which geometry regions influence each point
4. **Anchor-Query Design**: Flexible prediction at arbitrary points
5. **Hierarchical Encoding**: Supernodes for multi-scale features

## Performance Expectations

Based on similar architectures, expected performance:

| Model | Pressure R² | WSS R² | Training Time |
|-------|------------|--------|---------------|
| Lite | 0.90-0.92 | 0.85-0.88 | Fast |
| Base | 0.93-0.95 | 0.88-0.91 | Moderate |
| Large | 0.95-0.97 | 0.90-0.93 | Slow |

*Note: Actual performance to be validated with training.*

## Advantages of AB-UPT for Field Prediction

1. **Native Field Prediction**: Designed for this task from the ground up
2. **Multi-scale Features**: Hierarchical geometry encoding
3. **Attention Maps**: Interpretable - see which geometry affects each point
4. **Flexible Queries**: Predict at arbitrary points, not just training points
5. **Multi-field Output**: Predict multiple fields (pressure, WSS) jointly
6. **Long-range Dependencies**: Better than k-NN based methods

## Troubleshooting

### Common Issues

1. **VTK Loading Errors**
   - Check that VTK files contain 'p' or 'pressure' in point_data
   - Verify file paths are correct

2. **Out of Memory**
   - Reduce batch_size to 1
   - Use 'lite' model
   - Reduce num_surface_anchors

3. **Import Errors**
   - Ensure AB-UPT repo is in: `../../anchored-branched-universal-physics-transformers/`
   - Check Python path includes AB-UPT src directory

4. **Slow Data Loading**
   - Set `preprocess=True` and use caching
   - Cached data will be reused in subsequent runs
   - Reduce num_workers if CPU bottleneck

## Citation

If you use this implementation, please cite:

**DrivAerNet++:**
```bibtex
@article{drivaeernet,
  title={DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks},
  author={Elrefaie, Mohamed and others},
  year={2024}
}
```

**AB-UPT:**
```bibtex
@article{abupt,
  title={Anchored-Branched Universal Physics Transformers},
  author={[Authors]},
  year={2024}
}
```

## Future Improvements

- [ ] Add distributed training support
- [ ] Implement gradient checkpointing for larger models
- [ ] Add visualization tools for predictions
- [ ] Support for additional field quantities (velocity, vorticity)
- [ ] Mixed precision training (FP16)
- [ ] Progressive training (coarse to fine)

## License

Follows the licenses of both DrivAerNet++ and AB-UPT projects.

## Contact

For issues or questions:
- Check AB-UPT repository for architecture questions
- Check DrivAerNet repository for dataset questions
- Verify VTK file format matches expected structure
