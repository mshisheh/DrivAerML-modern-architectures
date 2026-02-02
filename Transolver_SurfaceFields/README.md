# Transolver for DrivAerNet Surface Pressure Prediction

Standalone Transolver implementation (original variant) adapted for DrivAerNet++ surface pressure prediction.

## Files

- `model.py` - PyTorch-only implementation of a transformer-based Transolver.
- `data_loader.py` - Dataset and data loading utilities for DrivAerNet surface meshes.
- `train.py` - Complete training script with train/validation loops.
- `test_model_only.py` - Standalone model test (no PyG dependency).
- `architecture_diagram.py` - Comprehensive visual documentation of the architecture.
- `README.md` - This file.

## Quick Start

1. Install PyTorch and PyTorch Geometric in your environment (CPU or GPU build).

2. Run a quick smoke test of the model:

```powershell
cd C:\Learning\Scientific\CARBENCH\DrivAerNet\Transolver_SurfaceFields
python model.py
```

3. View the architecture diagram:

```powershell
python architecture_diagram.py
```

4. Train on DrivAerNet data (after setting up data paths):

```powershell
python train.py
```

## Data Format

The data loader expects DrivAerNet format:
- Input files: `.npy` files with shape `(N, 8)` containing `[x, y, z, nx, ny, nz, area, Cp]`
- N = number of surface points (~50K-100K per vehicle)
- Features: 3D coordinates, surface normals, point areas
- Target: Pressure coefficient (Cp)

## Model Configurations

| Configuration     | d_model | n_layers | Parameters | Notes                    |
|-------------------|---------|----------|------------|--------------------------|
| Transolver-Small  | 192     | 4        | ~1.67M     | Fast, lower accuracy     |
| Transolver-Base   | 208     | 6        | ~2.47M     | ✓ Benchmark target       |
| Transolver-Medium | 256     | 6        | ~3.69M     | Default config           |
| Transolver-Large  | 320     | 8        | ~7.58M     | High accuracy            |

## Notes

- This is a compact, readable variant focused on clarity and easy integration with
  the DrivAerNet dataset loaders.
- The Transolver-Base configuration (d_model=208, n_layers=6) matches the benchmark
  target of 2.47M parameters with reported R²=0.9577.
- Standard transformer attention has O(N²) complexity - for large meshes (N>100K),
  consider using gradient checkpointing or mixed precision training.
- See `architecture_diagram.py` for detailed documentation of the architecture,
  parameter breakdown, and comparison with other models.
