# Using AB-UPT Architecture with DrivAerNet

This document provides a quick guide for using the **Anchored-Branched Universal Physics Transformers (AB-UPT)** architecture with the DrivAerNet dataset for aerodynamic drag coefficient prediction.

## Overview

The AB-UPT architecture has been adapted from the `anchored-branched-universal-physics-transformers` repository to work with DrivAerNet's point cloud data for predicting aerodynamic drag coefficients (Cd).

## Quick Start

### 1. File Structure

All AB-UPT related files are located in `DeepSurrogates/`:

```
DrivAerNet/DeepSurrogates/
├── DrivAerNetDataset_ABUPT.py       # Dataset adapter
├── abupt_collator_drivaeernet.py    # Collator for batching
├── abupt_model_drivaeernet.py       # Model wrapper
├── train_ABUPT.py                   # Training script
├── test_abupt.py                    # Test suite
├── compare_models.py                # Model comparison
└── README_ABUPT.md                  # Detailed documentation
```

### 2. Test Installation

Before training, run the test suite to verify everything works:

```bash
cd DeepSurrogates
python test_abupt.py
```

This will test:
- Dataset loading
- Collator functionality  
- Model initialization
- End-to-end pipeline

### 3. Train the Model

Basic training command:

```bash
# Quick test with lite model (recommended for first run)
python train_ABUPT.py --model_size lite --epochs 30

# Full training with base model
python train_ABUPT.py --model_size base --epochs 100

# High-accuracy training with large model
python train_ABUPT.py --model_size large --epochs 100
```

### 4. Compare Models

Compare AB-UPT with existing DrivAerNet models:

```bash
python compare_models.py
```

## Architecture Sizes

| Model | Parameters | Description |
|-------|-----------|-------------|
| **Lite** | ~2M | Fast training, good for experimentation |
| **Base** | ~8M | Balanced performance (recommended) |
| **Large** | ~30M | Best accuracy, requires more GPU memory |

## Key Features of AB-UPT

1. **Dual-Branch Architecture**: Separate processing for surface and volume information
2. **Supernode Pooling**: Hierarchical geometry encoding for efficiency
3. **Attention Mechanism**: Better captures long-range dependencies than GNN/PointNet
4. **Multi-Scale Processing**: Geometry encoding + anchor-based prediction

## Comparison with Existing Models

| Feature | GNN | RegPointNet | AB-UPT |
|---------|-----|-------------|--------|
| Input Type | Mesh graph | Point cloud | Point cloud |
| Architecture | Graph convolutions | Dynamic graphs | Transformer attention |
| Parameters | ~1M | ~2M | 2-30M |
| Training Speed | Fast | Fast | Moderate |
| Accuracy | Good | Good | Very Good* |
| Interpretability | Medium | Low | High (attention maps) |

*Expected based on architecture design; to be validated with training.

## Requirements

- PyTorch ≥ 2.0.0
- Access to `anchored-branched-universal-physics-transformers` repository
- DrivAerNet dataset with point clouds
- GPU with ≥ 16GB VRAM (for base/large models)

## Output & Checkpoints

Training saves to `DeepSurrogates/checkpoints_abupt/`:
- `best_model.pt` - Best model by validation loss
- `checkpoint_epoch_X.pt` - Periodic checkpoints
- `training_results.csv` - Training metrics

## Documentation

For detailed documentation, see:
- **DeepSurrogates/README_ABUPT.md** - Complete AB-UPT documentation
- **anchored-branched-universal-physics-transformers/README.md** - Original AB-UPT documentation

## Example Usage

```python
from DrivAerNetDataset_ABUPT import DrivAerNetABUPTDataset
from abupt_collator_drivaeernet import DrivAerNetABUPTCollator
from abupt_model_drivaeernet import ABUPTDragPredictor

# Load dataset
dataset = DrivAerNetABUPTDataset(
    root_dir='../DrivAerNet_FEN_Processed_Point_Clouds_100k',
    csv_file='../DrivAerNetPlusPlus_Cd_8k_Updated.csv',
)

# Create collator
collator = DrivAerNetABUPTCollator(dataset=dataset)

# Initialize model
model = ABUPTDragPredictor(model_size='base')

# Train (see train_ABUPT.py for complete training loop)
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `anchored-branched-universal-physics-transformers` is in parent directory
2. **Out of memory**: Use smaller model (`--model_size lite`) or reduce batch size
3. **Slow training**: Reduce number of anchor points in config
4. **Dataset not found**: Check paths in config match your directory structure

### Getting Help

- Run `python test_abupt.py` to diagnose issues
- Check `DeepSurrogates/README_ABUPT.md` for detailed troubleshooting
- Verify data paths in config match your setup

## Citation

If you use this implementation, please cite both DrivAerNet and AB-UPT papers (see README_ABUPT.md for citations).

## License

Follows licenses of both parent projects (DrivAerNet and AB-UPT).
