# MeshGraphNet for DrivAerNet Surface Pressure Prediction

Graph Neural Network for learning surface pressure fields on automotive geometries.

## Architecture

**Encode-Process-Decode** structure with message-passing graph neural networks:

```
Input Mesh → k-NN Graph → Encoder → Processor → Decoder → Pressure Predictions
```

### Components

1. **Graph Construction** (k-NN)
   - Nodes: Surface points with features `[x, y, z, n_x, n_y, n_z, area]` (7D)
   - Edges: k-nearest neighbors (typically k=6)
   - Edge features: `[dx, dy, dz, distance]` (4D)

2. **Encoder**
   - Node MLP: `7 → hidden_dim` (2 layers + LayerNorm)
   - Edge MLP: `4 → hidden_dim` (2 layers + LayerNorm)

3. **Processor** (Message-Passing Blocks × N)
   - Each block:
     - **Edge Update**: MLP(`[e_ij, h_i, h_j]` → `hidden_dim`) + Residual
     - **Node Update**: MLP(`[h_i, agg(e)]` → `hidden_dim`) + Residual
   - Standard: 15 blocks for benchmark
   - All blocks use LayerNorm

4. **Decoder**
   - Node MLP: `hidden_dim → 1` (2 layers, no LayerNorm)
   - Output: Pressure at each node

## Parameter Scaling

| Configuration | Processor Size | Hidden Dim | Parameters |
|---------------|----------------|------------|------------|
| Benchmark     | 15             | 128        | 2,340,609  |
| Medium        | 10             | 96         | 900,865    |
| Small         | 6              | 128        | 997,377    |
| Tiny          | 4              | 32         | 45,697     |

**Formula**:
```
Total ≈ Encoder + Processor × N + Decoder
Encoder ≈ 0.5 × hidden_dim²
Processor block ≈ 7.5 × hidden_dim²
Decoder ≈ 0.3 × hidden_dim²
```

## Usage

### Data Loading

```python
from data_loader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    data_dir='path/to/drivaeernet/surface_fields',
    batch_size=1,  # Typically 1 for variable-size graphs
    k_neighbors=6,
    normalize=True
)
```

### Model Creation

```python
from model import create_meshgraphnet

# Standard configuration (2.34M params)
model = create_meshgraphnet(
    input_dim_nodes=7,
    input_dim_edges=4,
    processor_size=15,
    hidden_dim=128
)

# Medium configuration (~1M params)
model = create_meshgraphnet(
    processor_size=6,
    hidden_dim=128
)
```

### Training

```python
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

criterion = nn.MSELoss()

for epoch in range(num_epochs):
    model.train()
    for data in train_loader:
        data = data.to(device)
        
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, data.y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Validation...
```

## Key Features

✅ **Graph-based representation**: Naturally handles irregular surface meshes  
✅ **Message passing**: Captures local interactions via edge updates  
✅ **Residual connections**: Every processing block has skip connections  
✅ **LayerNorm**: Stable training on physics data  
✅ **Scalable**: Adjustable processor size and hidden dimension  
✅ **k-NN graph**: Automatic neighborhood construction from point clouds

## File Structure

```
MeshGraphNet_SurfaceFields/
├── data_loader.py          # Dataset & graph construction
├── model.py                # MeshGraphNet architecture
├── train.py                # Training script
├── test_param_count.py     # Parameter counting utility
└── README.md               # This file
```

## Implementation Details

### MLP Structure
- **Input layer**: Linear → LayerNorm → ReLU
- **Hidden layers** (n-1): Linear → LayerNorm → ReLU
- **Output layer**: Linear
- **Residual**: Output = MLP(input) + input (when dims match)

### Edge Block
```
Input: [e_ij, h_i, h_j]  (edge_dim + 2 × node_dim)
    ↓ MLP
e'_ij = MLP(input) + e_ij  (residual)
```

### Node Block
```
agg_i = Σ_{j→i} e_ij  (sum aggregation)
Input: [h_i, agg_i]  (node_dim + edge_dim)
    ↓ MLP
h'_i = MLP(input) + h_i  (residual)
```

### Graph Construction
- **k-NN**: k=6 nearest neighbors (adjustable)
- **Edge features**: Relative position `[dx, dy, dz]` + Euclidean distance
- **Bidirectional**: Each pair (i,j) creates two directed edges

## Normalization

Features are standardized using training set statistics:
- **Position** (x, y, z): Zero mean, unit variance
- **Normals** (n_x, n_y, n_z): Zero mean, unit variance
- **Area**: Zero mean, unit variance
- **Pressure** (target): Zero mean, unit variance

Statistics saved in `normalization_stats.npz` for consistency across splits.

## Performance Tips

1. **Batch size**: Use 1 for variable-size graphs (typical for meshes)
2. **Gradient clipping**: Clip norm to 1.0 for stability
3. **Learning rate**: Start with 1e-3, use ReduceLROnPlateau
4. **k neighbors**: 6-8 works well for surface meshes
5. **Processor size**: 10-15 blocks for good accuracy
6. **Hidden dim**: 96-128 balances capacity and speed

## Computational Complexity

Per forward pass:
- **Encoder**: O(N × hidden_dim²)
- **Processor**: O(E × hidden_dim²) × processor_size
  - where E ≈ k × N (number of edges)
- **Decoder**: O(N × hidden_dim²)
- **Total**: O((k × processor_size) × N × hidden_dim²)

For N=50,000 points, k=6, processor_size=15, hidden_dim=128:
- Approximately 1.15 × 10⁹ FLOPs per forward pass
- Memory: ~2 GB (FP32) for single graph

## Reference

Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., & Battaglia, P. W. (2021).  
**Learning Mesh-Based Simulation with Graph Networks.**  
*International Conference on Machine Learning (ICML)*, pp. 7882-7893.

## Citation

```bibtex
@inproceedings{pfaff2021learning,
  title={Learning mesh-based simulation with graph networks},
  author={Pfaff, Tobias and Fortunato, Meire and Sanchez-Gonzalez, Alvaro and Battaglia, Peter W},
  booktitle={International Conference on Machine Learning},
  pages={7882--7893},
  year={2021},
  organization={PMLR}
}
```

## Notes

- This implementation uses PyTorch Geometric for graph operations
- Compatible with DrivAerNet surface field data (`.vtk` files)
- Supports both normalized and unnormalized training
- Automatically handles variable-size graphs in batching
