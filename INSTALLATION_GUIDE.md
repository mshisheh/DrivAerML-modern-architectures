# DrivAerML Data Loaders - Installation & Usage Guide

## ✅ Status: Complete & Verified

All 8 model data loaders have been successfully adapted and verified for DrivAerML dataset.

---

## 📦 Required Dependencies

### Core Libraries
```bash
pip install numpy pandas scipy
```

### Deep Learning
```bash
pip install torch
pip install torch-geometric
```

### Visualization & Mesh Processing
```bash
pip install pyvista vtk
```

### All-in-One Installation
```bash
pip install numpy pandas torch torch-geometric pyvista vtk scipy
```

**Note:** For torch-geometric, you may need to install with specific PyTorch/CUDA versions:
```bash
# For PyTorch 2.0+ with CUDA 11.8
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# For CPU only
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

---

## 🔍 Verification

Run the verification script to check everything is working:

```bash
cd C:\Learning\Scientific\CARBENCH\DrivAerML
python verify_dataloaders.py
```

This will:
- ✅ Check all required dependencies are installed
- ✅ Verify data directory structure
- ✅ Test VTP file loading with XAeroNet pattern
- ✅ Validate train/val/test splits
- ✅ Test each data loader can be imported

---

## 🚀 Quick Start

### 1. Load Train/Val/Test Splits

```python
import os

data_dir = "C:/Learning/Scientific/CARBENCH/DrivAerML"

# Load run IDs
with open(os.path.join(data_dir, "train_val_test_splits/train_run_ids.txt")) as f:
    train_ids = [int(line.strip()) for line in f]

with open(os.path.join(data_dir, "train_val_test_splits/val_run_ids.txt")) as f:
    val_ids = [int(line.strip()) for line in f]

with open(os.path.join(data_dir, "train_val_test_splits/test_run_ids.txt")) as f:
    test_ids = [int(line.strip()) for line in f]

print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
# Output: Train: 400, Val: 50, Test: 50
```

### 2. Create Data Loaders

#### GraphCast / FIGConvNet

```python
import sys
sys.path.append("C:/Learning/Scientific/CARBENCH/DrivAerML/GraphCast_SurfaceFields")
from data_loader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    data_dir=data_dir,
    train_ids=train_ids,
    val_ids=val_ids,
    test_ids=test_ids,
    batch_size=1,
    num_workers=4,
    normalize=True,
    verbose=True
)

# Test loading
for data in train_loader:
    print(f"Features shape: {data.x.shape}")
    print(f"Target shape: {data.y.shape}")
    break
```

#### Transolver / Transolver++

```python
import sys
sys.path.append("C:/Learning/Scientific/CARBENCH/DrivAerML/Transolver++_SurfaceFields")
from data_loader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    data_dir=data_dir,
    train_ids=train_ids,
    val_ids=val_ids,
    test_ids=test_ids,
    batch_size=1,
    num_workers=4,
    normalize=True,
    verbose=True
)

# Test loading
for batch in train_loader:
    sample = batch[0]  # First item (variable-sized)
    print(f"Positions: {sample['positions'].shape}")
    print(f"Features: {sample['features'].shape}")
    print(f"Pressures: {sample['pressures'].shape}")
    break
```

#### MeshGraphNet

```python
import sys
sys.path.append("C:/Learning/Scientific/CARBENCH/DrivAerML/MeshGraphNet_SurfaceFields")
from data_loader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    data_dir=data_dir,
    train_ids=train_ids,
    val_ids=val_ids,
    test_ids=test_ids,
    k_neighbors=6,
    batch_size=2,
    num_workers=4,
    normalize=True,
    verbose=True
)

# Test loading
for batch in train_loader:
    print(f"Node features: {batch.x.shape}")
    print(f"Edge index: {batch.edge_index.shape}")
    print(f"Edge attr: {batch.edge_attr.shape}")
    break
```

#### RegDGCNN

```python
import sys
sys.path.append("C:/Learning/Scientific/CARBENCH/DrivAerML/RegDGCNN_SurfaceFields")
from data_loader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    data_dir=data_dir,
    train_ids=train_ids,
    val_ids=val_ids,
    test_ids=test_ids,
    num_points=5000,  # Sample 5000 points
    batch_size=8,
    num_workers=4,
    normalize=True,
    verbose=True
)

# Test loading
for point_cloud, pressure in train_loader:
    print(f"Point cloud: {point_cloud.shape}")  # [batch, 1, 3, N]
    print(f"Pressure: {pressure.shape}")  # [batch, 1, N]
    break
```

#### NeuralOperator (FNO)

```python
import sys
sys.path.append("C:/Learning/Scientific/CARBENCH/DrivAerML/NeuralOperator_SurfaceFields")
from data_loader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    data_dir=data_dir,
    train_ids=train_ids,
    val_ids=val_ids,
    test_ids=test_ids,
    grid_resolution=32,  # 32x32x32 voxel grid
    num_sample_points=5000,
    batch_size=4,
    num_workers=4,
    normalize=True,
    verbose=True
)

# Test loading
for batch in train_loader:
    print(f"Voxel grid: {batch['voxel_grid'].shape}")  # [B, 4, 32, 32, 32]
    print(f"Positions: {batch['positions'].shape}")
    print(f"Pressures: {batch['pressures'].shape}")
    break
```

#### ABUPT

```python
import sys
sys.path.append("C:/Learning/Scientific/CARBENCH/DrivAerML/ABUPT_SurfaceFields")
from data_loader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    data_dir=data_dir,
    train_ids=train_ids,
    val_ids=val_ids,
    test_ids=test_ids,
    num_points=5000,
    load_wss=True,  # Load wall shear stress
    batch_size=8,
    num_workers=4,
    normalize=True,
    verbose=True
)

# Test loading
for batch in train_loader:
    print(f"Positions: {batch['surface_position_vtp'].shape}")
    print(f"Normals: {batch['surface_normals'].shape}")
    print(f"Pressure: {batch['surface_pressure'].shape}")
    if 'surface_wallshearstress' in batch:
        print(f"WSS: {batch['surface_wallshearstress'].shape}")
    break
```

---

## 🔧 Bug Fixes Applied

### Fixed: Cell Type Check Safety

All data loaders now safely check for empty meshes before accessing cells:

```python
# ✅ FIXED - Safe cell check
if surf.n_cells > 0 and surf.get_cell(0).type != vtk.VTK_TRIANGLE:
    # Convert to triangular mesh
    ...

# ❌ OLD - Could crash on empty mesh
if surf.get_cell(0).type != vtk.VTK_TRIANGLE:
    ...
```

### Files Updated:
- ✅ Transolver_SurfaceFields/data_loader.py
- ✅ RegDGCNN_SurfaceFields/data_loader.py
- ✅ NeuralOperator_SurfaceFields/data_loader.py
- ✅ MeshGraphNet_SurfaceFields/data_loader.py
- ✅ ABUPT_SurfaceFields/data_loader.py

---

## 📊 Data Format Summary

### Input Files
- **VTP Files:** `run_{1-500}/boundary_{1-500}.vtp`
- **Geometry CSVs:** `geo_parameters_{1-500}.csv` (16 design variables)

### Fields Available
- `pMeanTrim` - Pressure at points (PRIMARY)
- `wallShearStressMeanTrim` - Wall shear stress at points
- `CpMeanTrim` - Pressure coefficient at cells (fallback)

### Train/Val/Test Split
- **Train:** 400 samples (IDs 1-400) - 80%
- **Val:** 50 samples (IDs 401-450) - 10%
- **Test:** 50 samples (IDs 451-500) - 10%

---

## 🎯 Model-Specific Output Formats

| Model | Output Format | Batch Size | Notes |
|-------|---------------|------------|-------|
| GraphCast | PyG Data | 1 (typical) | Node features [N, 7] |
| FIGConvNet | PyG Data | 1 (typical) | Same as GraphCast |
| Transolver++ | Dict | 1 (typical) | Variable-sized point clouds |
| Transolver | Dict | 1 (typical) | Variable-sized point clouds |
| MeshGraphNet | PyG Batch | Any | Supports batching |
| RegDGCNN | Tensors | Any | [B, 1, 3, N] format |
| NeuralOperator | Dict | Any | 4-channel voxel grids |
| ABUPT | Dict | Any | Positions + normals |

---

## ⚠️ Important Notes

1. **XAeroNet Pattern:** All loaders follow the critical pattern:
   - Load VTP → Triangulate → `cell_data_to_point_data()` → Extract

2. **Field Priority:** Loaders check fields in order:
   - Pressure: `pMeanTrim` > `CpMeanTrim` > `pressure` > `p`
   - WSS: `wallShearStressMeanTrim` > `wallShearStress` > `WSS` > `tau`

3. **Point vs Cell Data:** After `cell_data_to_point_data()`, extract from `surf.point_data`, not `surf.cell_data`

4. **Mesh Vertices:** Use `surf.points` (mesh vertices), not `surf.cell_centers().points`

5. **Point Normals:** Use `compute_normals(point_normals=True, cell_normals=False)`

---

## 🐛 Troubleshooting

### Import Error: torch_geometric

```bash
# Install with specific torch version
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Import Error: pyvista or vtk

```bash
pip install pyvista vtk
```

### File Not Found: VTP files

Check data directory path is correct:
```python
import os
data_dir = "C:/Learning/Scientific/CARBENCH/DrivAerML"
assert os.path.exists(data_dir), f"Data directory not found: {data_dir}"
```

### Memory Issues

Reduce batch size or number of workers:
```python
train_loader = create_dataloaders(
    ...,
    batch_size=1,  # Reduce from 8
    num_workers=0,  # Reduce from 4
)
```

---

## 📝 Next Steps

1. ✅ **Install dependencies** (see above)
2. ✅ **Run verification script** (`python verify_dataloaders.py`)
3. ✅ **Test loading a few samples** (use examples above)
4. 🚀 **Start training your models!**

---

## 📧 Support

If you encounter any issues:
1. Check the verification script output
2. Verify all dependencies are installed
3. Check data directory structure matches expected format
4. Review the error message for specific file/field names

All data loaders are production-ready and verified against the XAeroNet preprocessor pattern!
