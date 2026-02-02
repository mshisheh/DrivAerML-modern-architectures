# DrivAerML Data Loader Adaptation - COMPLETE ✅

## Summary
Successfully adapted **ALL 8 model data loaders** from DrivAerNet++ format to DrivAerML format, following the XAeroNet preprocessor.py pattern for correct point-based data extraction.

**Date Completed:** February 2, 2026  
**Models Adapted:** 8/8 (100%)  
**Dataset:** DrivAerML (500 samples, VTP format)

---

## Core Pattern Applied (XAeroNet preprocessor.py)

All implementations follow this critical pattern from lines 155-161:

```python
# 1. Load VTP file
surf = pv.read(vtp_file)

# 2. Convert to triangular mesh
surf = convert_to_triangular_mesh(surf)

# 3. CRITICAL: Convert cell_data to point_data
surf = surf.cell_data_to_point_data()

# 4. Extract from point_data
points = surf.points  # Mesh vertices (not cell centers)
normals = surf.compute_normals(point_normals=True).point_data["Normals"]
pressure = surf.point_data["pMeanTrim"]  # From point_data!
```

**Key Insight:** The distinction between `CpMeanTrim` (cell data) and `pMeanTrim` (point data) is CRITICAL. After `cell_data_to_point_data()`, we extract from `point_data`.

---

## Completed Data Loaders

### 1. ✅ GraphCast + FIGConvNet
- **File:** `DrivAerML/GraphCast_SurfaceFields/data_loader.py`
- **Features:** Point positions + normals + area (7D)
- **Pattern:** VTP → triangulate → cell_data_to_point_data → extract
- **Output:** PyG Data with node features
- **Covers:** 2 models (GraphCast, FIGConvNet)

### 2. ✅ Transolver++
- **File:** `DrivAerML/Transolver++_SurfaceFields/data_loader.py`
- **Features:** 6D [x, y, z, nx, ny, nz]
- **Pattern:** XAeroNet + variable-sized point clouds
- **Output:** Dict with positions, normals, features, pressures
- **Special:** Custom collate for variable-sized batches

### 3. ✅ Transolver
- **File:** `DrivAerML/Transolver_SurfaceFields/data_loader.py`
- **Features:** 6D [x, y, z, nx, ny, nz]
- **Pattern:** Similar to Transolver++ but standard transformer
- **Output:** Dict with positions, normals, features, pressures

### 4. ✅ MeshGraphNet
- **File:** `DrivAerML/MeshGraphNet_SurfaceFields/data_loader.py`
- **Features:** 6D node features [x, y, z, nx, ny, nz]
- **Pattern:** XAeroNet + KNN graph construction (k=6)
- **Output:** PyG Data with nodes, edges, edge_attr
- **Special:** Edge features [dx, dy, dz, distance]

### 5. ✅ RegDGCNN
- **File:** `DrivAerML/RegDGCNN_SurfaceFields/data_loader.py`
- **Features:** 3D point cloud [x, y, z]
- **Pattern:** XAeroNet (no normals needed, dynamic graphs)
- **Output:** [1, 3, N] point cloud, [1, N] pressure
- **Special:** Format for dynamic graph CNN

### 6. ✅ NeuralOperator (FNO)
- **File:** `DrivAerML/NeuralOperator_SurfaceFields/data_loader.py`
- **Features:** 4-channel voxel grid [occupancy, x, y, z]
- **Pattern:** XAeroNet + voxelization (32³ or 64³ grid)
- **Output:** Dict with voxel_grid, positions, pressures
- **Special:** Converts point cloud to regular grid

### 7. ✅ ABUPT (Anchored-Branched)
- **File:** `DrivAerML/ABUPT_SurfaceFields/data_loader.py`
- **Features:** Positions + normals + optional WSS
- **Pattern:** XAeroNet + wall shear stress support
- **Output:** Dict with surface_position_vtp, surface_normals, surface_pressure
- **Special:** Optional wallShearStressMeanTrim loading

### 8. ✅ Train/Val/Test Splits
- **Files:** `DrivAerML/train_val_test_splits/*.txt`
- **Split:** 400 / 50 / 50 (80% / 10% / 10%)
- **IDs:** 1-400 (train), 401-450 (val), 451-500 (test)

---

## File Structure

```
DrivAerML/
├── GraphCast_SurfaceFields/
│   ├── data_loader.py (NEW - XAeroNet pattern)
│   └── data_loader_original.py (backup)
├── FIGConvNet_SurfaceFields/ (uses GraphCast loader)
├── Transolver++_SurfaceFields/
│   ├── data_loader.py (NEW - XAeroNet pattern)
│   └── data_loader_original.py (backup)
├── Transolver_SurfaceFields/
│   ├── data_loader.py (NEW - XAeroNet pattern)
│   └── data_loader_original.py (backup)
├── MeshGraphNet_SurfaceFields/
│   ├── data_loader.py (NEW - XAeroNet pattern + KNN)
│   └── data_loader_original.py (backup)
├── RegDGCNN_SurfaceFields/
│   ├── data_loader.py (NEW - XAeroNet pattern)
│   └── data_loader_original.py (backup)
├── NeuralOperator_SurfaceFields/
│   ├── data_loader.py (NEW - XAeroNet pattern + voxelization)
│   └── data_loader_original.py (backup)
├── ABUPT_SurfaceFields/
│   ├── data_loader.py (NEW - XAeroNet pattern + WSS)
│   └── data_loader_original.py (backup)
├── train_val_test_splits/
│   ├── train_run_ids.txt (400 samples)
│   ├── val_run_ids.txt (50 samples)
│   └── test_run_ids.txt (50 samples)
└── run_{1-500}/
    └── boundary_{1-500}.vtp
```

---

## Data Format

**Input:**
- VTP files: `run_{id}/boundary_{id}.vtp`
- Geometry parameters: `geo_parameters_{id}.csv` (16 design variables)

**Fields Used:**
- `pMeanTrim` - Pressure at points (PRIMARY, from point_data)
- `wallShearStressMeanTrim` - Wall shear stress at points (ABUPT only)
- `CpMeanTrim` - Pressure coefficient at cells (fallback)

**Critical:**
- ✅ Use `surf.cell_data_to_point_data()` BEFORE extraction
- ✅ Extract from `surf.point_data["pMeanTrim"]`
- ✅ Use `surf.points` (mesh vertices, not cell centers)
- ✅ Use point normals: `compute_normals(point_normals=True, cell_normals=False)`

---

## Verification Against XAeroNet

All implementations verified against:
- **Reference:** `physicsnemo/examples/cfd/external_aerodynamics/xaeronet/surface/preprocessor.py`
- **Lines:** 155-161 (critical pattern)
- **Key insight:** User guidance: "C is for cell and the other one is for points"

**Verification checklist:**
- ✅ Triangular mesh conversion
- ✅ cell_data_to_point_data() called
- ✅ Mesh vertices used (not cell centers)
- ✅ Point normals used (not cell normals)
- ✅ pMeanTrim priority over CpMeanTrim
- ✅ Extraction from point_data after conversion

---

## Key Differences from DrivAerNet++

| Aspect | DrivAerNet++ | DrivAerML |
|--------|--------------|-----------|
| Samples | 8,150 | 500 |
| Format | NumPy .npy | VTP .vtp |
| File structure | Flat directory | run_X/boundary_X.vtp |
| Preprocessing | Pre-processed | Raw VTP files |
| Field names | Generic "p" | "pMeanTrim" at points |
| Data source | Cell centers | Mesh vertices (after conversion) |
| Critical step | N/A | cell_data_to_point_data() |

---

## Next Steps

All data loaders are ready for training! To use:

```python
# Load splits
with open("train_val_test_splits/train_run_ids.txt") as f:
    train_ids = [int(line.strip()) for line in f]

# Create dataloaders
from data_loader import create_dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    data_dir="C:/Learning/Scientific/CARBENCH/DrivAerML",
    train_ids=train_ids,
    val_ids=val_ids,
    test_ids=test_ids,
    batch_size=8,
    num_workers=4,
    normalize=True,
)

# Train!
for batch in train_loader:
    # Model-specific batch processing
    pass
```

---

## Credits

- **XAeroNet Pattern:** `physicsnemo/examples/cfd/external_aerodynamics/xaeronet/surface/preprocessor.py`
- **Dataset:** DrivAerML (500 automotive surface meshes)
- **Models:** GraphCast, FIGConvNet, Transolver, Transolver++, MeshGraphNet, RegDGCNN, NeuralOperator, AB-UPT

**Implementation Date:** February 2, 2026  
**Status:** ✅ COMPLETE - All 8 models adapted and verified
