# GraphCast Data Loader Adaptation for DrivAerML

## Summary of Changes

The GraphCast data loader has been successfully adapted from DrivAerNet++ format to DrivAerML format, following the patterns from the XAeroNet production code.

## Key Adaptations

### 1. File Structure
**Before (DrivAerNet++):**
- Files: `{design_id}.npy` (NumPy arrays)
- IDs: String format like "DrivAer_F_D_WM_WW_0001"
- Dataset size: 8,150 samples

**After (DrivAerML):**
- Files: `run_{id}/boundary_{id}.vtp` (VTK PolyData)
- IDs: Integer format (1-500)
- Dataset size: 500 samples
- Additional: `geo_parameters_{id}.csv` (16 design variables)

### 2. Data Loading Method (Following XAeroNet Production Code)
**Before:**
```python
data_np = np.load(f"{design_id}.npy")
features = data_np[:, :7]  # [x,y,z,nx,ny,nz,area]
pressure = data_np[:, 7:8]  # [Cp]
```

**After (following XAeroNet preprocessor.py):**
```python
surf = pv.read(f"run_{run_id}/boundary_{run_id}.vtp")

# Step 1: Convert to triangular mesh
if surf.GetNumberOfCells() > 0:
    cell = surf.GetCell(0)
    if cell.GetNumberOfPoints() != 3:
        tet_filter = vtk.vtkDataSetTriangleFilter()
        tet_filter.SetInputData(surf)
        tet_filter.Update()
        surf = pv.wrap(tet_filter.GetOutput())

# Step 2: Convert cell_data to point_data (CRITICAL!)
surf = surf.cell_data_to_point_data()

# Step 3: Extract from point_data (not cell_data)
points = surf.points  # Mesh vertices, not cell centers
normals = surf.compute_normals(point_normals=True, cell_normals=False).point_data["Normals"]
area = surf.point_data.get("Area", np.ones(len(points)))

# Step 4: Extract pressure from point_data
# pMeanTrim = pressure at POINTS (after conversion)
# CpMeanTrim = pressure coefficient at CELLS (original)
pressure = surf.point_data["pMeanTrim"]  # Use pMeanTrim for point data
```

### 3. Critical Distinction: Cell Data vs Point Data

**Key Insight (provided by user):**
- **`CpMeanTrim`** = Pressure coefficient at **CELLS** (cell_data)
- **`pMeanTrim`** = Pressure at **POINTS** (point_data after cell_data_to_point_data)

The `C` prefix indicates **Cell** data, `p` prefix indicates **Point** data.

**XAeroNet Pattern (preprocessor.py:158-161):**
```python
surface_mesh = surface_mesh.cell_data_to_point_data()
node_attributes = surface_mesh.point_data
pressure_ref = node_attributes["pMeanTrim"]  # From point_data!
```

Our implementation follows this exactly:
1. Convert cell_data to point_data first
2. Extract pMeanTrim from point_data
3. Use mesh vertices (points), not cell centers
- **Geometry parameters**: 16 global design variables from CSV files
- **Robust field detection**: Handles both `CpMeanTrim` and `pMeanTrim`
- **Triangular mesh conversion**: Ensures proper mesh format
- **Explicit parameter specification**: Follows XAeroNet best practices

### 4. Data Structure
**PyG Data object now includes:**
- `x`: [num_nodes, 7] - features (x, y, z, nx, ny, nz, area)
- `pos`: [num_nodes, 3] - positions (x, y, z)  
- `y`: [num_nodes, 1] - target pressure
- `u`: [16] - global geometry parameters (**new**)
- `run_id`: int - run identifier (**changed from design_id**)

### 5. Split Files
**Before:**
- `train_design_ids.txt`, `val_design_ids.txt`, `test_design_ids.txt`
- Content: String design IDs

**After:**
- `train_run_ids.txt`, `val_run_ids.txt`, `test_run_ids.txt`
- Content: Integer run IDs (1-500)
- Split: 400 train / 50 val / 50 test

## Reference Code Sources

### Primary Reference: XAeroNet Production Code
File: `physicsnemo/examples/cfd/external_aerodynamics/xaeronet/surface/preprocessor.py`

Key patterns followed:
- Triangular mesh conversion
- Cell-based data extraction
- Robust field name handling
- Explicit parameter specification

### Secondary Reference: Training Notebooks
Files: `physicsnemo/example2/drivelml/Training_*.ipynb`

Confirmed:
- `CpMeanTrim` field name for DrivAerML
- Cell data (not point data) for raw loading
- KNN edge construction pattern
- Geometry parameters integration

## Dependencies
```python
import pyvista as pv  # VTP file reading
import vtk  # Mesh conversion
import pandas as pd  # CSV loading for geometry parameters
import torch
import torch_geometric
```

## Usage Example
```python
from GraphCast_SurfaceFields.data_loader import create_dataloaders, load_run_ids

# Load splits
train_ids, val_ids, test_ids = load_run_ids('train_val_test_splits')

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    data_dir='path/to/DrivAerML',  # Contains run_1, run_2, ..., run_500
    train_ids=train_ids,
    val_ids=val_ids,
    test_ids=test_ids,
    batch_size=1,
    normalize=True,
    verbose=True,
)

# Use in training
for batch in train_loader:
    data = batch[0]
    features = data.x  # [N, 7]
    pressure = data.y  # [N, 1]
    geo_params = data.u  # [16]
    # ... model forward pass
```

## Status
✅ **Complete** - GraphCast data loader fully adapted for DrivAerML
- Also works for FIGConvNet (shares same data loader)
- Handles 2 out of 8 model implementations

## Next Steps
Adapt remaining 6 data loaders:
1. Transolver++
2. Transolver
3. MeshGraphNet
4. RegDGCNN
5. NeuralOperator
6. ABUPT
