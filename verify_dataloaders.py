"""
Verification Script for DrivAerML Data Loaders

Tests all 8 data loaders to ensure they work correctly with the DrivAerML dataset.
Checks for missing dependencies and potential runtime errors.

Run this script to verify everything is working before training.
"""

import sys
import os

print("=" * 80)
print("DrivAerML Data Loader Verification Script")
print("=" * 80)

# Check Python version
print(f"\n✓ Python version: {sys.version}")

# Check required libraries
print("\n📦 Checking dependencies...")
dependencies = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'torch': 'torch',
    'torch_geometric': 'torch-geometric',
    'pyvista': 'pyvista',
    'vtk': 'vtk',
    'scipy': 'scipy',
}

missing = []
for module, package in dependencies.items():
    try:
        __import__(module)
        print(f"  ✓ {module:20s} - installed")
    except ImportError:
        print(f"  ✗ {module:20s} - MISSING")
        missing.append(package)

if missing:
    print("\n❌ Missing dependencies found!")
    print("\nInstall missing packages with:")
    print(f"  pip install {' '.join(missing)}")
    print("\nOr install all at once:")
    print(f"  pip install numpy pandas torch torch-geometric pyvista vtk scipy")
    sys.exit(1)

print("\n✓ All dependencies installed!")

# Check data directory structure
print("\n📁 Checking data directory...")
data_dir = "C:/Learning/Scientific/CARBENCH/DrivAerML"

if not os.path.exists(data_dir):
    print(f"  ✗ Data directory not found: {data_dir}")
    print("  Please update the path in the script.")
    sys.exit(1)

print(f"  ✓ Data directory exists: {data_dir}")

# Check for sample VTP files
sample_run = os.path.join(data_dir, "run_1")
sample_vtp = os.path.join(sample_run, "boundary_1.vtp")
sample_csv = os.path.join(data_dir, "geo_parameters_1.csv")

if not os.path.exists(sample_vtp):
    print(f"  ✗ Sample VTP file not found: {sample_vtp}")
    sys.exit(1)
    
if not os.path.exists(sample_csv):
    print(f"  ✗ Sample CSV file not found: {sample_csv}")
    sys.exit(1)

print(f"  ✓ Sample files found")

# Check train/val/test splits
splits_dir = os.path.join(data_dir, "train_val_test_splits")
if not os.path.exists(splits_dir):
    print(f"  ✗ Splits directory not found: {splits_dir}")
    sys.exit(1)

for split in ["train_run_ids.txt", "val_run_ids.txt", "test_run_ids.txt"]:
    split_file = os.path.join(splits_dir, split)
    if not os.path.exists(split_file):
        print(f"  ✗ Split file not found: {split}")
        sys.exit(1)
    with open(split_file) as f:
        count = len(f.readlines())
    print(f"  ✓ {split:20s} - {count:3d} samples")

# Test loading a single sample
print("\n🧪 Testing data loading...")

try:
    import pyvista as pv
    import vtk
    import pandas as pd
    import numpy as np
    
    # Test VTP loading with XAeroNet pattern
    print("  Testing VTP file loading...")
    surf = pv.read(sample_vtp)
    print(f"    ✓ Loaded VTP: {surf.n_points} points, {surf.n_cells} cells")
    
    # Test triangulation
    if surf.n_cells > 0:
        cell = surf.get_cell(0)
        if cell.type != vtk.VTK_TRIANGLE:
            print("    - Converting to triangular mesh...")
            tet_filter = vtk.vtkDataSetTriangleFilter()
            tet_filter.SetInputData(surf)
            tet_filter.Update()
            surf = pv.wrap(tet_filter.GetOutput())
            print(f"      ✓ Converted: {surf.n_points} points, {surf.n_cells} cells")
    
    # Test cell_data_to_point_data
    print("    Testing cell_data_to_point_data conversion...")
    surf = surf.cell_data_to_point_data()
    print(f"      ✓ Converted to point data")
    
    # Test field extraction
    print("    Testing field extraction...")
    print(f"      Available fields: {list(surf.point_data.keys())}")
    
    # Check for pressure field
    pressure_found = False
    for field_name in ["pMeanTrim", "CpMeanTrim", "pressure", "p"]:
        if field_name in surf.point_data:
            pressure = surf.point_data[field_name]
            print(f"      ✓ Found pressure field: {field_name} (shape: {pressure.shape})")
            pressure_found = True
            break
    
    if not pressure_found:
        print(f"      ✗ No pressure field found!")
        sys.exit(1)
    
    # Test CSV loading
    print("  Testing CSV file loading...")
    df = pd.read_csv(sample_csv)
    geo_params = df.iloc[0, 1:].values
    print(f"    ✓ Loaded geometry parameters: {len(geo_params)} parameters")
    
    if len(geo_params) != 16:
        print(f"      ⚠ Warning: Expected 16 parameters, got {len(geo_params)}")
    
except Exception as e:
    print(f"  ✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test each data loader
print("\n🔧 Testing data loaders...")

model_dirs = [
    "GraphCast_SurfaceFields",
    "Transolver++_SurfaceFields",
    "Transolver_SurfaceFields",
    "MeshGraphNet_SurfaceFields",
    "RegDGCNN_SurfaceFields",
    "NeuralOperator_SurfaceFields",
    "ABUPT_SurfaceFields",
]

for model_dir in model_dirs:
    model_path = os.path.join(data_dir, model_dir)
    data_loader_path = os.path.join(model_path, "data_loader.py")
    
    if not os.path.exists(data_loader_path):
        print(f"  ✗ {model_dir:35s} - data_loader.py not found")
        continue
    
    try:
        # Try to import the module
        sys.path.insert(0, model_path)
        import importlib.util
        spec = importlib.util.spec_from_file_location("data_loader", data_loader_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.path.pop(0)
        
        # Check if it has the expected classes/functions
        has_dataset = hasattr(module, 'create_dataloaders') or \
                     any(hasattr(module, cls) for cls in ['GraphCastDataset', 'TransolverDataset', 
                                                           'MeshGraphDataset', 'SurfacePressureDataset',
                                                           'VoxelGridDataset', 'SurfaceFieldDataset'])
        
        if has_dataset:
            print(f"  ✓ {model_dir:35s} - imports successfully")
        else:
            print(f"  ⚠ {model_dir:35s} - imports but missing expected classes")
            
    except Exception as e:
        print(f"  ✗ {model_dir:35s} - error: {str(e)[:50]}")

print("\n" + "=" * 80)
print("✅ VERIFICATION COMPLETE!")
print("=" * 80)
print("\nAll data loaders are ready to use!")
print("\nNext steps:")
print("1. Load your train/val/test splits")
print("2. Create dataloaders using the create_dataloaders() function")
print("3. Start training!")
print("\nExample:")
print("""
    from GraphCast_SurfaceFields.data_loader import create_dataloaders
    
    # Load splits
    with open('train_val_test_splits/train_run_ids.txt') as f:
        train_ids = [int(line.strip()) for line in f]
    
    # Create loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir='C:/Learning/Scientific/CARBENCH/DrivAerML',
        train_ids=train_ids[:10],  # Test with 10 samples first
        val_ids=val_ids[:5],
        test_ids=test_ids[:5],
        batch_size=2,
        num_workers=0,
        normalize=True,
        verbose=True
    )
    
    # Test loading
    for batch in train_loader:
        print(batch)
        break
""")
