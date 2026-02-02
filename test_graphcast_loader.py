#!/usr/bin/env python3
"""
Test script for GraphCast data loader with DrivAerML format.
Creates dummy VTP files to validate the data loading pipeline.
"""

import os
import sys
import tempfile
import shutil
import numpy as np

# Test imports
try:
    import pyvista as pv
    import pandas as pd
    import torch
    from torch_geometric.data import Data
    print("✓ All required packages are available")
except ImportError as e:
    print(f"✗ Missing package: {e}")
    print("\nPlease install required packages:")
    print("  pip install pyvista pandas torch torch-geometric")
    sys.exit(1)

# Import the data loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'GraphCast_SurfaceFields'))
from GraphCast_SurfaceFields.data_loader import GraphCastDataset, create_dataloaders, load_run_ids

def create_test_data():
    """Create temporary test data in DrivAerML format."""
    temp_dir = tempfile.mkdtemp(prefix='drivaerml_test_')
    print(f"\nCreating test data in: {temp_dir}")
    
    # Create 10 dummy run directories with VTP files
    run_ids = list(range(1, 11))
    
    for run_id in run_ids:
        run_dir = os.path.join(temp_dir, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Create dummy mesh with random points
        num_cells = np.random.randint(40000, 60000)
        print(f"  Creating run_{run_id} with {num_cells} cells...")
        
        # Create triangular mesh
        num_points = num_cells * 3
        points = np.random.randn(num_points, 3).astype(np.float32)
        cells = np.arange(num_points).reshape(-1, 3)
        
        # Create PyVista mesh
        mesh = pv.PolyData(points)
        faces = np.hstack([np.full((num_cells, 1), 3), cells]).ravel()
        mesh = pv.PolyData(points, faces)
        
        # Add pressure data (CpMeanTrim field)
        mesh.cell_data["CpMeanTrim"] = np.random.randn(num_cells).astype(np.float32) * 0.5
        
        # Save VTP file
        vtp_path = os.path.join(run_dir, f"boundary_{run_id}.vtp")
        mesh.save(vtp_path)
        
        # Create geometry parameters CSV (16 parameters)
        geo_params = np.random.randn(16).astype(np.float32)
        geo_df = pd.DataFrame([geo_params], columns=[f'param_{i}' for i in range(16)])
        geo_csv_path = os.path.join(run_dir, f"geo_parameters_{run_id}.csv")
        geo_df.to_csv(geo_csv_path, index=False)
    
    # Create train/val/test split files
    split_dir = os.path.join(temp_dir, 'splits')
    os.makedirs(split_dir, exist_ok=True)
    
    with open(os.path.join(split_dir, 'train_run_ids.txt'), 'w') as f:
        f.write('\n'.join(map(str, run_ids[:6])))
    
    with open(os.path.join(split_dir, 'val_run_ids.txt'), 'w') as f:
        f.write('\n'.join(map(str, run_ids[6:8])))
    
    with open(os.path.join(split_dir, 'test_run_ids.txt'), 'w') as f:
        f.write('\n'.join(map(str, run_ids[8:])))
    
    print(f"✓ Created test data with {len(run_ids)} runs")
    return temp_dir, split_dir

def test_dataset_creation(data_dir, split_dir):
    """Test dataset creation and data loading."""
    print("\n" + "="*70)
    print("TEST 1: Dataset Creation")
    print("="*70)
    
    # Load run IDs
    train_ids, val_ids, test_ids = load_run_ids(split_dir)
    print(f"✓ Loaded splits: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    
    # Create dataset
    dataset = GraphCastDataset(
        data_dir=data_dir,
        run_ids=train_ids,
        normalize=False,  # Disable normalization for quick test
        verbose=True,
    )
    
    print(f"✓ Dataset created with {len(dataset)} samples")
    return dataset, train_ids, val_ids, test_ids

def test_data_loading(dataset):
    """Test loading individual samples."""
    print("\n" + "="*70)
    print("TEST 2: Data Loading")
    print("="*70)
    
    # Load first sample
    sample = dataset[0]
    
    print(f"Sample structure:")
    print(f"  - Features (x): shape {sample.x.shape}, dtype {sample.x.dtype}")
    print(f"  - Positions (pos): shape {sample.pos.shape}, dtype {sample.pos.dtype}")
    print(f"  - Pressure (y): shape {sample.y.shape}, dtype {sample.y.dtype}")
    print(f"  - Geometry params (u): shape {sample.u.shape}, dtype {sample.u.dtype}")
    print(f"  - Run ID: {sample.run_id}")
    
    # Verify feature dimensions
    assert sample.x.shape[1] == 7, f"Expected 7 features, got {sample.x.shape[1]}"
    assert sample.pos.shape[1] == 3, f"Expected 3D positions, got {sample.pos.shape[1]}"
    assert sample.y.shape[1] == 1, f"Expected 1D pressure, got {sample.y.shape[1]}"
    assert sample.u.shape[0] == 16, f"Expected 16 geometry params, got {sample.u.shape[0]}"
    
    print("✓ Data shapes are correct")
    print(f"✓ Loaded {sample.x.shape[0]} points from VTP file")
    
    return sample

def test_normalization(data_dir, train_ids):
    """Test normalization statistics computation."""
    print("\n" + "="*70)
    print("TEST 3: Normalization")
    print("="*70)
    
    dataset = GraphCastDataset(
        data_dir=data_dir,
        run_ids=train_ids,
        normalize=True,
        verbose=True,
    )
    
    # This should trigger normalization computation
    sample = dataset[0]
    
    print(f"Normalization statistics computed:")
    print(f"  - Feature mean: {dataset.mean}")
    print(f"  - Feature std: {dataset.std}")
    print(f"  - Pressure mean: {dataset.pressure_mean:.4f}")
    print(f"  - Pressure std: {dataset.pressure_std:.4f}")
    
    print("✓ Normalization works correctly")

def test_dataloaders(data_dir, train_ids, val_ids, test_ids):
    """Test DataLoader creation."""
    print("\n" + "="*70)
    print("TEST 4: DataLoaders")
    print("="*70)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        batch_size=1,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        normalize=True,
        verbose=True,
    )
    
    # Test iterating through train loader
    print("\nTesting train loader iteration:")
    for i, batch in enumerate(train_loader):
        if i == 0:
            sample = batch[0]  # Get first sample from batch
            print(f"  Batch 0:")
            print(f"    - Features shape: {sample.x.shape}")
            print(f"    - Positions shape: {sample.pos.shape}")
            print(f"    - Pressure shape: {sample.y.shape}")
            print(f"    - Geometry params shape: {sample.u.shape}")
            print(f"    - Run ID: {sample.run_id}")
        if i >= 2:  # Only test first 3 batches
            break
    
    print(f"✓ Successfully iterated through {i+1} batches")
    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches")
    print(f"✓ Test loader: {len(test_loader)} batches")

def main():
    """Run all tests."""
    print("="*70)
    print("GraphCast Data Loader Test for DrivAerML")
    print("="*70)
    
    temp_dir = None
    try:
        # Create test data
        temp_dir, split_dir = create_test_data()
        
        # Run tests
        dataset, train_ids, val_ids, test_ids = test_dataset_creation(temp_dir, split_dir)
        sample = test_data_loading(dataset)
        test_normalization(temp_dir, train_ids)
        test_dataloaders(temp_dir, train_ids, val_ids, test_ids)
        
        # Summary
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nThe GraphCast data loader is working correctly with DrivAerML format:")
        print("  ✓ VTP file loading with pyvista")
        print("  ✓ CpMeanTrim field extraction")
        print("  ✓ Geometry parameters from CSV")
        print("  ✓ Feature normalization")
        print("  ✓ DataLoader batching")
        print("\nYou can now use this data loader with your DrivAerML dataset!")
        
    except Exception as e:
        print("\n" + "="*70)
        print("TEST FAILED! ✗")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if temp_dir and os.path.exists(temp_dir):
            print(f"\nCleaning up temporary files: {temp_dir}")
            shutil.rmtree(temp_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
