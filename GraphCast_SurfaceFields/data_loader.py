"""
Data Loader for GraphCast on DrivAerML

Prepares surface mesh data from DrivAerML VTP files for GraphCast training.
Handles VTP loading, normalization, and creates PyG Data objects.

Author: Implementation for DrivAerML benchmark
Date: February 2026
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from typing import List, Tuple, Optional

try:
    import pyvista as pv
    import vtk
except ImportError:
    raise ImportError(
        "pyvista and vtk are required for loading DrivAerML VTP files. "
        "Install with: pip install pyvista vtk"
    )


class GraphCastDataset(Dataset):
    """
    Dataset for GraphCast surface pressure prediction on DrivAerML.
    
    DrivAerML Data structure (following XAeroNet preprocessor.py pattern):
        - VTP files: run_{id}/boundary_{id}.vtp
        - Data extraction: cell_data -> point_data conversion
        - Point coordinates: (x, y, z) at mesh vertices
        - Point normals: (nx, ny, nz) computed at vertices
        - Point areas: Interpolated from cells to points
        - Target: pMeanTrim (pressure at points, after cell_data_to_point_data)
        - Geometry parameters: geo_parameters_{id}.csv (16 design variables)
    
    Note: 
        - CpMeanTrim = pressure coefficient at **cells**
        - pMeanTrim = pressure at **points** (after cell_data_to_point_data)
        - Following XAeroNet production code pattern, we use point data
    """
    
    def __init__(
        self,
        data_dir: str,
        run_ids: List[int],
        normalize: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            data_dir: Root directory containing DrivAerML data (with run_* folders)
            run_ids: List of run IDs (integers 1-500)
            normalize: Whether to normalize features
            verbose: Print loading information
        """
        self.data_dir = data_dir
        self.run_ids = run_ids
        self.normalize = normalize
        self.verbose = verbose
        
        # Normalization statistics (computed on first pass)
        self.mean = None
        self.std = None
        self.pressure_mean = None
        self.pressure_std = None
        
        # Verify data exists
        self._verify_data()
        
        if self.verbose:
            print(f"GraphCastDataset initialized with {len(self.run_ids)} runs from DrivAerML")
    
    def _verify_data(self):
        """Check that VTP files exist for DrivAerML"""
        missing = []
        for run_id in self.run_ids[:min(5, len(self.run_ids))]:
            vtp_file = os.path.join(self.data_dir, f"run_{run_id}", f"boundary_{run_id}.vtp")
            if not os.path.exists(vtp_file):
                missing.append(run_id)
        
        if missing:
            raise FileNotFoundError(
                f"Missing VTP files for runs: {missing}\n"
                f"Expected location: {self.data_dir}/run_{{id}}/boundary_{{id}}.vtp"
            )
    
    def compute_normalization_stats(self):
        """Compute mean and std for normalization from DrivAerML VTP files"""
        if self.verbose:
            print("Computing normalization statistics from DrivAerML data...")
        
        all_features = []
        all_pressures = []
        
        # Sample subset for stats (max 100 runs)
        sample_runs = self.run_ids[:min(100, len(self.run_ids))]
        
        for run_id in sample_runs:
            vtp_file = os.path.join(self.data_dir, f"run_{run_id}", f"boundary_{run_id}.vtp")
            
            # Load VTP file
            surf = pv.read(vtp_file)
            
            # Following XAeroNet preprocessor.py pattern
            # Step 1: Convert to triangular mesh if needed
            if surf.GetNumberOfCells() > 0:
                cell = surf.GetCell(0)
                if cell.GetNumberOfPoints() != 3:
                    tet_filter = vtk.vtkDataSetTriangleFilter()
                    tet_filter.SetInputData(surf)
                    tet_filter.Update()
                    surf = pv.wrap(tet_filter.GetOutput())
            
            # Step 2: Convert cell data to point data
            surf = surf.cell_data_to_point_data()
            
            # Step 3: Extract features from point data
            points = surf.points  # (N, 3) - xyz
            surf_n = surf.compute_normals(point_normals=True, cell_normals=False)
            normals = surf_n.point_data["Normals"]  # (N, 3) - nx, ny, nz
            
            # Get area at points (if available after conversion)
            if "Area" in surf.point_data:
                area = surf.point_data["Area"]
            else:
                area = np.ones(len(points))
            
            if len(area.shape) == 1:
                area = area[:, None]  # (N, 1)
            
            # Combine features: [x, y, z, nx, ny, nz, area]
            features = np.concatenate([points, normals, area], axis=1)  # (N, 7)
            
            # Extract pressure from point_data - use pMeanTrim (point data)
            pressure_field = None
            for field_name in ["pMeanTrim", "CpMeanTrim", "pressure", "p"]:
                if field_name in surf.point_data:
                    pressure_field = field_name
                    break
            
            if pressure_field is None:
                continue  # Skip this file if no pressure field found
            
            pressure = surf.point_data[pressure_field]
            if len(pressure.shape) == 1:
                pressure = pressure[:, None]  # (N, 1)
            
            all_features.append(features)
            all_pressures.append(pressure)
        
        all_features = np.concatenate(all_features, axis=0)
        all_pressures = np.concatenate(all_pressures, axis=0)
        
        self.mean = torch.tensor(all_features.mean(axis=0), dtype=torch.float32)
        self.std = torch.tensor(all_features.std(axis=0) + 1e-8, dtype=torch.float32)
        self.pressure_mean = torch.tensor(all_pressures.mean(), dtype=torch.float32)
        self.pressure_std = torch.tensor(all_pressures.std() + 1e-8, dtype=torch.float32)
        
        if self.verbose:
            print(f"Feature mean: {self.mean}")
            print(f"Feature std: {self.std}")
            print(f"Pressure mean: {self.pressure_mean:.4f}, std: {self.pressure_std:.4f}")
    
    def __len__(self) -> int:
        return len(self.run_ids)
    
    def __getitem__(self, idx: int) -> Data:
        """
        Load and return a single sample from DrivAerML VTP files.
        
        Returns:
            PyG Data object with:
                - x: [num_nodes, 7] - features (x, y, z, nx, ny, nz, area)
                - pos: [num_nodes, 3] - positions (x, y, z)
                - y: [num_nodes, 1] - target pressure (CpMeanTrim)
                - u: [16] - global geometry parameters (optional)
                - run_id: int - run identifier
        """
        run_id = self.run_ids[idx]
        vtp_file = os.path.join(self.data_dir, f"run_{run_id}", f"boundary_{run_id}.vtp")
        
        # Load VTP file
        surf = pv.read(vtp_file)
        
        # Following XAeroNet preprocessor.py pattern:
        # 1. Convert to triangular mesh
        # 2. Convert cell_data to point_data  
        # 3. Extract from point_data (not cell_data)
        
        # Step 1: Convert to triangular mesh if needed
        if surf.GetNumberOfCells() > 0:
            cell = surf.GetCell(0)
            if cell.GetNumberOfPoints() != 3:
                tet_filter = vtk.vtkDataSetTriangleFilter()
                tet_filter.SetInputData(surf)
                tet_filter.Update()
                surf = pv.wrap(tet_filter.GetOutput())
        
        # Step 2: Convert cell data to point data (BEFORE extracting anything)
        surf = surf.cell_data_to_point_data()
        
        # Step 3: Extract features from point data
        points = surf.points  # (N, 3) - xyz coordinates at mesh vertices
        
        # Compute normals at points
        surf_n = surf.compute_normals(point_normals=True, cell_normals=False)
        normals = surf_n.point_data["Normals"]  # (N, 3) - nx, ny, nz
        
        # For area: XAeroNet uses Tessellation.sample_boundary which provides area
        # Here we approximate by using point-based area (if available after conversion)
        # Otherwise use a placeholder since area is less critical for point-based methods
        if "Area" in surf.point_data:
            area = surf.point_data["Area"]
        else:
            # Use uniform area as fallback
            area = np.ones(len(points))
        
        if len(area.shape) == 1:
            area = area[:, None]  # (N, 1)
        
        # Combine features: [x, y, z, nx, ny, nz, area]
        features = np.concatenate([points, normals, area], axis=1)  # (N, 7)
        
        # Extract pressure from point_data
        # pMeanTrim = pressure at points (after cell_data_to_point_data)
        pressure_field = None
        for field_name in ["pMeanTrim", "CpMeanTrim", "pressure", "p"]:
            if field_name in surf.point_data:
                pressure_field = field_name
                break
        
        if pressure_field is None:
            raise ValueError(
                f"No pressure field found in point_data. Available fields: {list(surf.point_data.keys())}"
            )
        
        pressure = surf.point_data[pressure_field]
        if len(pressure.shape) == 1:
            pressure = pressure[:, None]  # (N, 1)
        
        # Convert to tensors
        features = torch.tensor(features, dtype=torch.float32)
        pressure = torch.tensor(pressure, dtype=torch.float32)
        positions = features[:, :3]  # x, y, z
        
        # Normalize
        if self.normalize:
            if self.mean is None:
                self.compute_normalization_stats()
            
            features = (features - self.mean) / self.std
            pressure = (pressure - self.pressure_mean) / self.pressure_std
        
        # Load geometry parameters (16 global design variables)
        geo_file = os.path.join(self.data_dir, f"run_{run_id}", f"geo_parameters_{run_id}.csv")
        if os.path.exists(geo_file):
            geo_df = pd.read_csv(geo_file)
            # Extract the 16 geometry parameter values
            geo_params = geo_df.iloc[0].values.astype(np.float32)
            geo_params = torch.tensor(geo_params, dtype=torch.float32)
        else:
            # If no geometry file, use zeros
            geo_params = torch.zeros(16, dtype=torch.float32)
        
        # Create PyG Data object
        data = Data(
            x=features,
            pos=positions,  # Original positions (not normalized)
            y=pressure,
            u=geo_params,  # Global geometry parameters
        )
        data.run_id = run_id
        
        return data


def collate_fn(batch: List[Data]) -> List[Data]:
    """
    Custom collate function for GraphCast.
    GraphCast processes each sample independently (no batching across samples).
    
    Args:
        batch: List of Data objects
    
    Returns:
        Same list (no collation)
    """
    return batch


def create_dataloaders(
    data_dir: str,
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    batch_size: int = 1,  # GraphCast typically uses batch_size=1
    num_workers: int = 4,
    normalize: bool = True,
    verbose: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for DrivAerML.
    
    Args:
        data_dir: Root directory containing DrivAerML data (with run_* folders)
        train_ids: List of training run IDs (integers)
        val_ids: List of validation run IDs (integers)
        test_ids: List of test run IDs (integers)
        batch_size: Batch size (typically 1 for GraphCast)
        num_workers: Number of data loading workers
        normalize: Whether to normalize features
        verbose: Print information
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = GraphCastDataset(
        data_dir=data_dir,
        run_ids=train_ids,
        normalize=normalize,
        verbose=verbose,
    )
    
    val_dataset = GraphCastDataset(
        data_dir=data_dir,
        run_ids=val_ids,
        normalize=normalize,
        verbose=verbose,
    )
    
    test_dataset = GraphCastDataset(
        data_dir=data_dir,
        run_ids=test_ids,
        normalize=normalize,
        verbose=verbose,
    )
    
    # Compute normalization stats from training data
    if normalize:
        train_dataset.compute_normalization_stats()
        # Share stats with val and test
        val_dataset.mean = train_dataset.mean
        val_dataset.std = train_dataset.std
        val_dataset.pressure_mean = train_dataset.pressure_mean
        val_dataset.pressure_std = train_dataset.pressure_std
        test_dataset.mean = train_dataset.mean
        test_dataset.std = train_dataset.std
        test_dataset.pressure_mean = train_dataset.pressure_mean
        test_dataset.pressure_std = train_dataset.pressure_std
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    if verbose:
        print(f"\nDataLoader Statistics:")
        print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
        print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def load_run_ids(split_dir: str) -> Tuple[List[int], List[int], List[int]]:
    """
    Load train/val/test run IDs from text files.
    
    Args:
        split_dir: Directory containing train/val/test split files
    
    Returns:
        train_ids, val_ids, test_ids (as integer lists)
    """
    def read_ids(filename):
        with open(os.path.join(split_dir, filename), 'r') as f:
            return [int(line.strip()) for line in f if line.strip()]
    
    train_ids = read_ids('train_run_ids.txt')
    val_ids = read_ids('val_run_ids.txt')
    test_ids = read_ids('test_run_ids.txt')
    
    return train_ids, val_ids, test_ids


# Alias for backward compatibility with train.py
load_design_ids = load_run_ids


if __name__ == "__main__":
    print("Testing GraphCast Data Loader for DrivAerML...\n")
    
    # Example usage
    data_dir = "path/to/DrivAerML"  # Contains run_1, run_2, ..., run_500 folders
    split_dir = "path/to/DrivAerML/train_val_test_splits"
    
    # For testing, create dummy data
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    temp_split_dir = tempfile.mkdtemp()
    
    try:
        # Create dummy VTP files and geometry parameters
        dummy_ids = list(range(1, 11))  # run_1 to run_10
        for run_id in dummy_ids:
            run_dir = os.path.join(temp_dir, f"run_{run_id}")
            os.makedirs(run_dir, exist_ok=True)
            
            # Create dummy VTP file using pyvista
            num_cells = np.random.randint(40000, 60000)
            points = np.random.randn(num_cells * 3, 3).astype(np.float32)
            cells = np.arange(num_cells * 3).reshape(-1, 3)
            
            # Create mesh
            mesh = pv.PolyData(points, cells)
            mesh.cell_data["CpMeanTrim"] = np.random.randn(num_cells).astype(np.float32)
            mesh.save(os.path.join(run_dir, f"boundary_{run_id}.vtp"))
            
            # Create dummy geometry parameters CSV
            geo_params = np.random.randn(16).astype(np.float32)
            geo_df = pd.DataFrame([geo_params])
            geo_df.to_csv(os.path.join(run_dir, f"geo_parameters_{run_id}.csv"), index=False)
        
        # Create dummy split files
        with open(os.path.join(temp_split_dir, 'train_run_ids.txt'), 'w') as f:
            f.write('\n'.join(map(str, dummy_ids[:6])))
        with open(os.path.join(temp_split_dir, 'val_run_ids.txt'), 'w') as f:
            f.write('\n'.join(map(str, dummy_ids[6:8])))
        with open(os.path.join(temp_split_dir, 'test_run_ids.txt'), 'w') as f:
            f.write('\n'.join(map(str, dummy_ids[8:])))
        
        # Load run IDs
        train_ids, val_ids, test_ids = load_run_ids(temp_split_dir)
        print(f"Loaded {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test run IDs")
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=temp_dir,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            batch_size=1,
            num_workers=0,
            normalize=True,
            verbose=True,
        )
        
        # Test loading a batch
        print("\nTesting data loading...")
        for batch in train_loader:
            data = batch[0]  # Get first (and only) sample in batch
            print(f"  Features shape: {data.x.shape}")
            print(f"  Positions shape: {data.pos.shape}")
            print(f"  Target shape: {data.y.shape}")
            print(f"  Geometry params shape: {data.u.shape}")
            print(f"  Run ID: {data.run_id}")
            break
        
        print("\n✓ GraphCast data loader test passed for DrivAerML!")
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        shutil.rmtree(temp_split_dir)
