"""
Data Loader for Transolver on DrivAerML

Loads VTP surface mesh files and prepares data for Transolver training.
Follows XAeroNet preprocessor pattern for data extraction.

Key Pattern (from XAeroNet preprocessor.py):
1. Load VTP file
2. Convert to triangular mesh
3. Convert cell_data to point_data (CRITICAL)
4. Extract from point_data: points, normals, pMeanTrim

Author: Implementation for DrivAerML benchmark
Date: February 2026
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from typing import List, Tuple, Optional, Dict
import pyvista as pv
import vtk


class TransolverDataset(Dataset):
    """
    Dataset for Transolver surface pressure prediction on DrivAerML.
    
    Data structure:
        - Surface points: (x, y, z) coordinates from mesh vertices
        - Surface normals: (nx, ny, nz) computed at points
        - Geometry parameters: 16 design variables from CSV
        - Target: Pressure (pMeanTrim) at points
        
    Following XAeroNet pattern:
        - Load VTP → Triangulate → cell_data_to_point_data → Extract
        - Use point_data["pMeanTrim"] (not cell_data["CpMeanTrim"])
        - Use mesh vertices (not cell centers)
        - Use point normals (not cell normals)
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
            data_dir: Root directory containing DrivAerML data (run_X folders)
            run_ids: List of run IDs to load (1-500)
            normalize: Whether to normalize features
            verbose: Print loading information
        """
        self.data_dir = data_dir
        self.run_ids = run_ids
        self.normalize = normalize
        self.verbose = verbose
        
        # Normalization statistics
        self.coords_mean = None
        self.coords_std = None
        self.normals_mean = None
        self.normals_std = None
        self.geo_params_mean = None
        self.geo_params_std = None
        self.pressure_mean = None
        self.pressure_std = None
        
        # Verify data exists
        self._verify_data()
        
        if self.verbose:
            print(f"TransolverDataset initialized with {len(self.run_ids)} runs")
    
    def _verify_data(self):
        """Check that data files exist"""
        missing_runs = []
        missing_csvs = []
        
        for run_id in self.run_ids[:min(5, len(self.run_ids))]:
            vtp_file = os.path.join(self.data_dir, f"run_{run_id}", f"boundary_{run_id}.vtp")
            csv_file = os.path.join(self.data_dir, f"geo_parameters_{run_id}.csv")
            
            if not os.path.exists(vtp_file):
                missing_runs.append(run_id)
            if not os.path.exists(csv_file):
                missing_csvs.append(run_id)
        
        if missing_runs:
            raise FileNotFoundError(
                f"Missing VTP files for runs: {missing_runs}\n"
                f"Expected location: {self.data_dir}/run_X/boundary_X.vtp"
            )
        if missing_csvs:
            raise FileNotFoundError(
                f"Missing geometry CSV files for runs: {missing_csvs}\n"
                f"Expected location: {self.data_dir}/geo_parameters_X.csv"
            )
    
    def _load_vtp_with_point_data(self, vtp_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load VTP file and extract point-based data.
        
        Follows XAeroNet preprocessor.py pattern (lines 155-161):
        1. Load VTP
        2. Convert to triangular mesh
        3. Convert cell_data to point_data (CRITICAL)
        4. Extract points, normals, pressure from point_data
        
        Args:
            vtp_file: Path to VTP file
            
        Returns:
            points: [N, 3] - mesh vertices (x, y, z)
            normals: [N, 3] - point normals (nx, ny, nz)
            pressure: [N] - pressure at points (pMeanTrim)
        """
        # Step 1: Load VTP
        surf = pv.read(vtp_file)
        
        # Step 2: Convert to triangular mesh (if not already)
        if surf.n_cells > 0 and surf.get_cell(0).type != vtk.VTK_TRIANGLE:
            tet_filter = vtk.vtkDataSetTriangleFilter()
            tet_filter.SetInputData(surf)
            tet_filter.Update()
            surf = pv.wrap(tet_filter.GetOutput())
        
        # Step 3: CRITICAL - Convert cell_data to point_data
        # This is the key step from XAeroNet preprocessor.py
        surf = surf.cell_data_to_point_data()
        
        # Step 4: Extract from point_data
        points = surf.points  # Mesh vertices (not cell centers)
        
        # Compute normals at points (not cells)
        surf_with_normals = surf.compute_normals(point_normals=True, cell_normals=False)
        normals = surf_with_normals.point_data["Normals"]
        
        # Extract pressure from point_data
        # Priority: pMeanTrim > CpMeanTrim > pressure > p
        pressure = None
        for field_name in ["pMeanTrim", "CpMeanTrim", "pressure", "p"]:
            if field_name in surf.point_data:
                pressure = surf.point_data[field_name]
                if self.verbose:
                    print(f"Using field: {field_name}")
                break
        
        if pressure is None:
            available = list(surf.point_data.keys())
            raise ValueError(f"No pressure field found in VTP. Available: {available}")
        
        return points, normals, pressure
    
    def _load_geometry_parameters(self, run_id: int) -> np.ndarray:
        """
        Load geometry parameters from CSV.
        
        Args:
            run_id: Run ID
            
        Returns:
            geo_params: [16] - design variables
        """
        csv_file = os.path.join(self.data_dir, f"geo_parameters_{run_id}.csv")
        df = pd.read_csv(csv_file)
        
        # Expected 16 parameters (excluding ID column)
        geo_params = df.iloc[0, 1:].values.astype(np.float32)
        
        if len(geo_params) != 16:
            raise ValueError(f"Expected 16 geometry parameters, got {len(geo_params)}")
        
        return geo_params
    
    def compute_normalization_stats(self):
        """Compute mean and std for normalization"""
        if self.verbose:
            print("Computing normalization statistics...")
        
        all_coords = []
        all_normals = []
        all_pressures = []
        all_geo_params = []
        
        # Sample subset for stats (first 50 or all if less)
        sample_size = min(50, len(self))
        for idx in range(sample_size):
            run_id = self.run_ids[idx]
            vtp_file = os.path.join(self.data_dir, f"run_{run_id}", f"boundary_{run_id}.vtp")
            
            points, normals, pressure = self._load_vtp_with_point_data(vtp_file)
            geo_params = self._load_geometry_parameters(run_id)
            
            all_coords.append(points)
            all_normals.append(normals)
            all_pressures.append(pressure)
            all_geo_params.append(geo_params)
        
        # Concatenate
        all_coords = np.concatenate(all_coords, axis=0)
        all_normals = np.concatenate(all_normals, axis=0)
        all_pressures = np.concatenate(all_pressures, axis=0)
        all_geo_params = np.stack(all_geo_params, axis=0)
        
        # Compute statistics
        self.coords_mean = torch.tensor(all_coords.mean(axis=0), dtype=torch.float32)
        self.coords_std = torch.tensor(all_coords.std(axis=0) + 1e-8, dtype=torch.float32)
        self.normals_mean = torch.tensor(all_normals.mean(axis=0), dtype=torch.float32)
        self.normals_std = torch.tensor(all_normals.std(axis=0) + 1e-8, dtype=torch.float32)
        self.geo_params_mean = torch.tensor(all_geo_params.mean(axis=0), dtype=torch.float32)
        self.geo_params_std = torch.tensor(all_geo_params.std(axis=0) + 1e-8, dtype=torch.float32)
        self.pressure_mean = torch.tensor(all_pressures.mean(), dtype=torch.float32)
        self.pressure_std = torch.tensor(all_pressures.std() + 1e-8, dtype=torch.float32)
        
        if self.verbose:
            print(f"Coordinates - mean: {self.coords_mean.numpy()}, std: {self.coords_std.numpy()}")
            print(f"Normals - mean: {self.normals_mean.numpy()}, std: {self.normals_std.numpy()}")
            print(f"Pressure - mean: {self.pressure_mean:.4f}, std: {self.pressure_std:.4f}")
            print(f"Geo params - mean shape: {self.geo_params_mean.shape}")
    
    def __len__(self) -> int:
        return len(self.run_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return a single sample.
        
        Returns:
            Dictionary with:
                - positions: [num_points, 3] - point coordinates (x, y, z)
                - normals: [num_points, 3] - point normals (nx, ny, nz)
                - features: [num_points, 6] - concatenated [coords, normals]
                - geo_params: [16] - geometry parameters
                - pressures: [num_points] - target pressure
                - run_id: int - run identifier
        """
        run_id = self.run_ids[idx]
        vtp_file = os.path.join(self.data_dir, f"run_{run_id}", f"boundary_{run_id}.vtp")
        
        # Load VTP with point-based data (XAeroNet pattern)
        points, normals, pressure = self._load_vtp_with_point_data(vtp_file)
        
        # Load geometry parameters
        geo_params = self._load_geometry_parameters(run_id)
        
        # Convert to tensors
        positions = torch.tensor(points, dtype=torch.float32)
        normals_tensor = torch.tensor(normals, dtype=torch.float32)
        pressures = torch.tensor(pressure, dtype=torch.float32)
        geo_params_tensor = torch.tensor(geo_params, dtype=torch.float32)
        
        # Normalize
        if self.normalize:
            if self.coords_mean is None:
                self.compute_normalization_stats()
            
            positions_norm = (positions - self.coords_mean) / self.coords_std
            normals_norm = (normals_tensor - self.normals_mean) / self.normals_std
            geo_params_norm = (geo_params_tensor - self.geo_params_mean) / self.geo_params_std
            pressures_norm = (pressures - self.pressure_mean) / self.pressure_std
            
            # Features: concatenate normalized coordinates and normals
            features = torch.cat([positions_norm, normals_norm], dim=-1)  # [N, 6]
            
            return {
                'positions': positions,  # Original (for visualization)
                'normals': normals_tensor,  # Original
                'features': features,  # Normalized [coords, normals]
                'geo_params': geo_params_norm,  # Normalized
                'pressures': pressures_norm,  # Normalized target
                'run_id': run_id,
            }
        else:
            # Features: concatenate coordinates and normals
            features = torch.cat([positions, normals_tensor], dim=-1)  # [N, 6]
            
            return {
                'positions': positions,
                'normals': normals_tensor,
                'features': features,
                'geo_params': geo_params_tensor,
                'pressures': pressures,
                'run_id': run_id,
            }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    """
    Custom collate function for variable-sized point clouds.
    
    Transolver processes each sample independently (no batching across samples).
    Each sample has a different number of points.
    
    Args:
        batch: List of sample dictionaries
    
    Returns:
        Same list (no collation)
    """
    return batch


def create_dataloaders(
    data_dir: str,
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    batch_size: int = 1,
    num_workers: int = 4,
    normalize: bool = True,
    verbose: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root directory containing DrivAerML data
        train_ids: List of training run IDs (1-500)
        val_ids: List of validation run IDs
        test_ids: List of test run IDs
        batch_size: Batch size (typically 1 for variable-sized point clouds)
        num_workers: Number of data loading workers
        normalize: Whether to normalize features
        verbose: Print information
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = TransolverDataset(
        data_dir=data_dir,
        run_ids=train_ids,
        normalize=normalize,
        verbose=verbose,
    )
    
    val_dataset = TransolverDataset(
        data_dir=data_dir,
        run_ids=val_ids,
        normalize=normalize,
        verbose=verbose,
    )
    
    test_dataset = TransolverDataset(
        data_dir=data_dir,
        run_ids=test_ids,
        normalize=normalize,
        verbose=False,
    )
    
    # Compute normalization stats on training set
    if normalize:
        train_dataset.compute_normalization_stats()
        
        # Share stats with val and test
        val_dataset.coords_mean = train_dataset.coords_mean
        val_dataset.coords_std = train_dataset.coords_std
        val_dataset.normals_mean = train_dataset.normals_mean
        val_dataset.normals_std = train_dataset.normals_std
        val_dataset.geo_params_mean = train_dataset.geo_params_mean
        val_dataset.geo_params_std = train_dataset.geo_params_std
        val_dataset.pressure_mean = train_dataset.pressure_mean
        val_dataset.pressure_std = train_dataset.pressure_std
        
        test_dataset.coords_mean = train_dataset.coords_mean
        test_dataset.coords_std = train_dataset.coords_std
        test_dataset.normals_mean = train_dataset.normals_mean
        test_dataset.normals_std = train_dataset.normals_std
        test_dataset.geo_params_mean = train_dataset.geo_params_mean
        test_dataset.geo_params_std = train_dataset.geo_params_std
        test_dataset.pressure_mean = train_dataset.pressure_mean
        test_dataset.pressure_std = train_dataset.pressure_std
    
    # Create dataloaders with custom collate
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
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Load train/val/test splits
    data_dir = "C:/Learning/Scientific/CARBENCH/DrivAerML"
    
    with open("../train_val_test_splits/train_run_ids.txt", "r") as f:
        train_ids = [int(line.strip()) for line in f]
    
    with open("../train_val_test_splits/val_run_ids.txt", "r") as f:
        val_ids = [int(line.strip()) for line in f]
    
    with open("../train_val_test_splits/test_run_ids.txt", "r") as f:
        test_ids = [int(line.strip()) for line in f]
    
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        batch_size=1,
        num_workers=0,
        normalize=True,
        verbose=True,
    )
    
    # Test loading
    print("\nTesting data loading...")
    for batch in train_loader:
        sample = batch[0]  # First item in batch
        print(f"Run ID: {sample['run_id']}")
        print(f"Positions shape: {sample['positions'].shape}")
        print(f"Normals shape: {sample['normals'].shape}")
        print(f"Features shape: {sample['features'].shape}")
        print(f"Geo params shape: {sample['geo_params'].shape}")
        print(f"Pressures shape: {sample['pressures'].shape}")
        break
    
    print("\nTransolver data loader ready!")
