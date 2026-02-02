"""
Data Loader for RegDGCNN on DrivAerML

Loads VTP surface mesh files and prepares point clouds for RegDGCNN training.
Follows XAeroNet preprocessor pattern for data extraction.

Key Pattern (from XAeroNet preprocessor.py):
1. Load VTP file
2. Convert to triangular mesh
3. Convert cell_data to point_data (CRITICAL)
4. Extract from point_data: points, normals, pMeanTrim

RegDGCNN (Dynamic Graph CNN):
- Input: Point cloud [x, y, z] (no normals needed, computed dynamically)
- Graph: Dynamically constructed in each layer based on feature space
- Target: Pressure at each point

Reference: Mohamed Elrefaie, RegDGCNN for DrivAerNet

Author: Implementation for DrivAerML benchmark
Date: February 2026
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import pyvista as pv
import vtk
import logging


class SurfacePressureDataset(Dataset):
    """
    Dataset for RegDGCNN surface pressure prediction on DrivAerML.
    
    Data structure:
        - Point cloud: [x, y, z] coordinates from mesh vertices
        - Geometry parameters: 16 design variables from CSV
        - Target: Pressure (pMeanTrim) at points
        
    Following XAeroNet pattern:
        - Load VTP → Triangulate → cell_data_to_point_data → Extract
        - Use point_data["pMeanTrim"] (not cell_data["CpMeanTrim"])
        - Use mesh vertices (not cell centers)
        
    Note: RegDGCNN doesn't need normals - it constructs dynamic graphs
    based on features in each layer.
    """
    
    def __init__(
        self,
        data_dir: str,
        run_ids: List[int],
        num_points: Optional[int] = None,
        normalize: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            data_dir: Root directory containing DrivAerML data (run_X folders)
            run_ids: List of run IDs to load (1-500)
            num_points: Optional - number of points to sample (None = use all)
            normalize: Whether to normalize features
            verbose: Print loading information
        """
        self.data_dir = data_dir
        self.run_ids = run_ids
        self.num_points = num_points
        self.normalize = normalize
        self.verbose = verbose
        
        # Normalization statistics
        self.coords_mean = None
        self.coords_std = None
        self.geo_params_mean = None
        self.geo_params_std = None
        self.pressure_mean = None
        self.pressure_std = None
        
        # Verify data exists
        self._verify_data()
        
        if self.verbose:
            print(f"SurfacePressureDataset initialized with {len(self.run_ids)} runs")
            if num_points:
                print(f"Sampling {num_points} points per mesh")
    
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
    
    def _load_vtp_with_point_data(self, vtp_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load VTP file and extract point-based data.
        
        Follows XAeroNet preprocessor.py pattern (lines 155-161):
        1. Load VTP
        2. Convert to triangular mesh
        3. Convert cell_data to point_data (CRITICAL)
        4. Extract points, pressure from point_data
        
        Args:
            vtp_file: Path to VTP file
            
        Returns:
            points: [N, 3] - mesh vertices (x, y, z)
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
        
        return points, pressure
    
    def _sample_points(self, points: np.ndarray, pressure: np.ndarray, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample n_points from the point cloud uniformly.
        
        Args:
            points: [N, 3] - all points
            pressure: [N] - all pressures
            n_points: number of points to sample
            
        Returns:
            sampled_points: [n_points, 3]
            sampled_pressure: [n_points]
        """
        if len(points) > n_points:
            indices = np.random.choice(len(points), n_points, replace=False)
        else:
            indices = np.arange(len(points))
            if self.verbose:
                logging.info(f"Mesh has only {len(points)} points. Using all available points.")
        
        return points[indices], pressure[indices]
    
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
        all_pressures = []
        all_geo_params = []
        
        # Sample subset for stats (first 50 or all if less)
        sample_size = min(50, len(self))
        for idx in range(sample_size):
            run_id = self.run_ids[idx]
            vtp_file = os.path.join(self.data_dir, f"run_{run_id}", f"boundary_{run_id}.vtp")
            
            points, pressure = self._load_vtp_with_point_data(vtp_file)
            
            # Sample if needed
            if self.num_points is not None:
                points, pressure = self._sample_points(points, pressure, self.num_points)
            
            geo_params = self._load_geometry_parameters(run_id)
            
            all_coords.append(points)
            all_pressures.append(pressure)
            all_geo_params.append(geo_params)
        
        # Concatenate
        all_coords = np.concatenate(all_coords, axis=0)
        all_pressures = np.concatenate(all_pressures, axis=0)
        all_geo_params = np.stack(all_geo_params, axis=0)
        
        # Compute statistics
        self.coords_mean = torch.tensor(all_coords.mean(axis=0), dtype=torch.float32)
        self.coords_std = torch.tensor(all_coords.std(axis=0) + 1e-8, dtype=torch.float32)
        self.geo_params_mean = torch.tensor(all_geo_params.mean(axis=0), dtype=torch.float32)
        self.geo_params_std = torch.tensor(all_geo_params.std(axis=0) + 1e-8, dtype=torch.float32)
        self.pressure_mean = torch.tensor(all_pressures.mean(), dtype=torch.float32)
        self.pressure_std = torch.tensor(all_pressures.std() + 1e-8, dtype=torch.float32)
        
        if self.verbose:
            print(f"Coordinates - mean: {self.coords_mean.numpy()}, std: {self.coords_std.numpy()}")
            print(f"Pressure - mean: {self.pressure_mean:.4f}, std: {self.pressure_std:.4f}")
    
    def __len__(self) -> int:
        return len(self.run_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single sample.
        
        Returns:
            point_cloud: [1, 3, N] - point coordinates (RegDGCNN format: [batch, channels, points])
            pressure: [1, N] - target pressure
            
        Note: RegDGCNN expects input shape [batch, 3, num_points]
        """
        run_id = self.run_ids[idx]
        vtp_file = os.path.join(self.data_dir, f"run_{run_id}", f"boundary_{run_id}.vtp")
        
        # Load VTP with point-based data (XAeroNet pattern)
        points, pressure = self._load_vtp_with_point_data(vtp_file)
        
        # Sample points if specified
        if self.num_points is not None:
            points, pressure = self._sample_points(points, pressure, self.num_points)
        
        # Load geometry parameters
        geo_params = self._load_geometry_parameters(run_id)
        
        # Convert to tensors
        point_cloud = torch.tensor(points, dtype=torch.float32)  # [N, 3]
        pressures = torch.tensor(pressure, dtype=torch.float32)  # [N]
        geo_params_tensor = torch.tensor(geo_params, dtype=torch.float32)  # [16]
        
        # Normalize
        if self.normalize:
            if self.coords_mean is None:
                self.compute_normalization_stats()
            
            point_cloud = (point_cloud - self.coords_mean) / self.coords_std
            geo_params_tensor = (geo_params_tensor - self.geo_params_mean) / self.geo_params_std
            pressures = (pressures - self.pressure_mean) / self.pressure_std
        
        # RegDGCNN format: [1, 3, N] for point cloud, [1, N] for pressure
        point_cloud = point_cloud.T.unsqueeze(0)  # [N, 3] -> [3, N] -> [1, 3, N]
        pressures = pressures.unsqueeze(0)  # [N] -> [1, N]
        
        return point_cloud, pressures


def create_dataloaders(
    data_dir: str,
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    num_points: Optional[int] = None,
    batch_size: int = 32,
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
        num_points: Optional - number of points to sample (None = use all)
        batch_size: Batch size
        num_workers: Number of data loading workers
        normalize: Whether to normalize features
        verbose: Print information
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = SurfacePressureDataset(
        data_dir=data_dir,
        run_ids=train_ids,
        num_points=num_points,
        normalize=normalize,
        verbose=verbose,
    )
    
    val_dataset = SurfacePressureDataset(
        data_dir=data_dir,
        run_ids=val_ids,
        num_points=num_points,
        normalize=normalize,
        verbose=verbose,
    )
    
    test_dataset = SurfacePressureDataset(
        data_dir=data_dir,
        run_ids=test_ids,
        num_points=num_points,
        normalize=normalize,
        verbose=False,
    )
    
    # Compute normalization stats on training set
    if normalize:
        train_dataset.compute_normalization_stats()
        
        # Share stats with val and test
        val_dataset.coords_mean = train_dataset.coords_mean
        val_dataset.coords_std = train_dataset.coords_std
        val_dataset.geo_params_mean = train_dataset.geo_params_mean
        val_dataset.geo_params_std = train_dataset.geo_params_std
        val_dataset.pressure_mean = train_dataset.pressure_mean
        val_dataset.pressure_std = train_dataset.pressure_std
        
        test_dataset.coords_mean = train_dataset.coords_mean
        test_dataset.coords_std = train_dataset.coords_std
        test_dataset.geo_params_mean = train_dataset.geo_params_mean
        test_dataset.geo_params_std = train_dataset.geo_params_std
        test_dataset.pressure_mean = train_dataset.pressure_mean
        test_dataset.pressure_std = train_dataset.pressure_std
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader, test_loader


# Backward compatibility: Export normalization constants
# These will be None until dataloaders are created
PRESSURE_MEAN = None
PRESSURE_STD = None


def get_dataloaders(
    data_dir: str,
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    batch_size: int = 32,
    num_workers: int = 4,
    k_neighbors: int = 16,
) -> Tuple:
    """
    Wrapper for create_dataloaders() to match train.py API.
    Also updates module-level PRESSURE_MEAN and PRESSURE_STD constants.
    """
    global PRESSURE_MEAN, PRESSURE_STD
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        batch_size=batch_size,
        num_workers=num_workers,
        k_neighbors=k_neighbors,
    )
    
    # Export normalization constants from training dataset
    PRESSURE_MEAN = train_loader.dataset.pressure_mean.item()
    PRESSURE_STD = train_loader.dataset.pressure_std.item()
    
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
        num_points=5000,  # Sample 5000 points per mesh
        batch_size=8,
        num_workers=0,
        normalize=True,
        verbose=True,
    )
    
    # Test loading
    print("\nTesting data loading...")
    for point_cloud, pressure in train_loader:
        print(f"Point cloud shape: {point_cloud.shape}")  # [batch, 1, 3, N]
        print(f"Pressure shape: {pressure.shape}")  # [batch, 1, N]
        print(f"Point cloud range: [{point_cloud.min():.3f}, {point_cloud.max():.3f}]")
        print(f"Pressure range: [{pressure.min():.3f}, {pressure.max():.3f}]")
        break
    
    print("\nRegDGCNN data loader ready!")
