"""
Data Loader for Fourier Neural Operator (FNO) on DrivAerML

Loads VTP surface mesh files, extracts point data following XAeroNet pattern,
then voxelizes for FNO processing.

Key Pattern (from XAeroNet preprocessor.py):
1. Load VTP file
2. Convert to triangular mesh
3. Convert cell_data to point_data (CRITICAL)
4. Extract from point_data: points, pressure (pMeanTrim)
5. Voxelize point cloud for FNO

FNO expects regular grid input with 4 channels:
- Channel 0: Occupancy (1 inside geometry, 0 outside)
- Channels 1-3: Normalized grid coordinates (x, y, z)

Reference: Li, Z. et al. Fourier neural operator for parametric PDEs. arXiv:2010.08895

Author: Implementation for DrivAerML benchmark
Date: February 2026
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import pyvista as pv
import vtk
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VoxelGridDataset(Dataset):
    """
    Dataset for FNO surface pressure prediction on DrivAerML.
    
    Converts surface meshes to voxel grids following XAeroNet extraction pattern.
    
    Data structure:
        - Voxel grid: [4, H, W, D] - occupancy + normalized coordinates
        - Surface points: Sampled from mesh vertices (for target evaluation)
        - Target: Pressure (pMeanTrim) at surface points
        - Geometry parameters: 16 design variables from CSV
        
    Following XAeroNet pattern:
        - Load VTP → Triangulate → cell_data_to_point_data → Extract
        - Use point_data["pMeanTrim"] (not cell_data["CpMeanTrim"])
        - Use mesh vertices (not cell centers)
        - Voxelize the extracted point cloud
    """
    
    def __init__(
        self,
        data_dir: str,
        run_ids: List[int],
        grid_resolution: int = 32,
        num_sample_points: Optional[int] = 5000,
        normalize: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            data_dir: Root directory containing DrivAerML data (run_X folders)
            run_ids: List of run IDs to load (1-500)
            grid_resolution: Voxel grid resolution (e.g., 32 for 32^3 grid)
            num_sample_points: Number of points to sample for target evaluation
            normalize: Whether to normalize features
            verbose: Print loading information
        """
        self.data_dir = data_dir
        self.run_ids = run_ids
        self.grid_resolution = grid_resolution
        self.num_sample_points = num_sample_points
        self.normalize = normalize
        self.verbose = verbose
        
        # Normalization statistics
        self.pressure_mean = None
        self.pressure_std = None
        self.geo_params_mean = None
        self.geo_params_std = None
        
        # Verify data exists
        self._verify_data()
        
        if self.verbose:
            print(f"VoxelGridDataset initialized with {len(self.run_ids)} runs")
            print(f"Grid resolution: {grid_resolution}^3 = {grid_resolution**3} voxels")
    
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
    
    def _compute_bounding_box(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute bounding box with padding.
        
        Args:
            points: [N, 3] - point coordinates
            
        Returns:
            min_coords: [3] - minimum coordinates
            max_coords: [3] - maximum coordinates
        """
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        
        # Add 10% padding to avoid boundary issues
        ranges = max_coords - min_coords
        min_coords -= 0.1 * ranges
        max_coords += 0.1 * ranges
        
        return min_coords, max_coords
    
    def _voxelize_point_cloud(
        self,
        points: np.ndarray,
        resolution: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert point cloud to voxel grid.
        
        Creates a 4-channel voxel grid:
        - Channel 0: Occupancy (1 where points exist, 0 elsewhere)
        - Channels 1-3: Normalized coordinates (x, y, z)
        
        Args:
            points: [N, 3] - point coordinates
            resolution: voxel grid resolution
            
        Returns:
            voxel_grid: [4, resolution, resolution, resolution]
            min_coords: [3] - bounding box minimum
            max_coords: [3] - bounding box maximum
        """
        min_coords, max_coords = self._compute_bounding_box(points)
        
        # Create coordinate grids
        x = np.linspace(min_coords[0], max_coords[0], resolution)
        y = np.linspace(min_coords[1], max_coords[1], resolution)
        z = np.linspace(min_coords[2], max_coords[2], resolution)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Initialize occupancy grid
        occupancy = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        
        # Map points to voxel indices
        normalized_points = (points - min_coords) / (max_coords - min_coords)
        voxel_indices = (normalized_points * (resolution - 1)).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, resolution - 1)
        
        # Mark occupied voxels
        occupancy[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.0
        
        # Create normalized coordinate grids
        x_norm = (X - min_coords[0]) / (max_coords[0] - min_coords[0])
        y_norm = (Y - min_coords[1]) / (max_coords[1] - min_coords[1])
        z_norm = (Z - min_coords[2]) / (max_coords[2] - min_coords[2])
        
        # Stack into 4-channel grid: [occupancy, x, y, z]
        voxel_grid = np.stack([occupancy, x_norm, y_norm, z_norm], axis=0).astype(np.float32)
        
        return voxel_grid, min_coords, max_coords
    
    def _sample_surface_points(
        self,
        points: np.ndarray,
        pressure: np.ndarray,
        n_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points from surface for target evaluation.
        
        Args:
            points: [N, 3] - all surface points
            pressure: [N] - pressure at all points
            n_points: number of points to sample
            
        Returns:
            sampled_points: [n_points, 3]
            sampled_pressure: [n_points]
        """
        if len(points) > n_points:
            indices = np.random.choice(len(points), n_points, replace=False)
        else:
            indices = np.arange(len(points))
        
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
        
        all_pressures = []
        all_geo_params = []
        
        # Sample subset for stats (first 50 or all if less)
        sample_size = min(50, len(self))
        for idx in range(sample_size):
            run_id = self.run_ids[idx]
            vtp_file = os.path.join(self.data_dir, f"run_{run_id}", f"boundary_{run_id}.vtp")
            
            _, pressure = self._load_vtp_with_point_data(vtp_file)
            geo_params = self._load_geometry_parameters(run_id)
            
            all_pressures.append(pressure)
            all_geo_params.append(geo_params)
        
        # Concatenate
        all_pressures = np.concatenate(all_pressures, axis=0)
        all_geo_params = np.stack(all_geo_params, axis=0)
        
        # Compute statistics
        self.pressure_mean = torch.tensor(all_pressures.mean(), dtype=torch.float32)
        self.pressure_std = torch.tensor(all_pressures.std() + 1e-8, dtype=torch.float32)
        self.geo_params_mean = torch.tensor(all_geo_params.mean(axis=0), dtype=torch.float32)
        self.geo_params_std = torch.tensor(all_geo_params.std(axis=0) + 1e-8, dtype=torch.float32)
        
        if self.verbose:
            print(f"Pressure - mean: {self.pressure_mean:.4f}, std: {self.pressure_std:.4f}")
    
    def __len__(self) -> int:
        return len(self.run_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return a single sample.
        
        Returns:
            Dictionary with:
                - voxel_grid: [4, H, W, D] - input grid (occupancy + coords)
                - positions: [N, 3] - sampled surface point positions
                - pressures: [N] - target pressure at positions
                - geo_params: [16] - geometry parameters
                - run_id: int - run identifier
                - bbox: tuple - (min_coords, max_coords)
        """
        run_id = self.run_ids[idx]
        vtp_file = os.path.join(self.data_dir, f"run_{run_id}", f"boundary_{run_id}.vtp")
        
        # Load VTP with point-based data (XAeroNet pattern)
        points, pressure = self._load_vtp_with_point_data(vtp_file)
        
        # Voxelize point cloud
        voxel_grid, min_coords, max_coords = self._voxelize_point_cloud(
            points, self.grid_resolution
        )
        
        # Sample surface points for target evaluation
        if self.num_sample_points is not None:
            sampled_points, sampled_pressure = self._sample_surface_points(
                points, pressure, self.num_sample_points
            )
        else:
            sampled_points = points
            sampled_pressure = pressure
        
        # Load geometry parameters
        geo_params = self._load_geometry_parameters(run_id)
        
        # Convert to tensors
        voxel_grid_tensor = torch.tensor(voxel_grid, dtype=torch.float32)
        positions = torch.tensor(sampled_points, dtype=torch.float32)
        pressures = torch.tensor(sampled_pressure, dtype=torch.float32)
        geo_params_tensor = torch.tensor(geo_params, dtype=torch.float32)
        
        # Normalize
        if self.normalize:
            if self.pressure_mean is None:
                self.compute_normalization_stats()
            
            pressures = (pressures - self.pressure_mean) / self.pressure_std
            geo_params_tensor = (geo_params_tensor - self.geo_params_mean) / self.geo_params_std
        
        return {
            'voxel_grid': voxel_grid_tensor,
            'positions': positions,
            'pressures': pressures,
            'geo_params': geo_params_tensor,
            'run_id': run_id,
            'bbox': (min_coords, max_coords),
        }


def create_dataloaders(
    data_dir: str,
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    grid_resolution: int = 32,
    num_sample_points: Optional[int] = 5000,
    batch_size: int = 8,
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
        grid_resolution: Voxel grid resolution (e.g., 32, 64)
        num_sample_points: Number of points to sample for target evaluation
        batch_size: Batch size
        num_workers: Number of data loading workers
        normalize: Whether to normalize features
        verbose: Print information
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = VoxelGridDataset(
        data_dir=data_dir,
        run_ids=train_ids,
        grid_resolution=grid_resolution,
        num_sample_points=num_sample_points,
        normalize=normalize,
        verbose=verbose,
    )
    
    val_dataset = VoxelGridDataset(
        data_dir=data_dir,
        run_ids=val_ids,
        grid_resolution=grid_resolution,
        num_sample_points=num_sample_points,
        normalize=normalize,
        verbose=verbose,
    )
    
    test_dataset = VoxelGridDataset(
        data_dir=data_dir,
        run_ids=test_ids,
        grid_resolution=grid_resolution,
        num_sample_points=num_sample_points,
        normalize=normalize,
        verbose=False,
    )
    
    # Compute normalization stats on training set
    if normalize:
        train_dataset.compute_normalization_stats()
        
        # Share stats with val and test
        val_dataset.pressure_mean = train_dataset.pressure_mean
        val_dataset.pressure_std = train_dataset.pressure_std
        val_dataset.geo_params_mean = train_dataset.geo_params_mean
        val_dataset.geo_params_std = train_dataset.geo_params_std
        
        test_dataset.pressure_mean = train_dataset.pressure_mean
        test_dataset.pressure_std = train_dataset.pressure_std
        test_dataset.geo_params_mean = train_dataset.geo_params_mean
        test_dataset.geo_params_std = train_dataset.geo_params_std
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
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
    voxel_size: Tuple[int, int, int] = (32, 32, 32),
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
        voxel_size=voxel_size,
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
        grid_resolution=32,
        num_sample_points=5000,
        batch_size=4,
        num_workers=0,
        normalize=True,
        verbose=True,
    )
    
    # Test loading
    print("\nTesting data loading...")
    for batch in train_loader:
        print(f"Batch size: {len(batch['run_id'])}")
        print(f"Voxel grid shape: {batch['voxel_grid'].shape}")  # [B, 4, H, W, D]
        print(f"Positions shape: {batch['positions'].shape}")  # [B, N, 3]
        print(f"Pressures shape: {batch['pressures'].shape}")  # [B, N]
        print(f"Geo params shape: {batch['geo_params'].shape}")  # [B, 16]
        break
    
    print("\nNeuralOperator (FNO) data loader ready!")
