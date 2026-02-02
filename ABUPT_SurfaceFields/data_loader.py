"""
Data Loader for AB-UPT (Anchored-Branched Universal Physics Transformers) on DrivAerML

Loads VTP surface mesh files and prepares data for AB-UPT training.
Follows XAeroNet preprocessor pattern for data extraction.

Key Pattern (from XAeroNet preprocessor.py):
1. Load VTP file
2. Convert to triangular mesh
3. Convert cell_data to point_data (CRITICAL)
4. Extract from point_data: points, normals, pMeanTrim, wallShearStressMeanTrim

AB-UPT (Branched Architecture):
- Input: Surface positions + normals
- Optional: Wall shear stress fields
- Target: Pressure at surface points
- Architecture: Branched transformer with anchored attention

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


class SurfaceFieldDataset(Dataset):
    """
    Dataset for AB-UPT surface field prediction on DrivAerML.
    
    Data structure:
        - Surface positions: [x, y, z] from mesh vertices
        - Surface normals: [nx, ny, nz] at points
        - Geometry parameters: 16 design variables from CSV
        - Target: Pressure (pMeanTrim) at points
        - Optional: Wall shear stress (wallShearStressMeanTrim) at points
        
    Following XAeroNet pattern:
        - Load VTP → Triangulate → cell_data_to_point_data → Extract
        - Use point_data["pMeanTrim"] (not cell_data["CpMeanTrim"])
        - Use point_data["wallShearStressMeanTrim"] for WSS
        - Use mesh vertices (not cell centers)
        - Use point normals (not cell normals)
    """
    
    def __init__(
        self,
        data_dir: str,
        run_ids: List[int],
        num_points: Optional[int] = None,
        load_wss: bool = False,
        normalize: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            data_dir: Root directory containing DrivAerML data (run_X folders)
            run_ids: List of run IDs to load (1-500)
            num_points: Optional - number of points to sample (None = use all)
            load_wss: Whether to load wall shear stress data
            normalize: Whether to normalize features
            verbose: Print loading information
        """
        self.data_dir = data_dir
        self.run_ids = run_ids
        self.num_points = num_points
        self.load_wss = load_wss
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
        self.wss_mean = None
        self.wss_std = None
        
        # Verify data exists
        self._verify_data()
        
        if self.verbose:
            print(f"SurfaceFieldDataset initialized with {len(self.run_ids)} runs")
            if num_points:
                print(f"Sampling {num_points} points per mesh")
            if load_wss:
                print("Loading wall shear stress data")
    
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
    
    def _load_vtp_with_point_data(self, vtp_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load VTP file and extract point-based data.
        
        Follows XAeroNet preprocessor.py pattern (lines 155-161):
        1. Load VTP
        2. Convert to triangular mesh
        3. Convert cell_data to point_data (CRITICAL)
        4. Extract points, normals, pressure, (optional) WSS from point_data
        
        Args:
            vtp_file: Path to VTP file
            
        Returns:
            points: [N, 3] - mesh vertices (x, y, z)
            normals: [N, 3] - point normals (nx, ny, nz)
            pressure: [N] - pressure at points (pMeanTrim)
            wss: [N, 3] or None - wall shear stress at points (wallShearStressMeanTrim)
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
                    print(f"Using pressure field: {field_name}")
                break
        
        if pressure is None:
            available = list(surf.point_data.keys())
            raise ValueError(f"No pressure field found in VTP. Available: {available}")
        
        # Extract wall shear stress if requested
        wss = None
        if self.load_wss:
            # Priority: wallShearStressMeanTrim > wallShearStress > WSS > tau
            for field_name in ["wallShearStressMeanTrim", "wallShearStress", "WSS", "tau"]:
                if field_name in surf.point_data:
                    wss = surf.point_data[field_name]
                    if self.verbose:
                        print(f"Using WSS field: {field_name}")
                    break
            
            if wss is None:
                logging.warning("Wall shear stress requested but not found in VTP file")
        
        return points, normals, pressure, wss
    
    def _sample_points(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        pressure: np.ndarray,
        wss: Optional[np.ndarray],
        n_points: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Sample n_points uniformly.
        
        Args:
            points: [N, 3] - all points
            normals: [N, 3] - all normals
            pressure: [N] - all pressures
            wss: [N, 3] or None - all WSS values
            n_points: number of points to sample
            
        Returns:
            Sampled versions of all inputs
        """
        if len(points) > n_points:
            indices = np.random.choice(len(points), n_points, replace=False)
        else:
            indices = np.arange(len(points))
            if self.verbose:
                logging.info(f"Mesh has only {len(points)} points. Using all available points.")
        
        sampled_wss = wss[indices] if wss is not None else None
        
        return points[indices], normals[indices], pressure[indices], sampled_wss
    
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
        all_wss = []
        
        # Sample subset for stats (first 50 or all if less)
        sample_size = min(50, len(self))
        for idx in range(sample_size):
            run_id = self.run_ids[idx]
            vtp_file = os.path.join(self.data_dir, f"run_{run_id}", f"boundary_{run_id}.vtp")
            
            points, normals, pressure, wss = self._load_vtp_with_point_data(vtp_file)
            
            # Sample if needed
            if self.num_points is not None:
                points, normals, pressure, wss = self._sample_points(
                    points, normals, pressure, wss, self.num_points
                )
            
            geo_params = self._load_geometry_parameters(run_id)
            
            all_coords.append(points)
            all_normals.append(normals)
            all_pressures.append(pressure)
            all_geo_params.append(geo_params)
            
            if wss is not None:
                all_wss.append(wss)
        
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
        
        if len(all_wss) > 0:
            all_wss = np.concatenate(all_wss, axis=0)
            self.wss_mean = torch.tensor(all_wss.mean(axis=0), dtype=torch.float32)
            self.wss_std = torch.tensor(all_wss.std(axis=0) + 1e-8, dtype=torch.float32)
        
        if self.verbose:
            print(f"Coordinates - mean: {self.coords_mean.numpy()}, std: {self.coords_std.numpy()}")
            print(f"Normals - mean: {self.normals_mean.numpy()}, std: {self.normals_std.numpy()}")
            print(f"Pressure - mean: {self.pressure_mean:.4f}, std: {self.pressure_std:.4f}")
            if self.wss_mean is not None:
                print(f"WSS - mean: {self.wss_mean.numpy()}, std: {self.wss_std.numpy()}")
    
    def __len__(self) -> int:
        return len(self.run_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return a single sample.
        
        Returns:
            Dictionary with:
                - surface_position_vtp: [N, 3] - surface positions
                - surface_normals: [N, 3] - surface normals
                - surface_pressure: [N] - target pressure
                - surface_wallshearstress: [N, 3] - WSS (if load_wss=True)
                - geo_params: [16] - geometry parameters
                - run_id: int - run identifier
        """
        run_id = self.run_ids[idx]
        vtp_file = os.path.join(self.data_dir, f"run_{run_id}", f"boundary_{run_id}.vtp")
        
        # Load VTP with point-based data (XAeroNet pattern)
        points, normals, pressure, wss = self._load_vtp_with_point_data(vtp_file)
        
        # Sample points if specified
        if self.num_points is not None:
            points, normals, pressure, wss = self._sample_points(
                points, normals, pressure, wss, self.num_points
            )
        
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
            
            sample = {
                'surface_position_vtp': positions_norm,
                'surface_normals': normals_norm,
                'surface_pressure': pressures_norm,
                'geo_params': geo_params_norm,
                'run_id': run_id,
            }
            
            if wss is not None and self.wss_mean is not None:
                wss_tensor = torch.tensor(wss, dtype=torch.float32)
                wss_norm = (wss_tensor - self.wss_mean) / self.wss_std
                sample['surface_wallshearstress'] = wss_norm
        else:
            sample = {
                'surface_position_vtp': positions,
                'surface_normals': normals_tensor,
                'surface_pressure': pressures,
                'geo_params': geo_params_tensor,
                'run_id': run_id,
            }
            
            if wss is not None:
                sample['surface_wallshearstress'] = torch.tensor(wss, dtype=torch.float32)
        
        return sample


def create_dataloaders(
    data_dir: str,
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    num_points: Optional[int] = None,
    load_wss: bool = False,
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
        num_points: Optional - number of points to sample (None = use all)
        load_wss: Whether to load wall shear stress data
        batch_size: Batch size
        num_workers: Number of data loading workers
        normalize: Whether to normalize features
        verbose: Print information
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = SurfaceFieldDataset(
        data_dir=data_dir,
        run_ids=train_ids,
        num_points=num_points,
        load_wss=load_wss,
        normalize=normalize,
        verbose=verbose,
    )
    
    val_dataset = SurfaceFieldDataset(
        data_dir=data_dir,
        run_ids=val_ids,
        num_points=num_points,
        load_wss=load_wss,
        normalize=normalize,
        verbose=verbose,
    )
    
    test_dataset = SurfaceFieldDataset(
        data_dir=data_dir,
        run_ids=test_ids,
        num_points=num_points,
        load_wss=load_wss,
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
        val_dataset.wss_mean = train_dataset.wss_mean
        val_dataset.wss_std = train_dataset.wss_std
        
        test_dataset.coords_mean = train_dataset.coords_mean
        test_dataset.coords_std = train_dataset.coords_std
        test_dataset.normals_mean = train_dataset.normals_mean
        test_dataset.normals_std = train_dataset.normals_std
        test_dataset.geo_params_mean = train_dataset.geo_params_mean
        test_dataset.geo_params_std = train_dataset.geo_params_std
        test_dataset.pressure_mean = train_dataset.pressure_mean
        test_dataset.pressure_std = train_dataset.pressure_std
        test_dataset.wss_mean = train_dataset.wss_mean
        test_dataset.wss_std = train_dataset.wss_std
    
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
    num_points: Optional[int] = None,
    load_wss: bool = False,
    batch_size: int = 8,
    num_workers: int = 4,
    normalize: bool = True,
    verbose: bool = False,
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
        num_points=num_points,
        load_wss=load_wss,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize,
        verbose=verbose,
    )
    
    # Export normalization constants from training dataset
    PRESSURE_MEAN = train_loader.dataset.pressure_mean.item()
    PRESSURE_STD = train_loader.dataset.pressure_std.item()
    
    return train_loader, val_loader, test_loader


def create_subset(
    dataset: SurfaceFieldDataset,
    run_ids: List[int],
) -> SurfaceFieldDataset:
    """
    Create a subset of the dataset with specific run IDs.
    
    Args:
        dataset: Full dataset
        run_ids: List of run IDs to include
    
    Returns:
        New dataset containing only specified run IDs
    """
    # Create new dataset with same parameters
    subset = SurfaceFieldDataset(
        data_dir=dataset.data_dir,
        run_ids=run_ids,
        num_points=dataset.num_points,
        load_wss=dataset.load_wss,
        normalize=False,  # Don't normalize yet
        verbose=False,
    )
    
    # Share normalization stats if available
    if dataset.pressure_mean is not None:
        subset.pressure_mean = dataset.pressure_mean
        subset.pressure_std = dataset.pressure_std
        subset.geo_params_mean = dataset.geo_params_mean
        subset.geo_params_std = dataset.geo_params_std
        if dataset.load_wss:
            subset.wss_mean = dataset.wss_mean
            subset.wss_std = dataset.wss_std
    
    return subset


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
        num_points=5000,
        load_wss=True,
        batch_size=8,
        num_workers=0,
        normalize=True,
        verbose=True,
    )
    
    # Test loading
    print("\nTesting data loading...")
    for batch in train_loader:
        print(f"Batch size: {len(batch['run_id'])}")
        print(f"Positions shape: {batch['surface_position_vtp'].shape}")
        print(f"Normals shape: {batch['surface_normals'].shape}")
        print(f"Pressure shape: {batch['surface_pressure'].shape}")
        print(f"Geo params shape: {batch['geo_params'].shape}")
        if 'surface_wallshearstress' in batch:
            print(f"WSS shape: {batch['surface_wallshearstress'].shape}")
        break
    
    print("\nAB-UPT data loader ready!")
