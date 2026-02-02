#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading utilities for Transolver++ on DrivAerML surface fields.

Adapted from DrivAerNet++ to load VTP files following XAeroNet preprocessor.py pattern.
Loads point cloud data with surface normals for physics-aware transformer processing.

Reference: Luo, H. et al. Transolver++: An accurate neural solver for pdes on 
           million-scale geometries. arXiv preprint arXiv:2502.02414 (2025).

@author: Transolver++ Implementation for DrivAerML
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import logging

try:
    import pyvista as pv
    import vtk
except ImportError:
    raise ImportError(
        "pyvista and vtk are required for loading DrivAerML VTP files. "
        "Install with: pip install pyvista vtk"
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PointCloudNormalDataset(Dataset):
    """
    Dataset that loads surface point clouds with normals for Transolver++.
    
    DrivAerML Data structure (following XAeroNet preprocessor.py pattern):
        - VTP files: run_{id}/boundary_{id}.vtp
        - Data extraction: cell_data -> point_data conversion
        - Point coordinates: (x, y, z) at mesh vertices
        - Point normals: (nx, ny, nz) computed at vertices
        - Target: pMeanTrim (pressure at points)
    
    Transolver++ expects 6D input per point:
    - Position: [x, y, z] (3D spatial coordinates)
    - Normal: [n_x, n_y, n_z] (surface normal vectors)
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
        
        # Normalization statistics
        self.mean = None
        self.std = None
        self.pressure_mean = None
        self.pressure_std = None
        
        # Verify data exists
        self._verify_data()
        
        if self.verbose:
            logging.info(f"PointCloudNormalDataset initialized with {len(self.run_ids)} runs from DrivAerML")
    
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
            logging.info("Computing normalization statistics from DrivAerML data...")
        
        all_positions = []
        all_normals = []
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
            
            # Step 3: Extract from point data
            points = surf.points  # (N, 3)
            surf_n = surf.compute_normals(point_normals=True, cell_normals=False)
            normals = surf_n.point_data["Normals"]  # (N, 3)
            
            # Extract pressure from point_data
            pressure_field = None
            for field_name in ["pMeanTrim", "CpMeanTrim", "pressure", "p"]:
                if field_name in surf.point_data:
                    pressure_field = field_name
                    break
            
            if pressure_field is None:
                continue
            
            pressure = surf.point_data[pressure_field]
            if len(pressure.shape) == 1:
                pressure = pressure[:, None]
            
            all_positions.append(points)
            all_normals.append(normals)
            all_pressures.append(pressure)
        
        all_positions = np.concatenate(all_positions, axis=0)
        all_normals = np.concatenate(all_normals, axis=0)
        all_pressures = np.concatenate(all_pressures, axis=0)
        
        # Compute statistics
        self.pos_mean = torch.tensor(all_positions.mean(axis=0), dtype=torch.float32)
        self.pos_std = torch.tensor(all_positions.std(axis=0) + 1e-8, dtype=torch.float32)
        self.normal_mean = torch.tensor(all_normals.mean(axis=0), dtype=torch.float32)
        self.normal_std = torch.tensor(all_normals.std(axis=0) + 1e-8, dtype=torch.float32)
        self.pressure_mean = torch.tensor(all_pressures.mean(), dtype=torch.float32)
        self.pressure_std = torch.tensor(all_pressures.std() + 1e-8, dtype=torch.float32)
        
        # Combined mean/std for 6D features [x,y,z,nx,ny,nz]
        self.mean = torch.cat([self.pos_mean, self.normal_mean])
        self.std = torch.cat([self.pos_std, self.normal_std])
        
        if self.verbose:
            logging.info(f"Position mean: {self.pos_mean}")
            logging.info(f"Position std: {self.pos_std}")
            logging.info(f"Pressure mean: {self.pressure_mean:.4f}, std: {self.pressure_std:.4f}")
    
    def __len__(self) -> int:
        return len(self.run_ids)
    
    def __getitem__(self, idx: int):
        """
        Get a sample.
        
        Returns:
            Dictionary with:
                - positions: (N, 3) coordinates
                - normals: (N, 3) surface normal vectors
                - features: (N, 6) concatenated [positions, normals]
                - pressures: (N,) pressure values
                - run_id: identifier
        """
        run_id = self.run_ids[idx]
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
        
        # Step 3: Extract from point data
        points = surf.points  # (N, 3)
        surf_n = surf.compute_normals(point_normals=True, cell_normals=False)
        normals = surf_n.point_data["Normals"]  # (N, 3)
        
        # Extract pressure from point_data
        pressure_field = None
        for field_name in ["pMeanTrim", "CpMeanTrim", "pressure", "p"]:
            if field_name in surf.point_data:
                pressure_field = field_name
                break
        
        if pressure_field is None:
            raise ValueError(
                f"No pressure field found in point_data. Available fields: {list(surf.point_data.keys())}"
            )
        
        pressures = surf.point_data[pressure_field]
        
        # Convert to tensors
        positions = torch.tensor(points, dtype=torch.float32)
        normals_tensor = torch.tensor(normals, dtype=torch.float32)
        pressures_tensor = torch.tensor(pressures, dtype=torch.float32)
        
        # Normalize
        if self.normalize:
            if self.mean is None:
                self.compute_normalization_stats()
            
            positions = (positions - self.pos_mean) / self.pos_std
            normals_tensor = (normals_tensor - self.normal_mean) / self.normal_std
            pressures_tensor = (pressures_tensor - self.pressure_mean) / self.pressure_std
        
        # Concatenate positions and normals for 6D input
        features = torch.cat([positions, normals_tensor], dim=1)  # (N, 6)
        
        sample = {
            'positions': positions,  # (N, 3)
            'normals': normals_tensor,  # (N, 3)
            'features': features,  # (N, 6)
            'pressures': pressures_tensor,  # (N,)
            'run_id': run_id,
        }
        
        return sample


def custom_collate_fn(batch):
    """Custom collate function that handles variable-sized point clouds."""
    if len(batch) == 0:
        return None
    
    return {
        'positions': [b['positions'] for b in batch],  # List of (N, 3)
        'normals': [b['normals'] for b in batch],      # List of (N, 3)
        'features': [b['features'] for b in batch],    # List of (N, 6)
        'pressures': [b['pressures'] for b in batch],  # List of (N,)
        'run_ids': [b['run_id'] for b in batch],
    }


def create_dataloaders(
    data_dir: str,
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    batch_size: int = 8,
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
        batch_size: Batch size
        num_workers: Number of data loading workers
        normalize: Whether to normalize features
        verbose: Print information
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = PointCloudNormalDataset(
        data_dir=data_dir,
        run_ids=train_ids,
        normalize=normalize,
        verbose=verbose,
    )
    
    val_dataset = PointCloudNormalDataset(
        data_dir=data_dir,
        run_ids=val_ids,
        normalize=normalize,
        verbose=verbose,
    )
    
    test_dataset = PointCloudNormalDataset(
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
        val_dataset.pos_mean = train_dataset.pos_mean
        val_dataset.pos_std = train_dataset.pos_std
        val_dataset.normal_mean = train_dataset.normal_mean
        val_dataset.normal_std = train_dataset.normal_std
        val_dataset.pressure_mean = train_dataset.pressure_mean
        val_dataset.pressure_std = train_dataset.pressure_std
        test_dataset.mean = train_dataset.mean
        test_dataset.std = train_dataset.std
        test_dataset.pos_mean = train_dataset.pos_mean
        test_dataset.pos_std = train_dataset.pos_std
        test_dataset.normal_mean = train_dataset.normal_mean
        test_dataset.normal_std = train_dataset.normal_std
        test_dataset.pressure_mean = train_dataset.pressure_mean
        test_dataset.pressure_std = train_dataset.pressure_std
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )
    
    if verbose:
        logging.info(f"\nDataLoader Statistics:")
        logging.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        logging.info(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
        logging.info(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
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


if __name__ == "__main__":
    logging.info("Testing Transolver++ Data Loader for DrivAerML...")
    
    # Example usage
    from pathlib import Path
    
    data_dir = "path/to/DrivAerML"
    split_dir = "path/to/DrivAerML/train_val_test_splits"
    
    # Load IDs
    train_ids, val_ids, test_ids = load_run_ids(split_dir)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        batch_size=4,
        num_workers=0,
        normalize=True,
        verbose=True,
    )
    
    # Test loading
    for batch in train_loader:
        logging.info(f"Batch size: {len(batch['features'])}")
        logging.info(f"First sample features shape: {batch['features'][0].shape}")
        logging.info(f"First sample pressures shape: {batch['pressures'][0].shape}")
        break
    
    logging.info("✓ Transolver++ data loader test passed for DrivAerML!")
