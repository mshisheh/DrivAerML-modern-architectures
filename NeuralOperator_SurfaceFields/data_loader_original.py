#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading utilities for Fourier Neural Operator on DrivAerNet++ surface fields.

This module converts point cloud data to voxel grids with occupancy and coordinates,
as required by the FNO architecture.

Reference: Li, Z. et al. Fourier neural operator for parametric partial differential equations. 
           arXiv preprint arXiv:2010.08895 (2020).

@author: NeuralOperator Implementation for DrivAerNet
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import pyvista as pv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VoxelGridDataset(Dataset):
    """
    Dataset that converts surface mesh data to voxel grids for Fourier Neural Operator.
    
    The FNO expects a regular grid input with 4 channels:
    1. Occupancy scalar field (1 inside geometry, 0 outside)
    2-4. Normalized grid coordinates (x, y, z)
    
    Args:
        root_dir: Directory containing VTK files
        grid_resolution: Resolution of voxel grid (e.g., 32 for 32^3)
        num_points: Number of points to sample from mesh for training targets
        preprocess: Whether to preprocess and cache data
        cache_dir: Directory for cached data
    """
    
    def __init__(
        self,
        root_dir: str,
        grid_resolution: int = 32,
        num_points: int = 10000,
        preprocess: bool = False,
        cache_dir: str = None,
    ):
        self.root_dir = root_dir
        self.vtk_files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith('.vtk')
        ])
        self.grid_resolution = grid_resolution
        self.num_points = num_points
        self.preprocess = preprocess
        self.cache_dir = cache_dir if cache_dir else os.path.join(root_dir, "processed_data_fno")
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        logging.info(f"Found {len(self.vtk_files)} VTK files in {root_dir}")
        logging.info(f"Grid resolution: {grid_resolution}^3 = {grid_resolution**3} voxels")
    
    def __len__(self):
        return len(self.vtk_files)
    
    def _get_cache_path(self, vtk_file_path):
        """Get cache file path."""
        base_name = os.path.basename(vtk_file_path).replace('.vtk', f'_grid{self.grid_resolution}.npz')
        return os.path.join(self.cache_dir, base_name)
    
    def _compute_bounding_box(self, points):
        """Compute bounding box with some padding."""
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        
        # Add 10% padding
        ranges = max_coords - min_coords
        min_coords -= 0.1 * ranges
        max_coords += 0.1 * ranges
        
        return min_coords, max_coords
    
    def _voxelize_mesh(self, mesh, resolution):
        """
        Convert mesh to voxel grid with occupancy and coordinates.
        
        Returns:
            voxel_grid: (4, H, W, D) tensor with occupancy + normalized coordinates
        """
        points = mesh.points
        min_coords, max_coords = self._compute_bounding_box(points)
        
        # Create grid
        x = np.linspace(min_coords[0], max_coords[0], resolution)
        y = np.linspace(min_coords[1], max_coords[1], resolution)
        z = np.linspace(min_coords[2], max_coords[2], resolution)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        # Check occupancy (point inside mesh)
        # For performance, we use a simplified approach: 
        # mark voxels that contain points from the mesh
        occupancy = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        
        # Discretize mesh points to voxels
        normalized_points = (points - min_coords) / (max_coords - min_coords)
        voxel_indices = (normalized_points * (resolution - 1)).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, resolution - 1)
        
        # Mark occupied voxels
        occupancy[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.0
        
        # Create normalized coordinate grids
        x_norm = (X - min_coords[0]) / (max_coords[0] - min_coords[0])
        y_norm = (Y - min_coords[1]) / (max_coords[1] - min_coords[1])
        z_norm = (Z - min_coords[2]) / (max_coords[2] - min_coords[2])
        
        # Stack into 4-channel input (occupancy + 3D coordinates)
        voxel_grid = np.stack([occupancy, x_norm, y_norm, z_norm], axis=0).astype(np.float32)
        
        return voxel_grid, min_coords, max_coords
    
    def _sample_surface_points(self, mesh, n_points):
        """Sample points from surface with pressure values."""
        if mesh.n_points > n_points:
            indices = np.random.choice(mesh.n_points, n_points, replace=False)
        else:
            indices = np.arange(mesh.n_points)
        
        positions = mesh.points[indices]
        
        # Get pressure data
        pressure_key = 'p' if 'p' in mesh.point_data else 'pressure'
        if pressure_key not in mesh.point_data:
            raise ValueError(f"No pressure data found. Available keys: {list(mesh.point_data.keys())}")
        
        pressures = mesh.point_data[pressure_key][indices]
        if len(pressures.shape) > 1:
            pressures = pressures.flatten()
        
        return positions, pressures
    
    def _save_to_cache(self, cache_path, voxel_grid, positions, pressures, bbox):
        """Save processed data to cache."""
        np.savez_compressed(
            cache_path,
            voxel_grid=voxel_grid,
            positions=positions,
            pressures=pressures,
            bbox_min=bbox[0],
            bbox_max=bbox[1],
        )
    
    def _load_from_cache(self, cache_path):
        """Load processed data from cache."""
        data = np.load(cache_path)
        return (
            data['voxel_grid'],
            data['positions'],
            data['pressures'],
            (data['bbox_min'], data['bbox_max'])
        )
    
    def __getitem__(self, idx):
        """
        Get a sample.
        
        Returns:
            Dictionary with:
                - voxel_grid: (4, H, W, D) input grid
                - positions: (N, 3) surface point positions
                - pressures: (N,) pressure values at positions
                - design_id: identifier
        """
        vtk_file_path = self.vtk_files[idx]
        cache_path = self._get_cache_path(vtk_file_path)
        
        # Try to load from cache
        if os.path.exists(cache_path):
            voxel_grid, positions, pressures, bbox = self._load_from_cache(cache_path)
        else:
            if not self.preprocess:
                logging.error(f"Cache not found for {vtk_file_path} and preprocessing disabled.")
                return None
            
            # Load mesh
            try:
                mesh = pv.read(vtk_file_path)
            except Exception as e:
                logging.error(f"Failed to load VTK file: {vtk_file_path}. Error: {e}")
                return None
            
            # Voxelize
            voxel_grid, bbox_min, bbox_max = self._voxelize_mesh(mesh, self.grid_resolution)
            
            # Sample surface points for targets
            positions, pressures = self._sample_surface_points(mesh, self.num_points)
            
            # Cache
            self._save_to_cache(cache_path, voxel_grid, positions, pressures, (bbox_min, bbox_max))
            bbox = (bbox_min, bbox_max)
        
        # Convert to tensors
        sample = {
            'voxel_grid': torch.tensor(voxel_grid, dtype=torch.float32),
            'positions': torch.tensor(positions, dtype=torch.float32),
            'pressures': torch.tensor(pressures, dtype=torch.float32),
            'design_id': os.path.basename(vtk_file_path).replace('.vtk', ''),
            'bbox': bbox,
        }
        
        return sample


def create_subset(dataset, ids_file):
    """Create subset from design IDs."""
    try:
        with open(ids_file, 'r') as file:
            subset_ids = [id_.strip() for id_ in file.readlines()]
        
        subset_files = [
            f for f in dataset.vtk_files
            if any(id_ in os.path.basename(f) for id_ in subset_ids)
        ]
        subset_indices = [dataset.vtk_files.index(f) for f in subset_files]
        
        if not subset_indices:
            logging.error(f"No matching VTK files found for IDs in {ids_file}")
            return None
        
        logging.info(f"Created subset with {len(subset_indices)} samples")
        return Subset(dataset, subset_indices)
    
    except FileNotFoundError as e:
        logging.error(f"Error loading subset file {ids_file}: {e}")
        return None


def custom_collate_fn(batch):
    """Custom collate function that filters None samples."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    return {
        'voxel_grid': torch.stack([b['voxel_grid'] for b in batch]),
        'positions': [b['positions'] for b in batch],  # Keep as list (variable size)
        'pressures': [b['pressures'] for b in batch],  # Keep as list
        'design_ids': [b['design_id'] for b in batch],
        'bboxes': [b['bbox'] for b in batch],
    }


def get_dataloaders(
    dataset_path: str,
    subset_dir: str,
    grid_resolution: int = 32,
    num_points: int = 10000,
    batch_size: int = 16,
    cache_dir: str = None,
    num_workers: int = 4,
) -> tuple:
    """
    Create dataloaders.
    
    Args:
        dataset_path: Path to VTK files
        subset_dir: Directory with train/val/test ID files
        grid_resolution: Voxel grid resolution
        num_points: Points to sample from surface
        batch_size: Batch size
        cache_dir: Cache directory
        num_workers: Number of workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create dataset
    full_dataset = VoxelGridDataset(
        root_dir=dataset_path,
        grid_resolution=grid_resolution,
        num_points=num_points,
        preprocess=True,
        cache_dir=cache_dir,
    )
    
    # Create subsets
    train_dataset = create_subset(full_dataset, os.path.join(subset_dir, 'train_design_ids.txt'))
    val_dataset = create_subset(full_dataset, os.path.join(subset_dir, 'val_design_ids.txt'))
    test_dataset = create_subset(full_dataset, os.path.join(subset_dir, 'test_design_ids.txt'))
    
    # Create loaders
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
    
    logging.info(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# Normalization constants
PRESSURE_MEAN = -94.5
PRESSURE_STD = 117.25


if __name__ == '__main__':
    # Test the data loader
    dataset = VoxelGridDataset(
        root_dir='../PressureVTK',
        grid_resolution=32,
        num_points=10000,
        preprocess=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        if sample:
            print(f"\nSample keys: {sample.keys()}")
            print(f"Voxel grid shape: {sample['voxel_grid'].shape}")
            print(f"Positions shape: {sample['positions'].shape}")
            print(f"Pressures shape: {sample['pressures'].shape}")
            print(f"Design ID: {sample['design_id']}")
