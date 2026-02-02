#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading utilities for Transolver++ on DrivAerNet++ surface fields.

This module loads point cloud data with surface normals for physics-aware
transformer processing.

Reference: Luo, H. et al. Transolver++: An accurate neural solver for pdes on 
           million-scale geometries. arXiv preprint arXiv:2502.02414 (2025).

@author: Transolver++ Implementation for DrivAerNet
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import pyvista as pv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PointCloudNormalDataset(Dataset):
    """
    Dataset that loads surface point clouds with normals for Transolver++.
    
    Transolver++ expects 6D input per point:
    - Position: [x, y, z] (3D spatial coordinates)
    - Normal: [n_x, n_y, n_z] (surface normal vectors)
    
    Args:
        root_dir: Directory containing VTK files
        num_points: Number of points to sample (default: 10000)
        preprocess: Whether to preprocess and cache data
        cache_dir: Directory for cached data
        compute_normals: Whether to compute normals if not present
    """
    
    def __init__(
        self,
        root_dir: str,
        num_points: int = 10000,
        preprocess: bool = False,
        cache_dir: str = None,
        compute_normals: bool = True,
    ):
        self.root_dir = root_dir
        self.vtk_files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith('.vtk')
        ])
        self.num_points = num_points
        self.preprocess = preprocess
        self.cache_dir = cache_dir if cache_dir else os.path.join(root_dir, "processed_data_transolver")
        self.compute_normals = compute_normals
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        logging.info(f"Found {len(self.vtk_files)} VTK files in {root_dir}")
        logging.info(f"Sampling {num_points} points per mesh with normals")
    
    def __len__(self):
        return len(self.vtk_files)
    
    def _get_cache_path(self, vtk_file_path):
        """Get cache file path."""
        base_name = os.path.basename(vtk_file_path).replace('.vtk', f'_pts{self.num_points}.npz')
        return os.path.join(self.cache_dir, base_name)
    
    def _compute_surface_normals(self, mesh):
        """Compute surface normals if not present."""
        if 'Normals' not in mesh.point_data:
            logging.debug("Computing surface normals...")
            mesh = mesh.compute_normals(point_normals=True, cell_normals=False)
        return mesh
    
    def _normalize_coordinates(self, positions):
        """Normalize coordinates to [0, 1] range."""
        min_coords = positions.min(axis=0)
        max_coords = positions.max(axis=0)
        
        # Add small epsilon to avoid division by zero
        ranges = max_coords - min_coords
        ranges = np.where(ranges < 1e-8, 1.0, ranges)
        
        normalized = (positions - min_coords) / ranges
        
        return normalized, (min_coords, max_coords)
    
    def _sample_points(self, mesh, n_points):
        """
        Sample points from mesh with normals and pressure values.
        
        Returns:
            positions: (N, 3) array
            normals: (N, 3) array
            pressures: (N,) array
            bbox: (min_coords, max_coords) tuple
        """
        if mesh.n_points > n_points:
            indices = np.random.choice(mesh.n_points, n_points, replace=False)
        else:
            indices = np.arange(mesh.n_points)
        
        # Get positions
        positions = mesh.points[indices].copy()
        
        # Normalize positions
        normalized_positions, bbox = self._normalize_coordinates(positions)
        
        # Get normals
        if 'Normals' in mesh.point_data:
            normals = mesh.point_data['Normals'][indices].copy()
        else:
            logging.warning("Normals not found, using zero vectors")
            normals = np.zeros_like(positions)
        
        # Normalize normal vectors (should already be normalized, but ensure)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        normals = normals / norms
        
        # Get pressure data
        pressure_key = 'p' if 'p' in mesh.point_data else 'pressure'
        if pressure_key not in mesh.point_data:
            raise ValueError(f"No pressure data found. Available keys: {list(mesh.point_data.keys())}")
        
        pressures = mesh.point_data[pressure_key][indices]
        if len(pressures.shape) > 1:
            pressures = pressures.flatten()
        
        return normalized_positions, normals, pressures, bbox
    
    def _save_to_cache(self, cache_path, positions, normals, pressures, bbox):
        """Save processed data to cache."""
        np.savez_compressed(
            cache_path,
            positions=positions,
            normals=normals,
            pressures=pressures,
            bbox_min=bbox[0],
            bbox_max=bbox[1],
        )
    
    def _load_from_cache(self, cache_path):
        """Load processed data from cache."""
        data = np.load(cache_path)
        return (
            data['positions'],
            data['normals'],
            data['pressures'],
            (data['bbox_min'], data['bbox_max'])
        )
    
    def __getitem__(self, idx):
        """
        Get a sample.
        
        Returns:
            Dictionary with:
                - positions: (N, 3) normalized coordinates
                - normals: (N, 3) surface normal vectors
                - features: (N, 6) concatenated [positions, normals]
                - pressures: (N,) pressure values
                - design_id: identifier
                - bbox: bounding box for denormalization
        """
        vtk_file_path = self.vtk_files[idx]
        cache_path = self._get_cache_path(vtk_file_path)
        
        # Try to load from cache
        if os.path.exists(cache_path):
            positions, normals, pressures, bbox = self._load_from_cache(cache_path)
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
            
            # Compute normals if needed
            if self.compute_normals:
                mesh = self._compute_surface_normals(mesh)
            
            # Sample points
            positions, normals, pressures, bbox = self._sample_points(mesh, self.num_points)
            
            # Cache
            self._save_to_cache(cache_path, positions, normals, pressures, bbox)
        
        # Concatenate positions and normals for 6D input
        features = np.concatenate([positions, normals], axis=1).astype(np.float32)
        
        # Convert to tensors
        sample = {
            'positions': torch.tensor(positions, dtype=torch.float32),
            'normals': torch.tensor(normals, dtype=torch.float32),
            'features': torch.tensor(features, dtype=torch.float32),  # (N, 6)
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
        'positions': [b['positions'] for b in batch],  # List of (N, 3)
        'normals': [b['normals'] for b in batch],      # List of (N, 3)
        'features': [b['features'] for b in batch],    # List of (N, 6)
        'pressures': [b['pressures'] for b in batch],  # List of (N,)
        'design_ids': [b['design_id'] for b in batch],
        'bboxes': [b['bbox'] for b in batch],
    }


def get_dataloaders(
    dataset_path: str,
    subset_dir: str,
    num_points: int = 10000,
    batch_size: int = 8,
    cache_dir: str = None,
    num_workers: int = 4,
) -> tuple:
    """
    Create dataloaders.
    
    Args:
        dataset_path: Path to VTK files
        subset_dir: Directory with train/val/test ID files
        num_points: Points to sample from surface
        batch_size: Batch size
        cache_dir: Cache directory
        num_workers: Number of workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create dataset
    full_dataset = PointCloudNormalDataset(
        root_dir=dataset_path,
        num_points=num_points,
        preprocess=True,
        cache_dir=cache_dir,
        compute_normals=True,
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
    dataset = PointCloudNormalDataset(
        root_dir='../PressureVTK',
        num_points=10000,
        preprocess=True,
        compute_normals=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        if sample:
            print(f"\nSample keys: {sample.keys()}")
            print(f"Positions shape: {sample['positions'].shape}")
            print(f"Normals shape: {sample['normals'].shape}")
            print(f"Features shape: {sample['features'].shape}")
            print(f"Pressures shape: {sample['pressures'].shape}")
            print(f"Design ID: {sample['design_id']}")
            print(f"\nFeature statistics:")
            print(f"  Position range: [{sample['positions'].min():.3f}, {sample['positions'].max():.3f}]")
            print(f"  Normal range: [{sample['normals'].min():.3f}, {sample['normals'].max():.3f}]")
