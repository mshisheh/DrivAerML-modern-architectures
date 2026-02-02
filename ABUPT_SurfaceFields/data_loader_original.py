#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading utilities for DrivAerNet++ surface field prediction using AB-UPT.

This module provides functionality for loading and preprocessing surface mesh data
with pressure and wall shear stress fields from VTK files.

@author: AB-UPT Adaptation for DrivAerNet
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import pyvista as pv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SurfaceFieldDataset(Dataset):
    """
    Dataset class for loading surface mesh data with pressure and wall shear stress fields.
    
    This dataset is designed for AB-UPT which expects:
    - Surface positions (geometry)
    - Surface pressure fields
    - Surface wall shear stress fields (optional)
    
    Args:
        root_dir: Directory containing VTK files
        num_points: Number of points to sample from each mesh
        preprocess: Whether to preprocess and cache data
        cache_dir: Directory to store cached preprocessed data
        load_wss: Whether to load wall shear stress data (if available)
    """
    
    def __init__(
        self, 
        root_dir: str, 
        num_points: int,
        preprocess: bool = False,
        cache_dir: str = None,
        load_wss: bool = False,
    ):
        self.root_dir = root_dir
        self.vtk_files = sorted([
            os.path.join(root_dir, f) 
            for f in os.listdir(root_dir) 
            if f.endswith('.vtk')
        ])
        self.num_points = num_points
        self.preprocess = preprocess
        self.cache_dir = cache_dir if cache_dir else os.path.join(root_dir, "processed_data_abupt")
        self.load_wss = load_wss
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        logging.info(f"Found {len(self.vtk_files)} VTK files in {root_dir}")
    
    def __len__(self):
        return len(self.vtk_files)
    
    def _get_cache_path(self, vtk_file_path):
        """Get the corresponding .npz file path for a given .vtk file."""
        base_name = os.path.basename(vtk_file_path).replace('.vtk', '.npz')
        return os.path.join(self.cache_dir, base_name)
    
    def _save_to_cache(self, cache_path, positions, pressures, wss=None):
        """Save preprocessed data to cache."""
        data = {
            'positions': positions,
            'pressures': pressures,
        }
        if wss is not None:
            data['wss'] = wss
        np.savez_compressed(cache_path, **data)
    
    def _load_from_cache(self, cache_path):
        """Load preprocessed data from cache."""
        data = np.load(cache_path)
        positions = data['positions']
        pressures = data['pressures']
        wss = data.get('wss', None)
        return positions, pressures, wss
    
    def _sample_surface_data(self, mesh, n_points):
        """
        Sample n_points from the surface mesh with corresponding field data.
        
        Args:
            mesh: PyVista mesh with pressure data in point_data
            n_points: Number of points to sample
            
        Returns:
            Tuple of (positions, pressures, wss)
        """
        if mesh.n_points > n_points:
            indices = np.random.choice(mesh.n_points, n_points, replace=False)
        else:
            indices = np.arange(mesh.n_points)
            logging.info(f"Mesh has only {mesh.n_points} points. Using all available points.")
        
        # Sample positions
        sampled_positions = mesh.points[indices]
        
        # Sample pressure (assuming key 'p' or 'pressure')
        pressure_key = 'p' if 'p' in mesh.point_data else 'pressure'
        if pressure_key not in mesh.point_data:
            raise ValueError(f"No pressure data found in mesh. Available keys: {list(mesh.point_data.keys())}")
        
        sampled_pressures = mesh.point_data[pressure_key][indices]
        if len(sampled_pressures.shape) > 1:
            sampled_pressures = sampled_pressures.flatten()
        
        # Sample wall shear stress if requested
        sampled_wss = None
        if self.load_wss:
            wss_keys = ['wallShearStress', 'WSS', 'wall_shear_stress', 'tau']
            wss_key = None
            for key in wss_keys:
                if key in mesh.point_data:
                    wss_key = key
                    break
            
            if wss_key:
                sampled_wss = mesh.point_data[wss_key][indices]
                if len(sampled_wss.shape) == 1:
                    # If it's scalar, convert to 3D (unlikely for WSS but handle it)
                    sampled_wss = np.stack([sampled_wss, np.zeros_like(sampled_wss), np.zeros_like(sampled_wss)], axis=1)
                elif sampled_wss.shape[1] != 3:
                    logging.warning(f"WSS has unexpected shape: {sampled_wss.shape}, expected (N, 3)")
        
        return sampled_positions, sampled_pressures, sampled_wss
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary with keys:
                - 'surface_position_vtp': Surface positions (N, 3)
                - 'surface_pressure': Pressure values (N,)
                - 'surface_wallshearstress': WSS values (N, 3) if load_wss=True
                - 'design_id': Design identifier
        """
        vtk_file_path = self.vtk_files[idx]
        cache_path = self._get_cache_path(vtk_file_path)
        
        # Check cache
        if os.path.exists(cache_path):
            positions, pressures, wss = self._load_from_cache(cache_path)
        else:
            if not self.preprocess:
                logging.error(f"Cache file not found for {vtk_file_path} and preprocessing is disabled.")
                return None
            
            # Load and preprocess
            try:
                mesh = pv.read(vtk_file_path)
            except Exception as e:
                logging.error(f"Failed to load VTK file: {vtk_file_path}. Error: {e}")
                return None
            
            positions, pressures, wss = self._sample_surface_data(mesh, self.num_points)
            self._save_to_cache(cache_path, positions, pressures, wss)
        
        # Convert to tensors
        sample = {
            'surface_position_vtp': torch.tensor(positions, dtype=torch.float32),
            'surface_pressure': torch.tensor(pressures, dtype=torch.float32),
            'design_id': os.path.basename(vtk_file_path).replace('.vtk', ''),
        }
        
        if wss is not None:
            sample['surface_wallshearstress'] = torch.tensor(wss, dtype=torch.float32)
        
        return sample


def create_subset(dataset, ids_file):
    """
    Create a subset of the dataset based on design IDs from a file.
    
    Args:
        dataset: The full dataset
        ids_file: Path to file containing design IDs (one per line)
        
    Returns:
        Subset of the dataset
    """
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
        
        logging.info(f"Created subset with {len(subset_indices)} samples from {ids_file}")
        return Subset(dataset, subset_indices)
    
    except FileNotFoundError as e:
        logging.error(f"Error loading subset file {ids_file}: {e}")
        return None


def get_dataloaders(
    dataset_path: str,
    subset_dir: str,
    num_points: int,
    batch_size: int,
    cache_dir: str = None,
    num_workers: int = 4,
    load_wss: bool = False,
) -> tuple:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        dataset_path: Path to directory containing VTK files
        subset_dir: Directory containing train/val/test ID files
        num_points: Number of points to sample from each mesh
        batch_size: Batch size
        cache_dir: Directory for cached preprocessed data
        num_workers: Number of dataloader workers
        load_wss: Whether to load wall shear stress data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create full dataset
    full_dataset = SurfaceFieldDataset(
        root_dir=dataset_path,
        num_points=num_points,
        preprocess=True,
        cache_dir=cache_dir,
        load_wss=load_wss,
    )
    
    # Create subsets
    train_dataset = create_subset(full_dataset, os.path.join(subset_dir, 'train_design_ids.txt'))
    val_dataset = create_subset(full_dataset, os.path.join(subset_dir, 'val_design_ids.txt'))
    test_dataset = create_subset(full_dataset, os.path.join(subset_dir, 'test_design_ids.txt'))
    
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
    
    logging.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# Normalization constants (computed from DrivAerNet++ dataset)
# These should ideally be computed from your actual dataset
PRESSURE_MEAN = -94.5
PRESSURE_STD = 117.25

# Wall shear stress normalization (if available)
WSS_MEAN = np.array([0.0, 0.0, 0.0])  # To be computed from data
WSS_STD = np.array([1.0, 1.0, 1.0])    # To be computed from data


if __name__ == '__main__':
    # Test the data loader
    dataset = SurfaceFieldDataset(
        root_dir='../PressureVTK',  # Update with your path
        num_points=10000,
        preprocess=True,
        load_wss=False,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        if sample:
            print(f"\nSample keys: {sample.keys()}")
            print(f"Surface position shape: {sample['surface_position_vtp'].shape}")
            print(f"Pressure shape: {sample['surface_pressure'].shape}")
            print(f"Design ID: {sample['design_id']}")
