"""
Data Loader for Transolver on DrivAerNet

Prepares surface mesh data for Transolver training.
Handles normalization and creates PyG Data objects.

Author: Implementation for DrivAerNet benchmark
Date: February 2026
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from typing import List, Tuple, Optional


class TransolverDataset(Dataset):
    """
    Dataset for Transolver surface pressure prediction.
    
    Data structure:
        - Surface points: (x, y, z) coordinates
        - Surface normals: (nx, ny, nz)
        - Point areas: Approximate surface area per point
        - Target: Pressure coefficient (Cp)
    """
    
    def __init__(
        self,
        data_dir: str,
        design_ids: List[str],
        normalize: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            data_dir: Root directory containing DrivAerNet data
            design_ids: List of design IDs to load
            normalize: Whether to normalize features
            verbose: Print loading information
        """
        self.data_dir = data_dir
        self.design_ids = design_ids
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
            print(f"TransolverDataset initialized with {len(self.design_ids)} designs")
    
    def _verify_data(self):
        """Check that data files exist"""
        missing = []
        for design_id in self.design_ids[:min(5, len(self.design_ids))]:
            data_file = os.path.join(self.data_dir, f"{design_id}.npy")
            if not os.path.exists(data_file):
                missing.append(design_id)
        
        if missing:
            raise FileNotFoundError(
                f"Missing data files for designs: {missing}\n"
                f"Expected location: {self.data_dir}"
            )
    
    def compute_normalization_stats(self):
        """Compute mean and std for normalization"""
        if self.verbose:
            print("Computing normalization statistics...")
        
        all_features = []
        all_pressures = []
        
        for idx in range(min(100, len(self))):  # Sample subset for stats
            data = np.load(os.path.join(self.data_dir, f"{self.design_ids[idx]}.npy"))
            features = data[:, :7]  # x, y, z, nx, ny, nz, area
            pressure = data[:, 7:8]  # Cp
            
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
        return len(self.design_ids)
    
    def __getitem__(self, idx: int) -> Data:
        """
        Load and return a single sample.
        
        Returns:
            PyG Data object with:
                - x: [num_nodes, 7] - features (x, y, z, nx, ny, nz, area)
                - pos: [num_nodes, 3] - positions (x, y, z)
                - y: [num_nodes, 1] - target pressure
                - design_id: str - design identifier
        """
        design_id = self.design_ids[idx]
        data_file = os.path.join(self.data_dir, f"{design_id}.npy")
        
        # Load data: [x, y, z, nx, ny, nz, area, Cp]
        data_np = np.load(data_file)
        
        # Features and target
        features = torch.tensor(data_np[:, :7], dtype=torch.float32)
        pressure = torch.tensor(data_np[:, 7:8], dtype=torch.float32)
        positions = features[:, :3]  # x, y, z
        
        # Normalize
        if self.normalize:
            if self.mean is None:
                self.compute_normalization_stats()
            
            features = (features - self.mean) / self.std
            pressure = (pressure - self.pressure_mean) / self.pressure_std
        
        # Create PyG Data object
        data = Data(
            x=features,
            pos=positions,  # Original positions (not normalized)
            y=pressure,
        )
        data.design_id = design_id
        
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
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    batch_size: int = 1,  # GraphCast typically uses batch_size=1
    num_workers: int = 4,
    normalize: bool = True,
    verbose: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root directory containing DrivAerNet data
        train_ids: List of training design IDs
        val_ids: List of validation design IDs
        test_ids: List of test design IDs
        batch_size: Batch size (typically 1 for GraphCast)
        num_workers: Number of data loading workers
        normalize: Whether to normalize features
        verbose: Print information
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = TransolverDataset(
        data_dir=data_dir,
        design_ids=train_ids,
        normalize=normalize,
        verbose=verbose,
    )
    
    val_dataset = TransolverDataset(
        data_dir=data_dir,
        design_ids=val_ids,
        normalize=normalize,
        verbose=verbose,
    )
    
    test_dataset = TransolverDataset(
        data_dir=data_dir,
        design_ids=test_ids,
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


def load_design_ids(split_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Load train/val/test design IDs from text files.
    
    Args:
        split_dir: Directory containing train/val/test split files
    
    Returns:
        train_ids, val_ids, test_ids
    """
    def read_ids(filename):
        with open(os.path.join(split_dir, filename), 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    train_ids = read_ids('train_design_ids.txt')
    val_ids = read_ids('val_design_ids.txt')
    test_ids = read_ids('test_design_ids.txt')
    
    return train_ids, val_ids, test_ids


if __name__ == "__main__":
    print("Testing GraphCast Data Loader...\n")
    
    # Example usage
    data_dir = "path/to/DrivAerNet/surface_field_data"
    split_dir = "path/to/DrivAerNet/train_val_test_splits"
    
    # For testing, create dummy data
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    temp_split_dir = tempfile.mkdtemp()
    
    try:
        # Create dummy data files
        dummy_ids = [f"design_{i:04d}" for i in range(10)]
        for design_id in dummy_ids:
            # Create dummy surface data: [x, y, z, nx, ny, nz, area, Cp]
            num_points = np.random.randint(40000, 60000)
            data = np.random.randn(num_points, 8).astype(np.float32)
            np.save(os.path.join(temp_dir, f"{design_id}.npy"), data)
        
        # Create dummy split files
        with open(os.path.join(temp_split_dir, 'train_design_ids.txt'), 'w') as f:
            f.write('\n'.join(dummy_ids[:6]))
        with open(os.path.join(temp_split_dir, 'val_design_ids.txt'), 'w') as f:
            f.write('\n'.join(dummy_ids[6:8]))
        with open(os.path.join(temp_split_dir, 'test_design_ids.txt'), 'w') as f:
            f.write('\n'.join(dummy_ids[8:]))
        
        # Load design IDs
        train_ids, val_ids, test_ids = load_design_ids(temp_split_dir)
        print(f"Loaded {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test IDs")
        
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
            print(f"  Design ID: {data.design_id}")
            break
        
        print("\n✓ GraphCast data loader test passed!")
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        shutil.rmtree(temp_split_dir)
