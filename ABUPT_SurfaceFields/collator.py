#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collator for AB-UPT surface field prediction on DrivAerNet++.

This collator prepares batches for the AB-UPT architecture to predict
surface pressure and wall shear stress fields.
"""

import sys
from pathlib import Path

# Add AB-UPT src directory to path
abupt_src = Path(__file__).parent.parent.parent / "anchored-branched-universal-physics-transformers" / "src"
sys.path.insert(0, str(abupt_src))

import torch
import numpy as np
from typing import List, Dict


class ABUPTSurfaceFieldCollator:
    """
    Collator for AB-UPT surface field prediction.
    
    Unlike the drag prediction task, here we predict fields (pressure, WSS) at query points.
    We use:
    - Geometry points: Subset of surface for encoding
    - Surface anchor points: Points where we make predictions
    - Query points: Same as anchors for training, can be different for inference
    
    Args:
        num_geometry_points: Points for geometry encoding
        num_surface_anchors: Anchor points for prediction
        num_geometry_supernodes: Number of supernodes
        use_query_positions: If True, split data into anchors (for attention) and queries (for prediction)
        seed: Random seed
    """
    
    def __init__(
        self,
        num_geometry_points: int = 8192,
        num_surface_anchors: int = 4096,
        num_geometry_supernodes: int = 512,
        use_query_positions: bool = False,
        seed: int = None,
    ):
        self.num_geometry_points = num_geometry_points
        self.num_surface_anchors = num_surface_anchors
        self.num_geometry_supernodes = num_geometry_supernodes
        self.use_query_positions = use_query_positions
        self.seed = seed
        
        # AB-UPT expects normalized positions in range [-0.5, 0.5] * scale
        # These values should be computed from the actual dataset
        self.raw_pos_min = torch.tensor([-3.0, -1.5, 0.0], dtype=torch.float32)
        self.raw_pos_max = torch.tensor([6.0, 1.5, 2.0], dtype=torch.float32)
        self.scale = 1000.0
    
    def normalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Normalize positions to AB-UPT's expected range."""
        normalized = (positions - self.raw_pos_min) / (self.raw_pos_max - self.raw_pos_min) - 0.5
        return normalized * self.scale
    
    def sample_points(self, points: torch.Tensor, num_points: int, seed: int = None) -> torch.Tensor:
        """Sample fixed number of points."""
        num_available = points.shape[0]
        
        if num_available >= num_points:
            if seed is not None:
                rng = np.random.RandomState(seed)
                indices = rng.choice(num_available, num_points, replace=False)
            else:
                indices = np.random.choice(num_available, num_points, replace=False)
            return indices
        else:
            if seed is not None:
                rng = np.random.RandomState(seed)
                indices = rng.choice(num_available, num_points, replace=True)
            else:
                indices = np.random.choice(num_available, num_points, replace=True)
            return indices
    
    def create_supernodes(self, points: torch.Tensor, num_supernodes: int, seed: int = None) -> torch.Tensor:
        """Create supernode assignments (simplified random assignment)."""
        num_points = points.shape[0]
        
        if seed is not None:
            rng = np.random.RandomState(seed)
            supernode_idx = torch.from_numpy(rng.randint(0, num_supernodes, size=(num_points,)))
        else:
            supernode_idx = torch.randint(0, num_supernodes, size=(num_points,))
        
        return supernode_idx
    
    def __call__(self, batch_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch for AB-UPT surface field prediction.
        
        Args:
            batch_list: List of samples from dataset
            
        Returns:
            Batch dictionary compatible with AB-UPT
        """
        # Filter out None samples
        batch_list = [b for b in batch_list if b is not None]
        if len(batch_list) == 0:
            return None
        
        batch_size = len(batch_list)
        
        # Lists to collect data
        geometry_positions = []
        geometry_supernode_idxs = []
        geometry_batch_idxs = []
        surface_anchor_positions = []
        surface_anchor_pressures = []
        surface_anchor_wallshearstress = []
        surface_query_positions = []
        surface_query_pressures = []
        surface_query_wallshearstress = []
        design_ids = []
        
        has_wss = 'surface_wallshearstress' in batch_list[0]
        
        for batch_idx, sample in enumerate(batch_list):
            surface_pos = sample['surface_position_vtp']  # (N, 3)
            surface_pressure = sample['surface_pressure']  # (N,)
            design_id = sample['design_id']
            
            # Get WSS if available
            if has_wss:
                surface_wss = sample['surface_wallshearstress']  # (N, 3)
            
            # Normalize positions
            surface_pos_norm = self.normalize_positions(surface_pos)
            
            # Sample geometry points
            seed_offset = (self.seed + batch_idx) if self.seed is not None else None
            geometry_indices = self.sample_points(
                surface_pos_norm,
                self.num_geometry_points,
                seed_offset
            )
            geometry_pos = surface_pos_norm[geometry_indices]
            
            # Create supernodes
            supernode_idx = self.create_supernodes(
                geometry_pos,
                self.num_geometry_supernodes,
                seed_offset
            )
            
            # Sample anchor/query points
            if self.use_query_positions:
                # Split into anchors and queries (for training with teacher forcing)
                num_points = surface_pos.shape[0]
                all_indices = np.arange(num_points)
                np.random.shuffle(all_indices)
                
                anchor_indices = all_indices[:self.num_surface_anchors]
                query_indices = all_indices[self.num_surface_anchors:2*self.num_surface_anchors]
                
                # Anchors
                anchor_pos = surface_pos_norm[anchor_indices]
                anchor_pressure = surface_pressure[anchor_indices]
                if has_wss:
                    anchor_wss = surface_wss[anchor_indices]
                
                # Queries
                query_pos = surface_pos_norm[query_indices]
                query_pressure = surface_pressure[query_indices]
                if has_wss:
                    query_wss = surface_wss[query_indices]
                
                surface_query_positions.append(query_pos)
                surface_query_pressures.append(query_pressure)
                if has_wss:
                    surface_query_wallshearstress.append(query_wss)
            else:
                # Use same points as both anchors and queries (standard training)
                anchor_indices = self.sample_points(
                    surface_pos_norm,
                    self.num_surface_anchors,
                    seed_offset + 1 if seed_offset else None
                )
                anchor_pos = surface_pos_norm[anchor_indices]
                anchor_pressure = surface_pressure[anchor_indices]
                if has_wss:
                    anchor_wss = surface_wss[anchor_indices]
            
            # Batch index for sparse tensors
            batch_idx_tensor = torch.full((geometry_pos.shape[0],), batch_idx, dtype=torch.long)
            
            # Collect data
            geometry_positions.append(geometry_pos)
            geometry_supernode_idxs.append(supernode_idx)
            geometry_batch_idxs.append(batch_idx_tensor)
            surface_anchor_positions.append(anchor_pos)
            surface_anchor_pressures.append(anchor_pressure)
            if has_wss:
                surface_anchor_wallshearstress.append(anchor_wss)
            design_ids.append(design_id)
        
        # Concatenate sparse tensors (geometry)
        geometry_position = torch.cat(geometry_positions, dim=0)
        geometry_supernode_idx = torch.cat(geometry_supernode_idxs, dim=0)
        geometry_batch_idx = torch.cat(geometry_batch_idxs, dim=0)
        
        # Stack dense tensors (surface anchors)
        surface_anchor_position = torch.stack(surface_anchor_positions, dim=0)
        surface_anchor_pressure = torch.stack(surface_anchor_pressures, dim=0)
        
        # Create batch dictionary
        batch = {
            'geometry_position': geometry_position,
            'geometry_supernode_idx': geometry_supernode_idx,
            'geometry_batch_idx': geometry_batch_idx,
            'surface_anchor_position': surface_anchor_position,
            'surface_anchor_pressure': surface_anchor_pressure,
            'design_ids': design_ids,
        }
        
        # Add WSS if available
        if has_wss:
            surface_anchor_wallshearstress = torch.stack(surface_anchor_wallshearstress, dim=0)
            batch['surface_anchor_wallshearstress'] = surface_anchor_wallshearstress
        
        # Add query data if using queries
        if self.use_query_positions and len(surface_query_positions) > 0:
            batch['surface_query_position'] = torch.stack(surface_query_positions, dim=0)
            batch['surface_query_pressure'] = torch.stack(surface_query_pressures, dim=0)
            if has_wss:
                batch['surface_query_wallshearstress'] = torch.stack(surface_query_wallshearstress, dim=0)
        
        # Add dummy volume data (AB-UPT expects both surface and volume branches)
        # We'll use the same surface data for volume to make architecture happy
        batch['volume_anchor_position'] = surface_anchor_position.clone()
        
        return batch


def test_collator():
    """Test the collator with dummy data."""
    collator = ABUPTSurfaceFieldCollator(
        num_geometry_points=1024,
        num_surface_anchors=512,
        num_geometry_supernodes=128,
        use_query_positions=False,
        seed=42,
    )
    
    # Create dummy samples
    batch_list = []
    for i in range(2):
        sample = {
            'surface_position_vtp': torch.randn(5000, 3),
            'surface_pressure': torch.randn(5000),
            'surface_wallshearstress': torch.randn(5000, 3),
            'design_id': f'Design_{i}',
        }
        batch_list.append(sample)
    
    # Collate
    batch = collator(batch_list)
    
    print("Collated batch:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")


if __name__ == '__main__':
    test_collator()
