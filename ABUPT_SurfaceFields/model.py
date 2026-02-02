#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AB-UPT model wrapper for surface field prediction on DrivAerNet++.

This module adapts the AB-UPT architecture for predicting surface pressure
and wall shear stress fields on car geometries.
"""

import sys
from pathlib import Path

# Add AB-UPT src directory to path
abupt_src = Path(__file__).parent.parent.parent / "anchored-branched-universal-physics-transformers" / "src"
sys.path.insert(0, str(abupt_src))

import torch
import torch.nn as nn

# Import AB-UPT model
from model import AnchoredBranchedUPT


class ABUPTSurfaceFieldPredictor(nn.Module):
    """
    AB-UPT wrapper for surface field prediction.
    
    This model uses the full AB-UPT architecture to predict:
    - Surface pressure (scalar field)
    - Surface wall shear stress (vector field) - optional
    
    Unlike the drag prediction task, this outputs fields at each query point,
    which is the original design intent of AB-UPT.
    
    Args:
        dim: Hidden dimension
        geometry_depth: Number of transformer blocks for geometry encoding
        num_heads: Number of attention heads
        blocks: Block architecture string (e.g., "pscscs")
        num_surface_blocks: Number of surface-specific blocks
        num_volume_blocks: Number of volume-specific blocks (not used but required)
        radius: Radius for supernode pooling
        predict_wss: Whether to predict wall shear stress in addition to pressure
    """
    
    def __init__(
        self,
        dim: int = 256,
        geometry_depth: int = 2,
        num_heads: int = 8,
        blocks: str = "pscs",
        num_surface_blocks: int = 6,
        num_volume_blocks: int = 2,  # Minimal since we focus on surface
        radius: float = 0.25,
        predict_wss: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        # Determine output dimensions
        # Pressure: 1D scalar field
        # Wall shear stress: 3D vector field
        output_dim_surface = 4 if predict_wss else 1  # 1 (pressure) + 3 (wss) or just 1
        output_dim_volume = 1  # Dummy output for volume branch
        
        # Create AB-UPT model
        self.model = AnchoredBranchedUPT(
            ndim=3,
            input_dim=3,
            output_dim_surface=output_dim_surface,
            output_dim_volume=output_dim_volume,
            dim=dim,
            geometry_depth=geometry_depth,
            num_heads=num_heads,
            blocks=blocks,
            num_surface_blocks=num_surface_blocks,
            num_volume_blocks=num_volume_blocks,
            radius=radius,
            **kwargs,
        )
        
        self.predict_wss = predict_wss
    
    def forward(self, batch: dict) -> dict:
        """
        Forward pass for surface field prediction.
        
        Args:
            batch: Dictionary containing:
                - geometry_position: (total_points, 3)
                - geometry_supernode_idx: (total_points,)
                - geometry_batch_idx: (total_points,)
                - surface_anchor_position: (B, N_surf, 3)
                - volume_anchor_position: (B, N_vol, 3) - dummy
                - surface_query_position: (B, M, 3) - optional
                
        Returns:
            Dictionary with predictions:
                - surface_anchor_pressure or surface_query_pressure: (B, N or M, 1)
                - surface_anchor_wallshearstress or surface_query_wallshearstress: (B, N or M, 3)
        """
        # Prepare inputs for AB-UPT
        geometry_position = batch['geometry_position']
        geometry_supernode_idx = batch['geometry_supernode_idx']
        geometry_batch_idx = batch.get('geometry_batch_idx', None)
        surface_anchor_position = batch['surface_anchor_position']
        volume_anchor_position = batch['volume_anchor_position']
        
        # Check if we have query positions (for evaluation)
        surface_query_position = batch.get('surface_query_position', None)
        volume_query_position = None  # We don't use volume queries
        
        # Forward through AB-UPT
        outputs = self.model(
            geometry_position=geometry_position,
            geometry_supernode_idx=geometry_supernode_idx,
            geometry_batch_idx=geometry_batch_idx,
            surface_anchor_position=surface_anchor_position,
            volume_anchor_position=volume_anchor_position,
            surface_query_position=surface_query_position,
            volume_query_position=volume_query_position,
        )
        
        # Parse outputs based on whether we have queries
        result = {}
        
        if surface_query_position is not None:
            # Predictions at query positions
            surface_predictions = outputs['surface_query']  # (B, M, output_dim)
            
            if self.predict_wss:
                # Split into pressure and WSS
                result['surface_query_pressure'] = surface_predictions[..., :1]  # (B, M, 1)
                result['surface_query_wallshearstress'] = surface_predictions[..., 1:4]  # (B, M, 3)
            else:
                result['surface_query_pressure'] = surface_predictions  # (B, M, 1)
        else:
            # Predictions at anchor positions
            surface_predictions = outputs['surface_anchor']  # (B, N, output_dim)
            
            if self.predict_wss:
                result['surface_anchor_pressure'] = surface_predictions[..., :1]
                result['surface_anchor_wallshearstress'] = surface_predictions[..., 1:4]
            else:
                result['surface_anchor_pressure'] = surface_predictions
        
        return result


class ABUPTSurfaceFieldPredictorLite(nn.Module):
    """
    Lightweight version of AB-UPT for surface field prediction.
    Faster training, suitable for quick experiments.
    """
    
    def __init__(self, predict_wss: bool = False, **kwargs):
        super().__init__()
        
        self.model = ABUPTSurfaceFieldPredictor(
            dim=128,
            geometry_depth=1,
            num_heads=4,
            blocks="ps",
            num_surface_blocks=2,
            num_volume_blocks=1,
            predict_wss=predict_wss,
            **kwargs,
        )
    
    def forward(self, batch: dict) -> dict:
        return self.model(batch)


def test_model():
    """Test the model with dummy data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Test with pressure only
    print("Testing model with pressure prediction only:")
    model = ABUPTSurfaceFieldPredictor(
        dim=128,
        geometry_depth=1,
        num_heads=4,
        blocks="ps",
        num_surface_blocks=2,
        num_volume_blocks=1,
        predict_wss=False,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy batch
    batch_size = 2
    num_geometry_points = 1024
    num_surface_anchors = 512
    
    batch = {
        'geometry_position': torch.randn(num_geometry_points * batch_size, 3).to(device),
        'geometry_supernode_idx': torch.randint(0, 128, (num_geometry_points * batch_size,)).to(device),
        'geometry_batch_idx': torch.cat([
            torch.zeros(num_geometry_points, dtype=torch.long),
            torch.ones(num_geometry_points, dtype=torch.long)
        ]).to(device),
        'surface_anchor_position': torch.randn(batch_size, num_surface_anchors, 3).to(device),
        'volume_anchor_position': torch.randn(batch_size, num_surface_anchors, 3).to(device),
    }
    
    # Forward pass
    with torch.no_grad():
        output = model(batch)
    
    print(f"\nOutput keys: {output.keys()}")
    print(f"Pressure prediction shape: {output['surface_anchor_pressure'].shape}")
    print(f"Pressure values (sample): {output['surface_anchor_pressure'][0, :5, 0]}")
    
    # Test with WSS prediction
    print("\n" + "="*60)
    print("Testing model with pressure + WSS prediction:")
    model_wss = ABUPTSurfaceFieldPredictor(
        dim=128,
        geometry_depth=1,
        num_heads=4,
        blocks="ps",
        num_surface_blocks=2,
        num_volume_blocks=1,
        predict_wss=True,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model_wss.parameters()):,}")
    
    with torch.no_grad():
        output_wss = model_wss(batch)
    
    print(f"\nOutput keys: {output_wss.keys()}")
    print(f"Pressure prediction shape: {output_wss['surface_anchor_pressure'].shape}")
    print(f"WSS prediction shape: {output_wss['surface_anchor_wallshearstress'].shape}")
    
    # Test with query positions
    print("\n" + "="*60)
    print("Testing model with query positions:")
    batch['surface_query_position'] = torch.randn(batch_size, 256, 3).to(device)
    
    with torch.no_grad():
        output_query = model_wss(batch)
    
    print(f"\nOutput keys: {output_query.keys()}")
    print(f"Query pressure shape: {output_query['surface_query_pressure'].shape}")
    print(f"Query WSS shape: {output_query['surface_query_wallshearstress'].shape}")


if __name__ == '__main__':
    test_model()
