#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fourier Neural Operator (FNO) implementation for DrivAerNet++ surface field prediction.

This module implements a 3D FNO that operates on voxel grids and predicts pressure
fields on surface points through interpolation and refinement.

Reference: Li, Z. et al. Fourier neural operator for parametric partial differential equations. 
           arXiv preprint arXiv:2010.08895 (2020).

@author: NeuralOperator Implementation for DrivAerNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv3d(nn.Module):
    """
    3D Spectral Convolution layer for Fourier Neural Operator.
    
    Performs convolution in Fourier space by:
    1. FFT of input
    2. Multiplication with learned weights (up to certain modes)
    3. Inverse FFT
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        modes1, modes2, modes3: Number of Fourier modes to keep in each dimension
    """
    
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply
        self.modes2 = modes2
        self.modes3 = modes3
        
        self.scale = 1 / (in_channels * out_channels)
        
        # Learnable weights for Fourier coefficients
        # For real FFT, we only need half the modes in the last dimension
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
    
    def compl_mul3d(self, input, weights):
        """Complex multiplication for 3D tensors."""
        # (batch, in_channel, x, y, z), (in_channel, out_channel, x, y, z) -> (batch, out_channel, x, y, z)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
    
    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, H, W, D)
        Returns:
            (batch, out_channels, H, W, D)
        """
        batchsize = x.shape[0]
        
        # Compute FFT
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Upper octant
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        
        # Lower octant (negative frequencies in first dimension)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        
        # Negative frequencies in second dimension
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        
        # Negative in both first and second dimensions
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        
        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)), dim=[-3, -2, -1])
        return x


class FNO3d(nn.Module):
    """
    3D Fourier Neural Operator.
    
    Architecture:
    - Lifting layer: projects input to hidden dimension
    - Fourier layers: spectral convolutions with skip connections
    - Projection layer: projects to output dimension
    
    Args:
        modes1, modes2, modes3: Number of Fourier modes
        width: Hidden dimension
        in_channels: Input channels (4: occupancy + 3D coords)
        out_channels: Output channels (1 for pressure)
        n_layers: Number of Fourier layers
    """
    
    def __init__(self, modes1=8, modes2=8, modes3=8, width=32, in_channels=4, out_channels=1, n_layers=4):
        super().__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.n_layers = n_layers
        
        # Lifting layer
        self.fc0 = nn.Linear(in_channels, self.width)
        
        # Fourier layers
        self.conv_layers = nn.ModuleList([
            SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
            for _ in range(n_layers)
        ])
        
        # Skip connections (1x1x1 convolutions)
        self.w_layers = nn.ModuleList([
            nn.Conv3d(self.width, self.width, 1)
            for _ in range(n_layers)
        ])
        
        # Projection layers (to output channels)
        self.fc1 = nn.Conv3d(self.width, self.width // 2, 1)
        self.fc2 = nn.Conv3d(self.width // 2, out_channels, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, H, W, D)
        Returns:
            (batch, out_channels, H, W, D)
        """
        # Permute to (batch, H, W, D, channels) for linear layer
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)  # Back to (batch, channels, H, W, D)
        
        # Fourier layers
        for i in range(self.n_layers):
            x1 = self.conv_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)
        
        # Projection (using conv for efficiency)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x


class FNOSurfaceFieldPredictor(nn.Module):
    """
    Complete FNO model for surface pressure prediction.
    
    Pipeline:
    1. FNO processes voxel grid -> volume field
    2. Trilinear interpolation to query points
    3. Point-wise refinement network
    
    Args:
        grid_resolution: Resolution of input voxel grid
        fno_modes: Number of Fourier modes (default 8)
        fno_width: Hidden dimension of FNO
        fno_layers: Number of FNO layers
        refine_hidden: Hidden size for refinement MLP
    """
    
    def __init__(
        self,
        grid_resolution=32,
        fno_modes=8,
        fno_width=16,  # Reduced to hit ~2.1M params
        fno_layers=4,
        refine_hidden=64,
    ):
        super().__init__()
        
        self.grid_resolution = grid_resolution
        
        # FNO backbone
        self.fno = FNO3d(
            modes1=fno_modes,
            modes2=fno_modes,
            modes3=fno_modes,
            width=fno_width,
            in_channels=4,  # occupancy + xyz coords
            out_channels=16,  # Intermediate features
            n_layers=fno_layers,
        )
        
        # Point-wise refinement network
        # Input: interpolated features (16) + normalized position (3)
        self.refinement = nn.Sequential(
            nn.Linear(16 + 3, refine_hidden),
            nn.GELU(),
            nn.Linear(refine_hidden, refine_hidden),
            nn.GELU(),
            nn.Linear(refine_hidden, 1),
        )
    
    def interpolate_to_points(self, volume_features, positions, bbox):
        """
        Interpolate volume features to query points using trilinear interpolation.
        
        Args:
            volume_features: (batch, channels, H, W, D)
            positions: List of (N_i, 3) point positions for each batch
            bbox: List of (min_coords, max_coords) tuples
            
        Returns:
            List of (N_i, channels) interpolated features
        """
        batch_size = volume_features.shape[0]
        interpolated_features = []
        
        for i in range(batch_size):
            # Normalize positions to [-1, 1] for grid_sample
            min_coords, max_coords = bbox[i]
            min_coords = torch.tensor(min_coords, device=positions[i].device, dtype=torch.float32)
            max_coords = torch.tensor(max_coords, device=positions[i].device, dtype=torch.float32)
            
            normalized_pos = (positions[i] - min_coords) / (max_coords - min_coords)
            normalized_pos = normalized_pos * 2 - 1  # Scale to [-1, 1]
            
            # Add batch and dummy dimensions for grid_sample
            # grid_sample expects (N, C, D_out, H_out, W_out) and (N, D_out, H_out, W_out, 3)
            query_points = normalized_pos.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, N_points, 3)
            
            # Note: grid_sample expects (x, y, z) order but we have (x, y, z) already
            # Reorder to match grid_sample convention (z, y, x) by flipping
            query_points = query_points[..., [2, 1, 0]]  # (1, 1, 1, N_points, 3) in (z, y, x) order
            
            volume = volume_features[i:i+1]  # (1, C, H, W, D)
            
            # Interpolate
            features = F.grid_sample(
                volume,
                query_points,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )  # (1, C, 1, 1, N_points)
            
            features = features.squeeze(0).squeeze(1).squeeze(1).T  # (N_points, C)
            interpolated_features.append(features)
        
        return interpolated_features
    
    def forward(self, voxel_grids, positions, bboxes):
        """
        Forward pass.
        
        Args:
            voxel_grids: (batch, 4, H, W, D)
            positions: List of (N_i, 3) point positions
            bboxes: List of bounding boxes
            
        Returns:
            List of (N_i, 1) pressure predictions
        """
        # Process through FNO
        volume_features = self.fno(voxel_grids)  # (batch, 16, H, W, D)
        
        # Interpolate to surface points
        interpolated_features = self.interpolate_to_points(volume_features, positions, bboxes)
        
        # Normalize positions for refinement
        predictions = []
        for i, (features, pos) in enumerate(zip(interpolated_features, positions)):
            # Normalize positions
            min_coords, max_coords = bboxes[i]
            min_coords = torch.tensor(min_coords, device=pos.device, dtype=torch.float32)
            max_coords = torch.tensor(max_coords, device=pos.device, dtype=torch.float32)
            normalized_pos = (pos - min_coords) / (max_coords - min_coords)
            
            # Concatenate features and positions
            combined = torch.cat([features, normalized_pos], dim=-1)
            
            # Refine
            pred = self.refinement(combined)
            predictions.append(pred)
        
        return predictions
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FNOSurfaceFieldPredictor(
        grid_resolution=32,
        fno_modes=8,
        fno_width=16,
        fno_layers=4,
        refine_hidden=64,
    ).to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Target: 2.10M parameters")
    
    # Test forward pass
    batch_size = 2
    voxel_grids = torch.randn(batch_size, 4, 32, 32, 32).to(device)
    positions = [torch.randn(1000, 3).to(device) for _ in range(batch_size)]
    bboxes = [(np.array([0, 0, 0]), np.array([1, 1, 1])) for _ in range(batch_size)]
    
    predictions = model(voxel_grids, positions, bboxes)
    print(f"\nTest successful!")
    print(f"Input voxel grid: {voxel_grids.shape}")
    print(f"Output predictions: {[p.shape for p in predictions]}")
