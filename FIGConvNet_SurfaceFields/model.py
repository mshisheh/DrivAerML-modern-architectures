"""
FIGConvNet Implementation for DrivAerNet Surface Pressure Prediction

Simplified self-contained implementation of Feature-Interacting Graph Convolution Network.
Based on NVIDIA PhysicsNemo's FIGConvUNet architecture.

Key Concept: Hybrid point-to-grid-to-point architecture
    1. Point → Grid: Project points onto multiple factorized grids
    2. Grid Processing: U-Net style convolutions on 2D grids
    3. Grid → Point: Sample grid features back to points

Author: Implementation for DrivAerNet benchmark
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from torch_geometric.data import Data
import math


class SinusoidalPositionEncoding(nn.Module):
    """Sinusoidal position encoding for coordinates"""
    
    def __init__(self, embed_dim: int = 32, scale: float = 2.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = scale
        
        # Frequency bands
        freq_bands = torch.linspace(0, embed_dim-1, embed_dim) / embed_dim
        self.register_buffer('freq_bands', freq_bands)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [B, N, 3] or [N, 3]
        Returns:
            encoded: [..., 3 * embed_dim]
        """
        original_shape = coords.shape
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)
        
        # Scale coordinates
        coords_scaled = coords / self.scale
        
        # Apply sinusoidal encoding to each dimension
        encoded_list = []
        for dim in range(3):
            coord_dim = coords_scaled[..., dim:dim+1]  # [B, N, 1]
            freqs = coord_dim * (2 ** self.freq_bands) * math.pi  # [B, N, embed_dim]
            encoded_list.append(torch.sin(freqs))
            encoded_list.append(torch.cos(freqs))
        
        encoded = torch.cat(encoded_list, dim=-1)  # [B, N, 6*embed_dim]
        
        if len(original_shape) == 2:
            encoded = encoded.squeeze(0)
        
        return encoded


class MLP(nn.Module):
    """Multi-layer perceptron with optional residual connections"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        use_residual: bool = False,
        activation: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self.use_residual = use_residual and (input_dim == output_dim)
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(activation)
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.use_residual:
            out = out + x
        return out


class PointToGrid(nn.Module):
    """
    Project point features onto a 2D grid using scatter operation.
    
    Key idea: For 3D points, project onto 2D planes (XY, XZ, YZ) and
    scatter features using their grid coordinates.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_resolution: Tuple[int, int],
        spatial_bounds: Tuple[float, float] = (-1.0, 1.0),
        use_position_encoding: bool = True,
        pos_encode_dim: int = 32,
    ):
        super().__init__()
        self.grid_resolution = grid_resolution
        self.spatial_bounds = spatial_bounds
        self.use_position_encoding = use_position_encoding
        
        if use_position_encoding:
            self.pos_encoder = SinusoidalPositionEncoding(
                embed_dim=pos_encode_dim,
                scale=spatial_bounds[1] - spatial_bounds[0]
            )
            total_input_dim = input_dim + 6 * pos_encode_dim  # 6 = 3 coords * 2 (sin+cos)
        else:
            total_input_dim = input_dim + 3  # Just add raw coordinates
        
        self.feature_mlp = MLP(
            input_dim=total_input_dim,
            output_dim=output_dim,
            hidden_dims=[output_dim],
            activation=nn.GELU(),
        )
    
    def _coords_to_grid_indices(
        self, coords: torch.Tensor, axis1: int, axis2: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert 3D coordinates to 2D grid indices for specified axes"""
        # Normalize to [0, 1]
        coords_norm = (coords - self.spatial_bounds[0]) / (
            self.spatial_bounds[1] - self.spatial_bounds[0]
        )
        coords_norm = torch.clamp(coords_norm, 0, 1)
        
        # Convert to grid indices
        grid_coords = coords_norm * torch.tensor(
            self.grid_resolution, device=coords.device
        ).unsqueeze(0)
        grid_coords = torch.clamp(grid_coords, 0, torch.tensor(self.grid_resolution) - 1)
        
        idx1 = grid_coords[:, axis1].long()
        idx2 = grid_coords[:, axis2].long()
        
        return idx1, idx2
    
    def forward(
        self, pos: torch.Tensor, features: torch.Tensor, plane_axes: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Args:
            pos: [N, 3] - point positions
            features: [N, C] - point features
            plane_axes: (axis1, axis2) - which axes to project onto
        
        Returns:
            grid: [1, output_dim, H, W] - grid features
        """
        # Encode positions
        if self.use_position_encoding:
            pos_encoded = self.pos_encoder(pos)
        else:
            pos_encoded = pos
        
        # Combine features and positions
        combined_features = torch.cat([features, pos_encoded], dim=-1)
        
        # Transform features
        transformed_features = self.feature_mlp(combined_features)  # [N, output_dim]
        
        # Get grid indices
        idx1, idx2 = self._coords_to_grid_indices(pos, plane_axes[0], plane_axes[1])
        
        # Create grid and scatter features
        H, W = self.grid_resolution
        C = transformed_features.size(1)
        grid = torch.zeros(C, H, W, device=pos.device, dtype=torch.float32)
        counts = torch.zeros(H, W, device=pos.device, dtype=torch.float32)
        
        # Scatter add features to grid
        for n in range(len(pos)):
            grid[:, idx1[n], idx2[n]] += transformed_features[n]
            counts[idx1[n], idx2[n]] += 1
        
        # Average overlapping points
        counts = counts.clamp(min=1.0)
        grid = grid / counts.unsqueeze(0)
        
        return grid.unsqueeze(0)  # [1, C, H, W]


class GridConvBlock(nn.Module):
    """2D Convolutional block for grid processing"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        up_stride: int = 1,
    ):
        super().__init__()
        
        self.up_stride = up_stride
        
        if up_stride > 1:
            # Upsampling path
            self.upsample = nn.Upsample(scale_factor=up_stride, mode='bilinear', align_corners=False)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        else:
            # Downsampling or same resolution path
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.up_stride > 1:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class GridUNet(nn.Module):
    """U-Net for processing grid features"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        kernel_size: int = 3,
        num_levels: int = 3,
    ):
        super().__init__()
        self.num_levels = num_levels
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        for i in range(num_levels):
            in_ch = in_channels if i == 0 else hidden_channels[i-1]
            out_ch = hidden_channels[i]
            self.down_blocks.append(
                GridConvBlock(in_ch, out_ch, kernel_size, stride=2)
            )
        
        # Bottleneck
        self.bottleneck = GridConvBlock(
            hidden_channels[-1], hidden_channels[-1], kernel_size
        )
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(num_levels)):
            in_ch = hidden_channels[i]
            out_ch = hidden_channels[i-1] if i > 0 else in_channels
            self.up_blocks.append(
                GridConvBlock(in_ch, out_ch, kernel_size, up_stride=2)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsampling with skip connections
        skip_connections = []
        for down in self.down_blocks:
            skip_connections.append(x)
            x = down(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Upsampling with skip connections
        for up, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = up(x)
            # Crop or pad to match skip connection size
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip
        
        return x


class GridToPoint(nn.Module):
    """Sample grid features back to point locations"""
    
    def __init__(self, grid_channels: int, point_channels: int, output_dim: int):
        super().__init__()
        
        self.feature_mlp = MLP(
            input_dim=grid_channels + point_channels,
            output_dim=output_dim,
            hidden_dims=[output_dim * 2],
            activation=nn.GELU(),
        )
    
    def _sample_grid(
        self, grid: torch.Tensor, pos: torch.Tensor, plane_axes: Tuple[int, int],
        spatial_bounds: Tuple[float, float]
    ) -> torch.Tensor:
        """Sample grid features at point locations using bilinear interpolation"""
        # Normalize coordinates to [-1, 1] for grid_sample
        pos_norm = (pos - spatial_bounds[0]) / (spatial_bounds[1] - spatial_bounds[0])
        pos_norm = pos_norm * 2 - 1  # [0, 1] -> [-1, 1]
        
        # Select appropriate axes
        coords_2d = pos_norm[:, [plane_axes[0], plane_axes[1]]]  # [N, 2]
        
        # Reshape for grid_sample: [1, N, 1, 2]
        coords_grid = coords_2d.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]
        
        # Sample from grid [1, C, H, W]
        sampled = F.grid_sample(
            grid, coords_grid, mode='bilinear', padding_mode='border', align_corners=False
        )
        
        # Reshape: [1, C, N, 1] -> [N, C]
        sampled = sampled.squeeze(0).squeeze(2).transpose(0, 1)
        
        return sampled
    
    def forward(
        self, grids: List[torch.Tensor], grid_axes: List[Tuple[int, int]],
        pos: torch.Tensor, point_features: torch.Tensor,
        spatial_bounds: Tuple[float, float]
    ) -> torch.Tensor:
        """
        Args:
            grids: List of [1, C, H, W] grids
            grid_axes: List of (axis1, axis2) for each grid
            pos: [N, 3] point positions
            point_features: [N, C'] point features
            spatial_bounds: (min, max) spatial bounds
        
        Returns:
            output: [N, output_dim] output features
        """
        # Sample from all grids
        sampled_features = []
        for grid, axes in zip(grids, grid_axes):
            sampled = self._sample_grid(grid, pos, axes, spatial_bounds)
            sampled_features.append(sampled)
        
        # Aggregate sampled features
        aggregated = torch.cat(sampled_features, dim=-1)  # [N, C*num_grids]
        aggregated = aggregated.mean(dim=-1, keepdim=True).expand(-1, sampled_features[0].size(1))
        
        # Combine with point features
        combined = torch.cat([aggregated, point_features], dim=-1)
        
        # Transform
        output = self.feature_mlp(combined)
        
        return output


class FIGConvNet(nn.Module):
    """
    Feature-Interacting Graph Convolution Network (FIGConvNet)
    
    Hybrid architecture:
        Points → Factorized 2D Grids → U-Net Processing → Points
    
    Parameters:
        input_dim: Input feature dimension
        output_dim: Output dimension (1 for pressure)
        hidden_dim: Hidden dimension for grid features
        grid_resolution: Resolution of 2D grids (H, W)
        hidden_channels: List of channel dims for U-Net levels
        num_levels: Number of U-Net levels
        spatial_bounds: (min, max) for coordinate normalization
        pos_encode_dim: Dimension for position encoding
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        output_dim: int = 1,
        hidden_dim: int = 64,
        grid_resolution: Tuple[int, int] = (64, 64),
        hidden_channels: List[int] = [64, 128, 256],
        num_levels: int = 3,
        spatial_bounds: Tuple[float, float] = (-1.5, 1.5),
        pos_encode_dim: int = 16,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.grid_resolution = grid_resolution
        self.spatial_bounds = spatial_bounds
        
        # Feature projection
        self.input_projection = MLP(
            input_dim=input_dim,
            output_dim=hidden_dim,
            hidden_dims=[hidden_dim],
            activation=nn.GELU(),
        )
        
        # Point-to-Grid projections (3 factorized grids: XY, XZ, YZ)
        self.point_to_grids = nn.ModuleList([
            PointToGrid(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                grid_resolution=grid_resolution,
                spatial_bounds=spatial_bounds,
                use_position_encoding=True,
                pos_encode_dim=pos_encode_dim,
            )
            for _ in range(3)
        ])
        
        # Grid axes for factorization
        self.grid_axes = [(0, 1), (0, 2), (1, 2)]  # XY, XZ, YZ
        
        # U-Net for each grid
        self.grid_unets = nn.ModuleList([
            GridUNet(
                in_channels=hidden_dim,
                hidden_channels=hidden_channels,
                kernel_size=3,
                num_levels=num_levels,
            )
            for _ in range(3)
        ])
        
        # Grid-to-Point
        self.grid_to_point = GridToPoint(
            grid_channels=hidden_dim,
            point_channels=hidden_dim,
            output_dim=hidden_dim * 2,
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: PyG Data object with:
                - data.x: [N, input_dim] features
                - data.pos: [N, 3] positions
        
        Returns:
            predictions: [N, output_dim]
        """
        pos = data.pos  # [N, 3]
        features = data.x  # [N, input_dim]
        
        # Project input features
        features_proj = self.input_projection(features)  # [N, hidden_dim]
        
        # Point-to-Grid: Project to 3 factorized grids
        grids = []
        for point_to_grid, axes in zip(self.point_to_grids, self.grid_axes):
            grid = point_to_grid(pos, features_proj, axes)  # [1, hidden_dim, H, W]
            grids.append(grid)
        
        # Process each grid with U-Net
        processed_grids = []
        for grid, unet in zip(grids, self.grid_unets):
            processed = unet(grid)  # [1, hidden_dim, H, W]
            processed_grids.append(processed)
        
        # Grid-to-Point: Sample back to points
        point_features_out = self.grid_to_point(
            processed_grids, self.grid_axes, pos, features_proj, self.spatial_bounds
        )  # [N, hidden_dim * 2]
        
        # Output projection
        output = self.output_projection(point_features_out)  # [N, output_dim]
        
        return output


def create_figconvnet(
    hidden_dim: int = 64,
    grid_resolution: Tuple[int, int] = (64, 64),
    hidden_channels: List[int] = [64, 128, 256],
    num_levels: int = 3,
    input_dim: int = 7,
    output_dim: int = 1,
) -> FIGConvNet:
    """
    Factory function to create FIGConvNet model.
    
    Parameter configurations for target sizes:
    - ~1.5M params: hidden_dim=48, grid_resolution=(48,48), hidden_channels=[48,96,192]
    - ~3M params: hidden_dim=64, grid_resolution=(64,64), hidden_channels=[64,128,256]
    - ~5M params: hidden_dim=80, grid_resolution=(80,80), hidden_channels=[80,160,320]
    """
    model = FIGConvNet(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        grid_resolution=grid_resolution,
        hidden_channels=hidden_channels,
        num_levels=num_levels,
        spatial_bounds=(-1.5, 1.5),
        pos_encode_dim=16,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing FIGConvNet Model...\n")
    
    # Create model
    model = create_figconvnet(
        hidden_dim=64,
        grid_resolution=(64, 64),
        hidden_channels=[64, 128, 256],
        num_levels=3,
    )
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Test forward pass
    from torch_geometric.data import Data
    
    num_points = 10000
    x = torch.randn(num_points, 7)
    pos = torch.randn(num_points, 3) * 0.5  # Keep within [-1.5, 1.5]
    data = Data(x=x, pos=pos)
    
    print(f"\nInput features: {x.shape}")
    print(f"Input positions: {pos.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(data)
    
    print(f"Output shape: {output.shape}")
    print("\n✓ FIGConvNet model test passed!")
