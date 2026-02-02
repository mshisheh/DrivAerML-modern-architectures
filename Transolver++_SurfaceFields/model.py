#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transolver++ implementation for DrivAerNet++ surface field prediction.

This module implements the physics-aware transformer architecture with slicing attention
following the exact architecture from the paper.

Reference: Luo, H. et al. Transolver++: An accurate neural solver for pdes on 
           million-scale geometries. arXiv preprint arXiv:2502.02414 (2025).

@author: Transolver++ Implementation for DrivAerNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_softmax(logits, tau=1.0, hard=False):
    """
    Gumbel-Softmax trick for differentiable sampling.
    
    Args:
        logits: (*, n_slices) unnormalized log probabilities
        tau: Temperature for softmax
        hard: If True, return one-hot (straight-through estimator)
        
    Returns:
        Soft assignment weights
    """
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    
    y = logits + gumbel_noise
    y = y / tau
    y = F.softmax(y, dim=-1)
    
    if hard:
        _, y_hard = y.max(dim=-1)
        y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard.unsqueeze(-1), 1.0)
        y = (y_one_hot - y).detach() + y
    
    return y


class PhysicsAwareSlicingAttention(nn.Module):
    """
    Physics-aware attention with slicing mechanism for irregular geometric meshes.
    
    Process:
    1. Distribute points across n_slices using Gumbel-Softmax
    2. Aggregate information within each slice (slice-level tokens)
    3. Self-attention across slice tokens
    4. Redistribute back to individual points
    
    Args:
        dim: Model dimension
        heads: Number of attention heads (default: 8)
        dim_head: Dimension per head
        dropout: Dropout rate
        slice_num: Number of slices (default: 32)
    """
    
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=32):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.slice_num = slice_num
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        # Temperature bias for Gumbel-Softmax
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        
        # Temperature projection network
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU()
        )
        
        # Project input to inner dimension
        self.in_project_x = nn.Linear(dim, inner_dim)
        
        # Slice assignment network
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        
        # Q, K, V projections
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, N, C) - batch of point features
            
        Returns:
            (B, N, C) - processed features
        """
        B, N, C = x.shape
        
        # Project and reshape to multi-head format
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head)
        x_mid = x_mid.permute(0, 2, 1, 3).contiguous()  # (B, H, N, C)
        
        # Compute adaptive temperature
        temperature = self.proj_temperature(x_mid) + self.bias  # (B, H, N, 1)
        temperature = torch.clamp(temperature, min=0.01)
        
        # Soft assignment to slices via Gumbel-Softmax
        slice_logits = self.in_project_slice(x_mid)  # (B, H, N, slice_num)
        slice_weights = gumbel_softmax(
            slice_logits, 
            tau=temperature
        )  # (B, H, N, slice_num)
        
        # Normalize by number of points in each slice
        slice_norm = slice_weights.sum(2)  # (B, H, slice_num)
        
        # Aggregate to slice-level tokens
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights).contiguous()
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
        
        # Apply Q, K, V projections
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        
        # Scaled dot-product attention
        out_slice_token = F.scaled_dot_product_attention(
            q_slice_token, k_slice_token, v_slice_token
        )
        
        # Redistribute back to points (inverse slicing)
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        
        # Reshape to original format
        out_x = out_x.permute(0, 2, 1, 3).contiguous()
        out_x = out_x.reshape(B, N, -1)
        
        return self.to_out(out_x)


class MLP(nn.Module):
    """
    Multi-layer perceptron with residual connections.
    
    Args:
        n_input: Input dimension
        n_hidden: Hidden dimension
        n_output: Output dimension
        n_layers: Number of hidden layers
        act: Activation function
        res: Use residual connections
    """
    
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        
        self.linear_pre = nn.Sequential(
            nn.Linear(n_input, n_hidden), 
            nn.GELU()
        )
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([
            nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.GELU()) 
            for _ in range(n_layers)
        ])
    
    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class TransolverPPBlock(nn.Module):
    """
    Single Transolver++ transformer block.
    
    Components:
    - Layer Normalization + Physics-aware slicing attention + residual
    - Layer Normalization + Feed-forward network + residual
    - Optional output projection for last layer
    
    Args:
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension
        dropout: Dropout rate
        act: Activation function
        mlp_ratio: FFN expansion ratio
        last_layer: Whether this is the last layer
        out_dim: Output dimension (for last layer)
        slice_num: Number of slices
    """
    
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act='gelu',
        mlp_ratio=1,
        last_layer=False,
        out_dim=1,
        slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        
        # Pre-norm for attention
        self.ln_1 = nn.LayerNorm(hidden_dim)
        
        # Physics-aware slicing attention
        self.Attn = PhysicsAwareSlicingAttention(
            hidden_dim, 
            heads=num_heads, 
            dim_head=hidden_dim // num_heads,
            dropout=dropout, 
            slice_num=slice_num
        )
        
        # Pre-norm for FFN
        self.ln_2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.mlp = MLP(
            hidden_dim, 
            hidden_dim * mlp_ratio, 
            hidden_dim, 
            n_layers=0, 
            res=False, 
            act=act
        )
        
        # Output head for last layer
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, fx):
        """
        Args:
            fx: (B, N, hidden_dim)
            
        Returns:
            (B, N, hidden_dim) or (B, N, out_dim) if last layer
        """
        # Attention block with residual
        fx = fx + self.Attn(self.ln_1(fx))
        
        # FFN block with residual
        fx = fx + self.mlp(self.ln_2(fx))
        
        # Output projection for last layer
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class TransolverPP(nn.Module):
    """
    Complete Transolver++ model for surface pressure prediction.
    
    Architecture:
    1. Preprocess: MLP embedding (6D → n_hidden)
    2. Add trainable global placeholder vector
    3. 5 sequential transformer blocks with physics-aware attention
    4. Final layer outputs scalar predictions
    
    Args:
        space_dim: Spatial dimension (3 for 3D)
        n_layers: Number of transformer layers (default: 5)
        n_hidden: Hidden dimension
        dropout: Dropout rate (default: 0.0)
        n_head: Number of attention heads (default: 8)
        act: Activation function
        mlp_ratio: FFN expansion ratio (default: 1)
        fun_dim: Function dimension (6 for [x,y,z,nx,ny,nz])
        out_dim: Output dimension (1 for pressure)
        slice_num: Number of slices (default: 32)
    """
    
    def __init__(
        self,
        space_dim=3,
        n_layers=5,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        act='gelu',
        mlp_ratio=1,
        fun_dim=6,
        out_dim=1,
        slice_num=32,
    ):
        super().__init__()
        
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        
        # Preprocess: MLP to embed input features
        self.preprocess = MLP(
            fun_dim, 
            n_hidden * 2, 
            n_hidden, 
            n_layers=0,
            res=False, 
            act=act
        )
        
        # Trainable global placeholder vector
        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransolverPPBlock(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                out_dim=out_dim,
                slice_num=slice_num,
                last_layer=(_ == n_layers - 1)
            )
            for _ in range(n_layers)
        ])
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights using truncated normal."""
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, features):
        """
        Forward pass.
        
        Args:
            features: List of (N_i, 6) feature tensors
            
        Returns:
            List of (N_i, 1) pressure predictions
        """
        predictions = []
        
        for feat in features:
            # Preprocess input features
            fx = self.preprocess(feat.unsqueeze(0))  # (1, N, n_hidden)
            
            # Add global placeholder
            fx = fx + self.placeholder[None, None, :]
            
            # Process through transformer blocks
            for block in self.blocks:
                fx = block(fx)
            
            # Remove batch dimension
            pred = fx.squeeze(0)  # (N, out_dim)
            predictions.append(pred)
        
        return predictions
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_transolver_pp(n_hidden=172, target_params=1.81e6):
    """
    Create Transolver++ model calibrated to target parameter count.
    
    Args:
        n_hidden: Hidden dimension (tuned to hit ~1.81M params)
        target_params: Target parameter count
        
    Returns:
        TransolverPP model
    """
    model = TransolverPP(
        space_dim=3,
        n_layers=5,
        n_hidden=n_hidden,
        dropout=0.0,
        n_head=8,
        act='gelu',
        mlp_ratio=1,
        fun_dim=6,
        out_dim=1,
        slice_num=32,
    )
    
    actual_params = model.count_parameters()
    print(f"Created Transolver++ with {actual_params:,} parameters (target: {target_params:,.0f})")
    
    return model


if __name__ == '__main__':
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Calibrated n_hidden=284 for exactly 1.81M params
    model = create_transolver_pp(n_hidden=284).to(device)
    
    print(f"\nModel parameters: {model.count_parameters():,}")
    print(f"Target: 1,810,000 (1.81M)")
    
    # Test forward pass
    batch_size = 2
    n_points = 1000
    features = [torch.randn(n_points, 6).to(device) for _ in range(batch_size)]
    
    predictions = model(features)
    print(f"\nTest successful!")
    print(f"Input features: {[f.shape for f in features]}")
    print(f"Output predictions: {[p.shape for p in predictions]}")

