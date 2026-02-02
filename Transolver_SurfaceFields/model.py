#!/usr/bin/env python3
"""
Transolver (original) implementation for DrivAerNet surface pressure prediction.

This is a lightweight, standalone re-implementation of the original Transolver
idea: a transformer-based neural solver for irregular geometric meshes.

This variant avoids PhysicsNemo dependencies and uses PyTorch only.

Output: per-point scalar pressure (Cp)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for 3D coordinates."""

    def __init__(self, d_model: int, max_freq=10.0, n_freqs=6):
        super().__init__()
        self.d_model = d_model
        freqs = torch.logspace(0.0, math.log10(max_freq), n_freqs)
        self.register_buffer("freqs", freqs)

    def forward(self, coords):
        # coords: (N, 3)
        # produce (N, d_model) encoding
        N, C = coords.shape
        freqs = self.freqs[None, :, None]  # (1, n_freqs, 1)
        coords_exp = coords.unsqueeze(1) * freqs  # (N, n_freqs, 3)
        enc = torch.cat([torch.sin(coords_exp), torch.cos(coords_exp)], dim=-1)
        enc = enc.view(N, -1)
        if enc.shape[1] >= self.d_model:
            return enc[:, : self.d_model]
        else:
            pad = enc.new_zeros(N, self.d_model - enc.shape[1])
            return torch.cat([enc, pad], dim=1)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransolverBlock(nn.Module):
    """Standard transformer block (pre-norm) applied to point features."""

    def __init__(self, dim, n_heads=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x, attn_mask=None):
        # x: (B, N, C)
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class Transolver(nn.Module):
    """Simple Transolver variant for point-wise pressure prediction.

    - embeds geometric + auxiliary features
    - adds positional encoding from coordinates
    - passes features through a stack of transformer blocks
    - outputs scalar pressure per point
    """

    def __init__(
        self,
        fun_dim=6,
        coord_dim=3,
        d_model=256,
        n_layers=6,
        n_heads=8,
        mlp_ratio=2.5,
        dropout=0.0,
        out_dim=1,
    ):
        super().__init__()
        self.d_model = d_model

        # Embedding for input features (e.g., [x,y,z,nx,ny,nz,area] minus coords if provided separately)
        self.input_mlp = nn.Sequential(
            nn.Linear(fun_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Positional encoding to mix spatial coords
        self.pos_enc = PositionalEncoding(d_model=d_model // 2, max_freq=20.0, n_freqs=8)
        self.coord_proj = nn.Linear((d_model // 2), d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransolverBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)
        ])

        # Final head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, features, coords=None):
        """Forward.

        Args:
            features: list of (N_i, fun_dim) tensors OR a single tensor (N, fun_dim)
            coords: optional list of (N_i, 3) tensors or a single (N, 3)

        Returns:
            list of (N_i, out_dim) predictions or single tensor
        """
        single_input = False
        if isinstance(features, torch.Tensor):
            features = [features]
            if coords is not None and isinstance(coords, torch.Tensor):
                coords = [coords]
            single_input = True

        outputs = []
        for i, feat in enumerate(features):
            # feat: (N, fun_dim)
            x = self.input_mlp(feat).unsqueeze(0)  # (1, N, d_model)

            if coords is not None:
                c = coords[i]
                pos = self.pos_enc(c)  # (N, d_model//2)
                pos = self.coord_proj(pos).unsqueeze(0)  # (1, N, d_model)
                x = x + pos

            # Transformer stack
            for block in self.blocks:
                x = block(x)

            out = self.head(x.squeeze(0))
            outputs.append(out)

        if single_input:
            return outputs[0]
        return outputs

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_transolver(d_model=256, n_layers=6):
    model = Transolver(d_model=d_model, n_layers=n_layers)
    print(f"Created Transolver with {model.count_parameters():,} parameters")
    return model


if __name__ == '__main__':
    # Smoke test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_transolver(d_model=256, n_layers=6).to(device)

    # Create dummy data: batch of 2 models
    batch_size = 2
    n_points = 1500
    features = [torch.randn(n_points, 6).to(device) for _ in range(batch_size)]
    coords = [torch.randn(n_points, 3).to(device) for _ in range(batch_size)]

    preds = model(features, coords=coords)
    print(f"Test forward OK. Output shapes: {[p.shape for p in preds]}")
