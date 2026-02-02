#!/usr/bin/env python3
"""
Simple test of Transolver model without data loader dependencies.
"""
import torch
from model import create_transolver


def test_model():
    """Test model forward pass with various configurations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on: {device}\n")
    
    # Test different configurations
    configs = [
        ("Small", 192, 4),
        ("Base", 208, 6),
        ("Medium", 256, 6),
    ]
    
    for name, d_model, n_layers in configs:
        print(f"Testing Transolver-{name} (d_model={d_model}, n_layers={n_layers})...")
        model = create_transolver(d_model=d_model, n_layers=n_layers).to(device)
        
        # Test with single input
        n_points = 2000
        features = torch.randn(n_points, 6).to(device)
        coords = torch.randn(n_points, 3).to(device)
        
        pred = model(features, coords=coords)
        assert pred.shape == (n_points, 1), f"Expected {(n_points, 1)}, got {pred.shape}"
        
        # Test with batch (list) input
        batch_size = 3
        features_list = [torch.randn(n_points, 6).to(device) for _ in range(batch_size)]
        coords_list = [torch.randn(n_points, 3).to(device) for _ in range(batch_size)]
        
        preds = model(features_list, coords=coords_list)
        assert len(preds) == batch_size
        assert all(p.shape == (n_points, 1) for p in preds)
        
        print(f"  ✓ Forward pass OK\n")
    
    print("All tests passed!")


if __name__ == '__main__':
    test_model()
