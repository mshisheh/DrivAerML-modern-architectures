"""
Test Parameter Count for GraphCast Model

Verifies that the model has the expected number of parameters
for different configurations.

Author: Implementation for DrivAerNet benchmark
Date: February 2026
"""

import torch
from model import create_graphcast, count_parameters


def test_parameter_count():
    """Test parameter counts for different configurations"""
    
    print("=" * 80)
    print("GraphCast Parameter Count Verification")
    print("=" * 80)
    
    configs = [
        # (name, hidden_dim, num_mesh_nodes, num_processor_layers, expected_params_M)
        ("Tiny", 192, 400, 6, 0.8),
        ("Small", 256, 600, 8, 1.5),
        ("Medium", 384, 800, 12, 3.0),
        ("Large", 512, 1000, 16, 5.5),
    ]
    
    print("\nTesting different configurations:\n")
    
    for name, hidden_dim, num_mesh_nodes, num_processor_layers, expected_M in configs:
        print(f"{name:10s}  (D={hidden_dim}, M={num_mesh_nodes}, L={num_processor_layers})")
        
        # Create model
        model = create_graphcast(
            hidden_dim=hidden_dim,
            num_mesh_nodes=num_mesh_nodes,
            num_processor_layers=num_processor_layers,
            num_mlp_layers=1,
        )
        
        # Count parameters
        total_params = count_parameters(model)
        actual_M = total_params / 1e6
        
        # Check if within reasonable range (±20%)
        lower_bound = expected_M * 0.8
        upper_bound = expected_M * 1.2
        status = "✓" if lower_bound <= actual_M <= upper_bound else "✗"
        
        print(f"  {status} Parameters: {total_params:,} ({actual_M:.2f}M)")
        print(f"     Expected: ~{expected_M:.1f}M, Range: [{lower_bound:.1f}M - {upper_bound:.1f}M]")
        print()
    
    print("=" * 80)


def test_component_breakdown():
    """Test parameter breakdown by component"""
    
    print("\n" + "=" * 80)
    print("Component Parameter Breakdown (Medium Config)")
    print("=" * 80)
    
    # Create medium model
    model = create_graphcast(
        hidden_dim=384,
        num_mesh_nodes=800,
        num_processor_layers=12,
        num_mlp_layers=1,
    )
    
    # Count by component
    embedder_params = (
        count_parameters(model.grid_embedder) +
        count_parameters(model.mesh_embedder) +
        count_parameters(model.edge_embedder)
    )
    encoder_params = count_parameters(model.encoder)
    processor_params = count_parameters(model.processor)
    decoder_params = count_parameters(model.decoder)
    output_params = count_parameters(model.output_mlp)
    
    total = embedder_params + encoder_params + processor_params + decoder_params + output_params
    
    print(f"\nEmbedders:   {embedder_params:,} ({embedder_params/total*100:.1f}%)")
    print(f"Encoder:     {encoder_params:,} ({encoder_params/total*100:.1f}%)")
    print(f"Processor:   {processor_params:,} ({processor_params/total*100:.1f}%)")
    print(f"Decoder:     {decoder_params:,} ({decoder_params/total*100:.1f}%)")
    print(f"Output MLP:  {output_params:,} ({output_params/total*100:.1f}%)")
    print(f"\nTotal:       {total:,} ({total/1e6:.2f}M)")
    
    print("\n" + "=" * 80)


def test_forward_pass():
    """Test forward pass with dummy data"""
    
    print("\n" + "=" * 80)
    print("Forward Pass Test")
    print("=" * 80)
    
    # Create model
    model = create_graphcast(
        hidden_dim=256,  # Smaller for testing
        num_mesh_nodes=400,
        num_processor_layers=4,
        num_mlp_layers=1,
    )
    
    print(f"\nModel created with {count_parameters(model):,} parameters")
    
    # Create dummy data
    from torch_geometric.data import Data
    
    num_nodes = 1000  # Small for testing
    x = torch.randn(num_nodes, 7)
    pos = torch.randn(num_nodes, 3)
    data = Data(x=x, pos=pos)
    
    print(f"Input shape: {x.shape}")
    print(f"Position shape: {pos.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    try:
        with torch.no_grad():
            output = model(data)
        
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Output range: [{output.min():.4f}, {output.max():.4f}]")
        print("\n✓ Forward pass successful!")
    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        raise
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run tests
    test_parameter_count()
    test_component_breakdown()
    test_forward_pass()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
