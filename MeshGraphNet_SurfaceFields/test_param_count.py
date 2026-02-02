"""
Simple test script for MeshGraphNet parameter counting (without torch_geometric dependency).
"""

import torch
import torch.nn as nn


def count_mlp_params(input_dim, output_dim, hidden_dim, num_layers, use_layer_norm=True):
    """Calculate parameters for an MLP."""
    if num_layers == 0:
        return 0
    
    params = 0
    
    # First layer
    params += input_dim * hidden_dim + hidden_dim  # Linear
    if use_layer_norm:
        params += 2 * hidden_dim  # LayerNorm (scale + shift)
    
    # Hidden layers
    for _ in range(num_layers - 1):
        params += hidden_dim * hidden_dim + hidden_dim  # Linear
        if use_layer_norm:
            params += 2 * hidden_dim  # LayerNorm
    
    # Output layer
    params += hidden_dim * output_dim + output_dim  # Linear
    
    return params


def count_meshgraphnet_params(
    input_dim_nodes=7,
    input_dim_edges=4,
    output_dim=1,
    processor_size=15,
    hidden_dim=128,
    num_layers_node=2,
    num_layers_edge=2
):
    """Calculate total parameters for MeshGraphNet."""
    
    # Encoder
    node_encoder_params = count_mlp_params(input_dim_nodes, hidden_dim, hidden_dim, 2, True)
    edge_encoder_params = count_mlp_params(input_dim_edges, hidden_dim, hidden_dim, 2, True)
    encoder_params = node_encoder_params + edge_encoder_params
    
    # Processor - one block
    # Edge block MLP: input = edge_dim + 2*node_dim
    edge_block_params = count_mlp_params(
        hidden_dim + 2*hidden_dim, hidden_dim, hidden_dim, num_layers_edge, True
    )
    # Node block MLP: input = node_dim + edge_dim
    node_block_params = count_mlp_params(
        hidden_dim + hidden_dim, hidden_dim, hidden_dim, num_layers_node, True
    )
    block_params = edge_block_params + node_block_params
    processor_params = block_params * processor_size
    
    # Decoder
    decoder_params = count_mlp_params(hidden_dim, output_dim, hidden_dim, 2, False)
    
    total_params = encoder_params + processor_params + decoder_params
    
    return {
        'encoder': encoder_params,
        'processor': processor_params,
        'decoder': decoder_params,
        'total': total_params
    }


if __name__ == '__main__':
    print("="*70)
    print("MeshGraphNet Parameter Scaling Analysis")
    print("="*70)
    
    # Test different configurations
    configs = [
        {'name': 'Benchmark Standard', 'processor_size': 15, 'hidden_dim': 128},
        {'name': 'Medium', 'processor_size': 10, 'hidden_dim': 96},
        {'name': 'Small', 'processor_size': 5, 'hidden_dim': 64},
        {'name': 'Tiny', 'processor_size': 4, 'hidden_dim': 32},
    ]
    
    for config in configs:
        params = count_meshgraphnet_params(
            input_dim_nodes=7,
            input_dim_edges=4,
            processor_size=config['processor_size'],
            hidden_dim=config['hidden_dim']
        )
        
        print(f"\n{config['name']} Configuration:")
        print(f"  Processor size: {config['processor_size']}")
        print(f"  Hidden dim: {config['hidden_dim']}")
        print(f"  Encoder: {params['encoder']:,} params")
        print(f"  Processor: {params['processor']:,} params ({config['processor_size']} blocks)")
        print(f"  Decoder: {params['decoder']:,} params")
        print(f"  TOTAL: {params['total']:,} params")
    
    print("\n" + "="*70)
    print("Finding configuration for specific parameter counts...")
    print("="*70)
    
    # Find configs for specific targets
    targets = [1_000_000, 2_000_000, 5_000_000]
    
    for target in targets:
        best_config = None
        best_diff = float('inf')
        
        for proc_size in range(5, 25):
            for hidden_dim in range(64, 256, 8):
                params = count_meshgraphnet_params(
                    processor_size=proc_size,
                    hidden_dim=hidden_dim
                )
                diff = abs(params['total'] - target)
                if diff < best_diff:
                    best_diff = diff
                    best_config = (proc_size, hidden_dim, params['total'])
        
        proc_size, hidden_dim, actual_params = best_config
        print(f"\nTarget: {target:,} params")
        print(f"  Best config: processor_size={proc_size}, hidden_dim={hidden_dim}")
        print(f"  Actual params: {actual_params:,}")
        print(f"  Difference: {actual_params - target:+,}")
