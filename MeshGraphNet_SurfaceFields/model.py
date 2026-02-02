"""
MeshGraphNet model for DrivAerNet surface pressure prediction.

Architecture: Encode-Process-Decode with graph neural networks
- Encoder: Maps node/edge features to latent space
- Processor: Multiple message-passing blocks (EdgeBlock → NodeBlock)
- Decoder: Maps latent node features to pressure predictions

Reference: Pfaff, T. et al. Learning mesh-based simulation with graph networks. ICML 2021.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from typing import Optional


class MLP(nn.Module):
    """
    Multi-layer perceptron with LayerNorm and optional residual.
    
    Parameters
    ----------
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    hidden_dim : int
        Hidden layer dimension
    num_layers : int
        Number of hidden layers
    activation : nn.Module, optional
        Activation function, by default ReLU
    use_layer_norm : bool, optional
        Whether to use LayerNorm, by default True
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU(),
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        if num_layers == 0:
            # Identity mapping
            self.layers = nn.Identity()
            self.is_identity = True
        else:
            self.is_identity = False
            layers = []
            
            # First layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation)
            
            # Hidden layers
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(activation)
            
            # Output layer
            layers.append(nn.Linear(hidden_dim, output_dim))
            
            self.layers = nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class MeshEdgeBlock(nn.Module):
    """
    Edge update block: processes edges using [edge_feat, src_node, dst_node].
    
    Parameters
    ----------
    node_dim : int
        Node feature dimension
    edge_dim : int
        Edge feature dimension
    hidden_dim : int
        Hidden layer dimension
    num_layers : int
        Number of MLP layers
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int
    ):
        super().__init__()
        # Input: [edge_features, src_node_features, dst_node_features]
        input_dim = edge_dim + 2 * node_dim
        self.edge_mlp = MLP(input_dim, edge_dim, hidden_dim, num_layers)
    
    def forward(
        self,
        edge_feat: Tensor,
        node_feat: Tensor,
        edge_index: Tensor
    ) -> Tensor:
        """
        Parameters
        ----------
        edge_feat : Tensor
            Edge features, shape (num_edges, edge_dim)
        node_feat : Tensor
            Node features, shape (num_nodes, node_dim)
        edge_index : Tensor
            Edge connectivity, shape (2, num_edges)
        
        Returns
        -------
        Tensor
            Updated edge features, shape (num_edges, edge_dim)
        """
        src, dst = edge_index
        # Concatenate: [edge_features, src_node_features, dst_node_features]
        edge_input = torch.cat([edge_feat, node_feat[src], node_feat[dst]], dim=1)
        # Update edges with residual connection
        edge_feat_new = self.edge_mlp(edge_input) + edge_feat
        return edge_feat_new


class MeshNodeBlock(nn.Module):
    """
    Node update block: aggregates incoming edges and updates nodes.
    
    Parameters
    ----------
    node_dim : int
        Node feature dimension
    edge_dim : int
        Edge feature dimension
    hidden_dim : int
        Hidden layer dimension
    num_layers : int
        Number of MLP layers
    aggregation : str, optional
        Edge aggregation method ('sum' or 'mean'), by default 'sum'
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        aggregation: str = 'sum'
    ):
        super().__init__()
        self.aggregation = aggregation
        # Input: [node_features, aggregated_edge_features]
        input_dim = node_dim + edge_dim
        self.node_mlp = MLP(input_dim, node_dim, hidden_dim, num_layers)
    
    def forward(
        self,
        edge_feat: Tensor,
        node_feat: Tensor,
        edge_index: Tensor
    ) -> Tensor:
        """
        Parameters
        ----------
        edge_feat : Tensor
            Edge features, shape (num_edges, edge_dim)
        node_feat : Tensor
            Node features, shape (num_nodes, node_dim)
        edge_index : Tensor
            Edge connectivity, shape (2, num_edges)
        
        Returns
        -------
        Tensor
            Updated node features, shape (num_nodes, node_dim)
        """
        src, dst = edge_index
        num_nodes = node_feat.size(0)
        
        # Aggregate incoming edge features for each destination node
        if self.aggregation == 'sum':
            aggregated = torch.zeros(
                num_nodes, edge_feat.size(1),
                dtype=edge_feat.dtype, device=edge_feat.device
            )
            aggregated.index_add_(0, dst, edge_feat)
        elif self.aggregation == 'mean':
            aggregated = torch.zeros(
                num_nodes, edge_feat.size(1),
                dtype=edge_feat.dtype, device=edge_feat.device
            )
            count = torch.zeros(
                num_nodes, 1,
                dtype=edge_feat.dtype, device=edge_feat.device
            )
            aggregated.index_add_(0, dst, edge_feat)
            count.index_add_(0, dst, torch.ones_like(edge_feat[:, :1]))
            aggregated = aggregated / (count + 1e-8)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Concatenate node features with aggregated edges
        node_input = torch.cat([node_feat, aggregated], dim=1)
        # Update nodes with residual connection
        node_feat_new = self.node_mlp(node_input) + node_feat
        return node_feat_new


class MeshGraphNet(nn.Module):
    """
    MeshGraphNet: Encode-Process-Decode architecture for mesh-based learning.
    
    Architecture:
    1. Encoder: Maps node/edge features to latent space
    2. Processor: Multiple message-passing blocks (edge update → node update)
    3. Decoder: Maps latent node features to output predictions
    
    Parameters
    ----------
    input_dim_nodes : int
        Number of node features (default: 7 for [x,y,z,nx,ny,nz,area])
    input_dim_edges : int
        Number of edge features (default: 4 for [dx,dy,dz,distance])
    output_dim : int, optional
        Number of outputs, by default 1 (pressure)
    processor_size : int, optional
        Number of message-passing blocks, by default 15
    hidden_dim : int, optional
        Latent dimension, by default 128
    num_layers_node : int, optional
        MLP layers in node processor, by default 2
    num_layers_edge : int, optional
        MLP layers in edge processor, by default 2
    aggregation : str, optional
        Edge aggregation method, by default 'sum'
    """
    
    def __init__(
        self,
        input_dim_nodes: int = 7,
        input_dim_edges: int = 4,
        output_dim: int = 1,
        processor_size: int = 15,
        hidden_dim: int = 128,
        num_layers_node: int = 2,
        num_layers_edge: int = 2,
        aggregation: str = 'sum'
    ):
        super().__init__()
        
        self.input_dim_nodes = input_dim_nodes
        self.input_dim_edges = input_dim_edges
        self.output_dim = output_dim
        self.processor_size = processor_size
        self.hidden_dim = hidden_dim
        self.num_layers_node = num_layers_node
        self.num_layers_edge = num_layers_edge
        self.aggregation = aggregation
        
        # Encoder: Map input features to latent space
        self.node_encoder = MLP(
            input_dim_nodes, hidden_dim, hidden_dim, 2
        )
        self.edge_encoder = MLP(
            input_dim_edges, hidden_dim, hidden_dim, 2
        )
        
        # Processor: Message passing blocks
        self.edge_blocks = nn.ModuleList([
            MeshEdgeBlock(hidden_dim, hidden_dim, hidden_dim, num_layers_edge)
            for _ in range(processor_size)
        ])
        self.node_blocks = nn.ModuleList([
            MeshNodeBlock(hidden_dim, hidden_dim, hidden_dim, num_layers_node, aggregation)
            for _ in range(processor_size)
        ])
        
        # Decoder: Map latent features to output
        self.node_decoder = MLP(
            hidden_dim, output_dim, hidden_dim, 2, use_layer_norm=False
        )
    
    def forward(self, data: Data) -> Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph with x, edge_index, edge_attr
        
        Returns
        -------
        Tensor
            Predicted pressure values, shape (num_nodes,)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Encode
        node_feat = self.node_encoder(x)
        edge_feat = self.edge_encoder(edge_attr)
        
        # Process: Alternate edge and node updates
        for i in range(self.processor_size):
            edge_feat = self.edge_blocks[i](edge_feat, node_feat, edge_index)
            node_feat = self.node_blocks[i](edge_feat, node_feat, edge_index)
        
        # Decode
        output = self.node_decoder(node_feat)
        return output.squeeze(-1)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_meshgraphnet(
    input_dim_nodes: int = 7,
    input_dim_edges: int = 4,
    processor_size: int = 15,
    hidden_dim: int = 128
) -> MeshGraphNet:
    """
    Factory function to create MeshGraphNet with standard configuration.
    
    For DrivAerNet surface pressure prediction:
    - input_dim_nodes = 7: [x, y, z, n_x, n_y, n_z, area]
    - input_dim_edges = 4: [dx, dy, dz, distance]
    
    Parameters
    ----------
    input_dim_nodes : int, optional
        Node feature dimension, by default 7
    input_dim_edges : int, optional
        Edge feature dimension, by default 4
    processor_size : int, optional
        Number of message-passing blocks, by default 15
    hidden_dim : int, optional
        Latent dimension, by default 128
    
    Returns
    -------
    MeshGraphNet
        Initialized model
    """
    model = MeshGraphNet(
        input_dim_nodes=input_dim_nodes,
        input_dim_edges=input_dim_edges,
        output_dim=1,
        processor_size=processor_size,
        hidden_dim=hidden_dim,
        num_layers_node=2,
        num_layers_edge=2,
        aggregation='sum'
    )
    
    params = model.count_parameters()
    print(f"Created MeshGraphNet with {params:,} parameters")
    print(f"  Processor size: {processor_size}")
    print(f"  Hidden dim: {hidden_dim}")
    
    return model


if __name__ == '__main__':
    # Test model creation and parameter counting
    from torch_geometric.data import Data
    
    print("Testing MeshGraphNet...")
    
    # Create test configurations
    configs = [
        (15, 128, "Standard"),  # processor_size, hidden_dim, name
        (10, 96, "Medium"),
        (5, 64, "Small"),
    ]
    
    print("\n" + "="*70)
    print("MeshGraphNet Parameter Scaling")
    print("="*70)
    
    for processor_size, hidden_dim, name in configs:
        model = create_meshgraphnet(
            input_dim_nodes=7,
            input_dim_edges=4,
            processor_size=processor_size,
            hidden_dim=hidden_dim
        )
        
        print(f"\n{name} Configuration:")
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Processor blocks: {processor_size}")
        print(f"  Hidden dim: {hidden_dim}")
    
    # Test forward pass
    print("\n" + "="*70)
    print("Testing forward pass...")
    print("="*70)
    
    model = create_meshgraphnet(processor_size=5, hidden_dim=64)
    
    # Create dummy graph
    num_nodes = 100
    num_edges = 300
    
    x = torch.randn(num_nodes, 7)  # Node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Edge connectivity
    edge_attr = torch.randn(num_edges, 4)  # Edge features
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    output = model(data)
    print(f"\nInput: {num_nodes} nodes, {num_edges} edges")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("\nForward pass successful!")
