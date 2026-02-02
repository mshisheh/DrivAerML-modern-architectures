"""
GraphCast Implementation for DrivAerNet Surface Pressure Prediction

Self-contained implementation without PhysicsNemo dependencies.
Based on "GraphCast: Learning skillful medium-range global weather forecasting"
(https://arxiv.org/abs/2212.12794)

Architecture:
    Encoder-Processor-Decoder with multi-scale mesh
    - Encoder: Grid-to-Mesh with learned embeddings
    - Processor: Multi-layer mesh message passing
    - Decoder: Mesh-to-Grid for final predictions

Author: Implementation for DrivAerNet benchmark
Date: February 2026
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch_geometric.data import Data


class MLP(nn.Module):
    """Multi-layer perceptron with optional layer normalization."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        activation: nn.Module = nn.SiLU(),
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_layers - 1:  # No activation/norm after last layer
                if use_layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(activation)
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class GraphCastEdgeBlock(nn.Module):
    """
    Edge update block for GraphCast.
    Updates edge features based on source/dest node features and current edge features.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        
        # Edge MLP takes: [edge_feat, src_node_feat, dst_node_feat]
        input_dim = edge_dim + 2 * node_dim
        self.edge_mlp = MLP(
            input_dim=input_dim,
            output_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            use_layer_norm=True,
        )
    
    def forward(
        self,
        edge_attr: torch.Tensor,
        node_feat_src: torch.Tensor,
        node_feat_dst: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            edge_attr: [num_edges, edge_dim]
            node_feat_src: [num_src_nodes, node_dim]
            node_feat_dst: [num_dst_nodes, node_dim]
            edge_index: [2, num_edges] - (src, dst) indices
        
        Returns:
            Updated edge features [num_edges, edge_dim]
        """
        src_idx, dst_idx = edge_index[0], edge_index[1]
        src_feat = node_feat_src[src_idx]  # [num_edges, node_dim]
        dst_feat = node_feat_dst[dst_idx]  # [num_edges, node_dim]
        
        # Concatenate: [edge_attr, src_feat, dst_feat]
        edge_input = torch.cat([edge_attr, src_feat, dst_feat], dim=-1)
        
        # Update edge features with residual
        return edge_attr + self.edge_mlp(edge_input)


class GraphCastNodeBlock(nn.Module):
    """
    Node update block for GraphCast.
    Updates destination node features by aggregating edge messages.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        aggregation: str = "sum",
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.aggregation = aggregation
        
        # Node MLP takes: [node_feat, aggregated_edge_feat]
        input_dim = node_dim + edge_dim
        self.node_mlp = MLP(
            input_dim=input_dim,
            output_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            use_layer_norm=True,
        )
    
    def forward(
        self,
        node_feat: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Args:
            node_feat: [num_nodes, node_dim]
            edge_attr: [num_edges, edge_dim]
            edge_index: [2, num_edges] - (src, dst) indices
            num_nodes: Number of destination nodes
        
        Returns:
            Updated node features [num_nodes, node_dim]
        """
        dst_idx = edge_index[1]
        
        # Aggregate edge features to destination nodes
        if self.aggregation == "sum":
            aggregated = torch.zeros(
                num_nodes, edge_attr.size(1),
                dtype=edge_attr.dtype, device=edge_attr.device
            )
            aggregated.index_add_(0, dst_idx, edge_attr)
        elif self.aggregation == "mean":
            aggregated = torch.zeros(
                num_nodes, edge_attr.size(1),
                dtype=edge_attr.dtype, device=edge_attr.device
            )
            counts = torch.zeros(num_nodes, 1, dtype=torch.float32, device=edge_attr.device)
            aggregated.index_add_(0, dst_idx, edge_attr)
            counts.index_add_(0, dst_idx, torch.ones(len(dst_idx), 1, device=edge_attr.device))
            aggregated = aggregated / (counts + 1e-8)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Concatenate node features with aggregated edge features
        node_input = torch.cat([node_feat, aggregated], dim=-1)
        
        # Update node features with residual
        return node_feat + self.node_mlp(node_input)


class GraphCastEncoder(nn.Module):
    """
    Grid-to-Mesh Encoder.
    Encodes grid features onto mesh nodes via bipartite graph.
    """
    
    def __init__(
        self,
        grid_dim: int,
        mesh_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        aggregation: str = "sum",
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.aggregation = aggregation
        
        # Edge update block
        self.edge_block = GraphCastEdgeBlock(
            node_dim=hidden_dim,  # After embedding
            edge_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
        )
        
        # Node update for mesh nodes
        self.mesh_node_block = GraphCastNodeBlock(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            aggregation=aggregation,
            activation=activation,
        )
        
        # Simple MLP for grid nodes (identity-like)
        self.grid_mlp = MLP(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            use_layer_norm=True,
        )
    
    def forward(
        self,
        grid_feat: torch.Tensor,
        mesh_feat: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            grid_feat: [num_grid_nodes, hidden_dim]
            mesh_feat: [num_mesh_nodes, hidden_dim]
            edge_attr: [num_edges, hidden_dim]
            edge_index: [2, num_edges] - grid-to-mesh edges
        
        Returns:
            Updated grid_feat, mesh_feat
        """
        # Update edges
        edge_attr = self.edge_block(edge_attr, grid_feat, mesh_feat, edge_index)
        
        # Update mesh nodes
        mesh_feat = self.mesh_node_block(mesh_feat, edge_attr, edge_index, mesh_feat.size(0))
        
        # Update grid nodes (simple residual)
        grid_feat = grid_feat + self.grid_mlp(grid_feat)
        
        return grid_feat, mesh_feat


class GraphCastProcessor(nn.Module):
    """
    Mesh Processor.
    Applies message passing on the mesh graph for multiple layers.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_processor_layers: int = 16,
        num_mlp_layers: int = 1,
        aggregation: str = "sum",
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'edge_block': GraphCastEdgeBlock(
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_mlp_layers,
                    activation=activation,
                ),
                'node_block': GraphCastNodeBlock(
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_mlp_layers,
                    aggregation=aggregation,
                    activation=activation,
                )
            })
            for _ in range(num_processor_layers)
        ])
    
    def forward(
        self,
        node_feat: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_feat: [num_mesh_nodes, node_dim]
            edge_attr: [num_edges, edge_dim]
            edge_index: [2, num_edges] - mesh edges
        
        Returns:
            Updated node_feat, edge_attr
        """
        for layer in self.layers:
            # Update edges
            edge_attr = layer['edge_block'](edge_attr, node_feat, node_feat, edge_index)
            
            # Update nodes
            node_feat = layer['node_block'](node_feat, edge_attr, edge_index, node_feat.size(0))
        
        return node_feat, edge_attr


class GraphCastDecoder(nn.Module):
    """
    Mesh-to-Grid Decoder.
    Decodes mesh features back to grid for final predictions.
    """
    
    def __init__(
        self,
        mesh_dim: int,
        grid_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        aggregation: str = "sum",
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        
        # Node update for grid nodes
        self.grid_node_block = GraphCastNodeBlock(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            aggregation=aggregation,
            activation=activation,
        )
    
    def forward(
        self,
        grid_feat: torch.Tensor,
        mesh_feat: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            grid_feat: [num_grid_nodes, hidden_dim]
            mesh_feat: [num_mesh_nodes, hidden_dim]
            edge_attr: [num_edges, hidden_dim]
            edge_index: [2, num_edges] - mesh-to-grid edges
        
        Returns:
            Updated grid_feat
        """
        # Update grid nodes from mesh
        grid_feat = self.grid_node_block(grid_feat, edge_attr, edge_index, grid_feat.size(0))
        
        return grid_feat


class GraphCast(nn.Module):
    """
    Complete GraphCast model for DrivAerNet surface pressure prediction.
    
    Architecture:
        1. Embedder: Embed grid/mesh node features and edge features
        2. Encoder: Grid-to-Mesh encoding
        3. Processor: Multi-layer mesh message passing
        4. Decoder: Mesh-to-Grid decoding
        5. Output: Final MLP for pressure prediction
    
    Parameters:
        input_dim: Input feature dimension (7: x,y,z,nx,ny,nz,area)
        output_dim: Output dimension (1: pressure)
        hidden_dim: Hidden dimension throughout network
        num_mesh_nodes: Number of mesh nodes for latent representation
        num_processor_layers: Number of processor message passing layers
        num_mlp_layers: Number of MLP layers in each block
        mesh_k: Number of k-nearest neighbors for mesh connectivity
        g2m_k: Number of k-nearest neighbors for grid-to-mesh
        m2g_k: Number of k-nearest neighbors for mesh-to-grid
        aggregation: Message aggregation method ('sum' or 'mean')
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        output_dim: int = 1,
        hidden_dim: int = 512,
        num_mesh_nodes: int = 1000,
        num_processor_layers: int = 16,
        num_mlp_layers: int = 1,
        mesh_k: int = 10,
        g2m_k: int = 4,
        m2g_k: int = 4,
        aggregation: str = "sum",
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_mesh_nodes = num_mesh_nodes
        self.mesh_k = mesh_k
        self.g2m_k = g2m_k
        self.m2g_k = m2g_k
        
        activation = nn.SiLU()
        
        # Embedders
        self.grid_embedder = MLP(
            input_dim=input_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            activation=activation,
            use_layer_norm=True,
        )
        
        self.mesh_embedder = MLP(
            input_dim=3,  # x, y, z
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            activation=activation,
            use_layer_norm=True,
        )
        
        # Edge feature dimension: [dx, dy, dz, distance]
        self.edge_embedder = MLP(
            input_dim=4,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            activation=activation,
            use_layer_norm=True,
        )
        
        # Encoder (Grid-to-Mesh)
        self.encoder = GraphCastEncoder(
            grid_dim=hidden_dim,
            mesh_dim=hidden_dim,
            edge_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            aggregation=aggregation,
            activation=activation,
        )
        
        # Processor (Mesh processing)
        self.processor = GraphCastProcessor(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_processor_layers=num_processor_layers,
            num_mlp_layers=num_mlp_layers,
            aggregation=aggregation,
            activation=activation,
        )
        
        # Decoder (Mesh-to-Grid)
        self.decoder = GraphCastDecoder(
            mesh_dim=hidden_dim,
            grid_dim=hidden_dim,
            edge_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            aggregation=aggregation,
            activation=activation,
        )
        
        # Output MLP
        self.output_mlp = MLP(
            input_dim=hidden_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            activation=activation,
            use_layer_norm=False,  # No norm in final layer
        )
    
    def _compute_edge_features(
        self, pos_src: torch.Tensor, pos_dst: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute edge features: [dx, dy, dz, distance]"""
        src_idx, dst_idx = edge_index[0], edge_index[1]
        src_pos = pos_src[src_idx]
        dst_pos = pos_dst[dst_idx]
        
        diff = dst_pos - src_pos
        dist = torch.norm(diff, dim=-1, keepdim=True)
        
        edge_feat = torch.cat([diff, dist], dim=-1)
        return edge_feat
    
    def _build_mesh_graph(self, grid_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build mesh graph by sampling mesh nodes and constructing k-NN connectivity.
        
        Returns:
            mesh_pos: [num_mesh_nodes, 3]
            mesh_edge_index: [2, num_mesh_edges]
            mesh_edge_attr: [num_mesh_edges, 4]
        """
        # Sample mesh nodes using FPS (Farthest Point Sampling) approximation
        # For simplicity, use random sampling (can be replaced with FPS)
        num_grid = grid_pos.size(0)
        if num_grid <= self.num_mesh_nodes:
            # If grid is smaller than mesh, just use all points
            mesh_indices = torch.arange(num_grid, device=grid_pos.device)
        else:
            # Random sampling for now (could use FPS for better coverage)
            mesh_indices = torch.randperm(num_grid, device=grid_pos.device)[:self.num_mesh_nodes]
        
        mesh_pos = grid_pos[mesh_indices]
        
        # Build k-NN graph on mesh
        from torch_geometric.nn import knn_graph
        mesh_edge_index = knn_graph(mesh_pos, k=self.mesh_k, loop=False)
        
        # Compute edge features
        mesh_edge_attr = self._compute_edge_features(mesh_pos, mesh_pos, mesh_edge_index)
        
        return mesh_pos, mesh_edge_index, mesh_edge_attr
    
    def _build_bipartite_graph(
        self, src_pos: torch.Tensor, dst_pos: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build bipartite graph between source and destination nodes.
        
        Returns:
            edge_index: [2, num_edges]
            edge_attr: [num_edges, 4]
        """
        # For each src node, find k nearest dst nodes
        # Compute pairwise distances
        # src: [N, 3], dst: [M, 3]
        # dist: [N, M]
        dist = torch.cdist(src_pos, dst_pos)
        
        # Get k nearest neighbors
        _, knn_indices = torch.topk(dist, k=min(k, dst_pos.size(0)), largest=False, dim=1)
        
        # Build edge index
        num_src = src_pos.size(0)
        src_indices = torch.arange(num_src, device=src_pos.device).unsqueeze(1).expand(-1, knn_indices.size(1))
        edge_index = torch.stack([src_indices.reshape(-1), knn_indices.reshape(-1)], dim=0)
        
        # Compute edge features
        edge_attr = self._compute_edge_features(src_pos, dst_pos, edge_index)
        
        return edge_index, edge_attr
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: PyG Data object with:
                - data.x: [num_nodes, input_dim] - node features
                - data.pos: [num_nodes, 3] - node positions (x, y, z)
        
        Returns:
            predictions: [num_nodes, output_dim] - predicted pressure
        """
        grid_feat = data.x
        grid_pos = data.pos
        
        # Build mesh graph (latent space)
        mesh_pos, mesh_edge_index, mesh_edge_attr_raw = self._build_mesh_graph(grid_pos)
        
        # Build bipartite graphs
        g2m_edge_index, g2m_edge_attr_raw = self._build_bipartite_graph(grid_pos, mesh_pos, self.g2m_k)
        m2g_edge_index, m2g_edge_attr_raw = self._build_bipartite_graph(mesh_pos, grid_pos, self.m2g_k)
        
        # Embed features
        grid_feat_emb = self.grid_embedder(grid_feat)
        mesh_feat_emb = self.mesh_embedder(mesh_pos)
        g2m_edge_attr = self.edge_embedder(g2m_edge_attr_raw)
        mesh_edge_attr = self.edge_embedder(mesh_edge_attr_raw)
        m2g_edge_attr = self.edge_embedder(m2g_edge_attr_raw)
        
        # Encoder: Grid-to-Mesh
        grid_feat_enc, mesh_feat_enc = self.encoder(
            grid_feat_emb, mesh_feat_emb, g2m_edge_attr, g2m_edge_index
        )
        
        # Processor: Mesh message passing
        mesh_feat_proc, mesh_edge_attr_proc = self.processor(
            mesh_feat_enc, mesh_edge_attr, mesh_edge_index
        )
        
        # Decoder: Mesh-to-Grid
        grid_feat_dec = self.decoder(
            grid_feat_enc, mesh_feat_proc, m2g_edge_attr, m2g_edge_index
        )
        
        # Output
        output = self.output_mlp(grid_feat_dec)
        
        return output


def create_graphcast(
    hidden_dim: int = 512,
    num_mesh_nodes: int = 1000,
    num_processor_layers: int = 16,
    num_mlp_layers: int = 1,
    input_dim: int = 7,
    output_dim: int = 1,
) -> GraphCast:
    """
    Factory function to create GraphCast model.
    
    Parameter configurations for target sizes:
    - ~3M params: hidden_dim=384, num_mesh_nodes=800, num_processor_layers=12
    - ~4M params: hidden_dim=448, num_mesh_nodes=900, num_processor_layers=14
    - ~5M params: hidden_dim=512, num_mesh_nodes=1000, num_processor_layers=16
    """
    model = GraphCast(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_mesh_nodes=num_mesh_nodes,
        num_processor_layers=num_processor_layers,
        num_mlp_layers=num_mlp_layers,
        mesh_k=10,
        g2m_k=4,
        m2g_k=4,
        aggregation="sum",
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("Testing GraphCast Model Creation...\n")
    
    # Create model
    model = create_graphcast(
        hidden_dim=384,
        num_mesh_nodes=800,
        num_processor_layers=12,
    )
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Test forward pass
    import torch
    from torch_geometric.data import Data
    
    # Dummy data
    num_nodes = 50000
    x = torch.randn(num_nodes, 7)
    pos = torch.randn(num_nodes, 3)
    data = Data(x=x, pos=pos)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Position shape: {pos.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(data)
    
    print(f"Output shape: {output.shape}")
    print("\n✓ GraphCast model test passed!")
