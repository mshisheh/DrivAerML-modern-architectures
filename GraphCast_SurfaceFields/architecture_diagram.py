"""
GraphCast Architecture Diagram for DrivAerNet

Comprehensive visualization of the GraphCast architecture adapted for
automotive aerodynamics surface pressure prediction.

Author: Implementation for DrivAerNet benchmark
Date: February 2026
"""

import sys
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def print_graphcast_architecture():
    """Print comprehensive GraphCast architecture diagram"""
    
    diagram = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    GraphCast Architecture for DrivAerNet                    ║
║                         Surface Pressure Prediction                          ║
║                                                                              ║
║  Based on: "GraphCast: Learning skillful medium-range global weather        ║
║             forecasting" (Lam et al., Nature 2023)                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════
                              INPUT STAGE
═══════════════════════════════════════════════════════════════════════════════

Surface Points: N ≈ 50,000 - 100,000 points per vehicle
─────────────────────────────────────────────────────────────────────────────

    Input Features (N, 7):
    ┌─────────────────────────────────────────────────┐
    │  • Coordinates:   x, y, z                       │
    │  • Normals:       nx, ny, nz                    │
    │  • Area:          Approximate area per point    │
    └─────────────────────────────────────────────────┘
                        ↓


═══════════════════════════════════════════════════════════════════════════════
                          EMBEDDING STAGE
═══════════════════════════════════════════════════════════════════════════════

Three parallel embedders map features to hidden dimension:

┌────────────────────────────┐  ┌────────────────────────────┐
│   Grid Embedder MLP        │  │   Mesh Embedder MLP        │
│   Input:  (N, 7)           │  │   Input:  (M, 3)           │
│   Output: (N, hidden_dim)  │  │   Output: (M, hidden_dim)  │
│                            │  │                            │
│   • x, y, z                │  │   • Mesh positions         │
│   • nx, ny, nz             │  │   • Sampled from surface   │
│   • area                   │  │   • M ≈ 800-1200 nodes     │
└────────────────────────────┘  └────────────────────────────┘
                        ↓
             ┌────────────────────────────┐
             │   Edge Embedder MLP        │
             │   Input:  (*, 4)           │
             │   Output: (*, hidden_dim)  │
             │                            │
             │   • dx, dy, dz             │
             │   • distance               │
             └────────────────────────────┘

Parameters: ~1.5M for hidden_dim=384


═══════════════════════════════════════════════════════════════════════════════
                          ENCODER STAGE (Grid-to-Mesh)
═══════════════════════════════════════════════════════════════════════════════

Bipartite graph connecting surface points to latent mesh:

    Grid Features (N, D)    ←────→    Mesh Features (M, D)
         │                                     │
         │  Grid-to-Mesh Edges (E_g2m)       │
         │  • k-NN connectivity (k=4)         │
         │  • ~4N edges                       │
         └──────────────┬────────────────────┘
                        ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Encoder Message Passing                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Edge Update:                                                            │
│     e_ij' = e_ij + MLP([e_ij, h_i^grid, h_j^mesh])                        │
│                                                                             │
│     Input:  [edge_attr, src_feat, dst_feat]  → [D, D, D] = 3D             │
│     MLP:    3D → D (with LayerNorm, SiLU activation)                       │
│     Residual: e_ij' = e_ij + MLP_output                                    │
│                                                                             │
│  2. Mesh Node Update:                                                       │
│     h_j^mesh' = h_j^mesh + MLP([h_j^mesh, Σ_i e_ij'])                    │
│                                                                             │
│     Aggregation: Sum edge messages to mesh nodes                           │
│     Input:  [node_feat, aggregated_edges]  → [D, D] = 2D                  │
│     MLP:    2D → D (with LayerNorm, SiLU activation)                       │
│     Residual: h_j^mesh' = h_j^mesh + MLP_output                           │
│                                                                             │
│  3. Grid Node Update:                                                       │
│     h_i^grid' = h_i^grid + MLP(h_i^grid)                                  │
│                                                                             │
│     Simple identity-like update for grid                                   │
│     Residual connection maintains information                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Output: 
  • Grid Features (N, D) - encoded grid state
  • Mesh Features (M, D) - information aggregated from grid

Parameters: ~0.3M for hidden_dim=384


═══════════════════════════════════════════════════════════════════════════════
                        PROCESSOR STAGE (Mesh)
═══════════════════════════════════════════════════════════════════════════════

Multi-layer message passing on latent mesh (L layers, typically 12-16):

    Mesh Graph: k-NN connectivity (k=10)
    Mesh Nodes: M ≈ 800-1200
    Mesh Edges: ~10M edges

┌─────────────────────────────────────────────────────────────────────────────┐
│                  Processor Layer × L (L = 12-16)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  For each layer l = 1, ..., L:                                              │
│                                                                             │
│  1. Edge Update:                                                            │
│     e_ij^(l) = e_ij^(l-1) + EdgeMLP([e_ij^(l-1), h_i^(l-1), h_j^(l-1)])  │
│                                                                             │
│     • EdgeMLP: 3D → D with LayerNorm + SiLU                                │
│     • Combines edge features with source/dest node features                │
│     • Residual connection for gradient flow                                │
│                                                                             │
│  2. Node Update:                                                            │
│     h_j^(l) = h_j^(l-1) + NodeMLP([h_j^(l-1), Σ_i e_ij^(l)])             │
│                                                                             │
│     • Aggregate edge messages (sum or mean)                                │
│     • NodeMLP: 2D → D with LayerNorm + SiLU                                │
│     • Residual connection maintains long-range information                 │
│                                                                             │
│  This structure allows mesh nodes to communicate and refine their          │
│  representations over multiple hops, capturing global flow patterns.       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        Processor Layer Visualization                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│     Layer 1:  Initial mesh communication                                   │
│              ○─────○─────○                                                 │
│              │╲   ╱│╲   ╱│                                                 │
│              │ ╲ ╱ │ ╲ ╱ │                                                 │
│              │  ○  │  ○  │                                                 │
│              ↓  ↓  ↓  ↓  ↓                                                 │
│                                                                             │
│     Layer 2-11: Deep processing (captures long-range dependencies)         │
│              ○═══○═══○═══○                                                 │
│              ║   ║   ║   ║                                                 │
│              ○═══○═══○═══○                                                 │
│              ↓  ↓  ↓  ↓  ↓                                                 │
│                                                                             │
│     Layer 12: Final refinement                                             │
│              ○─────○─────○                                                 │
│              │╲   ╱│╲   ╱│                                                 │
│              │ ╲ ╱ │ ╲ ╱ │                                                 │
│              │  ○  │  ○  │                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Output: Refined Mesh Features (M, D)

Parameters: ~14M for hidden_dim=384, L=12 (dominant component!)


═══════════════════════════════════════════════════════════════════════════════
                        DECODER STAGE (Mesh-to-Grid)
═══════════════════════════════════════════════════════════════════════════════

Bipartite graph connecting mesh back to surface points:

    Mesh Features (M, D)    ────→    Grid Features (N, D)
         │                                     │
         │  Mesh-to-Grid Edges (E_m2g)       │
         │  • k-NN connectivity (k=4)         │
         │  • ~4N edges                       │
         └──────────────┬────────────────────┘
                        ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Decoder Message Passing                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Grid Node Update:                                                          │
│     h_i^grid = h_i^grid + MLP([h_i^grid, Σ_j e_ji])                       │
│                                                                             │
│     • Aggregate edge messages from mesh to grid                            │
│     • MLP: 2D → D with LayerNorm + SiLU                                    │
│     • Combines encoded grid state with processed mesh information          │
│     • Residual connection preserves original grid features                 │
│                                                                             │
│  This transfers the refined global information from the mesh back to       │
│  individual surface points for local predictions.                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Output: Decoded Grid Features (N, D)

Parameters: ~0.15M for hidden_dim=384


═══════════════════════════════════════════════════════════════════════════════
                            OUTPUT STAGE
═══════════════════════════════════════════════════════════════════════════════

    Decoded Grid Features (N, D)
                ↓
    ┌────────────────────────────┐
    │     Output MLP             │
    │                            │
    │  Linear: D → D             │
    │  LayerNorm                 │
    │  SiLU Activation           │
    │  Linear: D → 1             │
    └────────────────────────────┘
                ↓
    Pressure Predictions (N, 1)

Parameters: ~0.4K for hidden_dim=384


═══════════════════════════════════════════════════════════════════════════════
                         PARAMETER BREAKDOWN
═══════════════════════════════════════════════════════════════════════════════

For hidden_dim = 384, num_processor_layers = 12, num_mesh_nodes = 800:

┌─────────────────────────────────────────────────────────────────────────────┐
│ Component              │ Parameters  │ % of Total │ Description             │
├────────────────────────┼─────────────┼────────────┼─────────────────────────┤
│ Embedders              │   ~1.5M     │    10%     │ Grid/Mesh/Edge MLPs     │
│ Encoder                │   ~0.3M     │     2%     │ Grid-to-Mesh            │
│ Processor              │  ~14.0M     │    86%     │ 12 Mesh layers          │
│ Decoder                │   ~0.15M    │     1%     │ Mesh-to-Grid            │
│ Output MLP             │   ~0.4K     │    <1%     │ Final prediction        │
├────────────────────────┼─────────────┼────────────┼─────────────────────────┤
│ TOTAL                  │  ~16.0M     │   100%     │                         │
└─────────────────────────────────────────────────────────────────────────────┘

Key observations:
• Processor dominates parameter count (86%)
• Scaling: Params ≈ D² × L × 10
• For target ~3M params: D=384, L=12
• For target ~5M params: D=512, L=16


═══════════════════════════════════════════════════════════════════════════════
                      MODEL SIZE CONFIGURATIONS
═══════════════════════════════════════════════════════════════════════════════

┌────────┬─────────┬─────────────────┬────────────────────┬────────────────┐
│ Config │ hidden_ │ num_mesh_nodes  │ num_processor_     │ Total Params   │
│        │   dim   │                 │    layers          │                │
├────────┼─────────┼─────────────────┼────────────────────┼────────────────┤
│ Tiny   │   192   │      400        │         6          │    ~0.8M       │
│ Small  │   256   │      600        │         8          │    ~1.5M       │
│ Medium │   384   │      800        │        12          │    ~3.0M       │
│ Large  │   512   │     1000        │        16          │    ~5.5M       │
│ XLarge │   640   │     1200        │        20          │    ~9.0M       │
└────────┴─────────┴─────────────────┴────────────────────┴────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                          COMPLEXITY ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

Time Complexity (per forward pass):
───────────────────────────────────────────────────────────────────────────────

1. Mesh Construction:       O(N log N)  [k-NN search]
2. Bipartite Graphs:         O(N log M)  [g2m, m2g k-NN]
3. Embedders:                O(N·D + M·D + E·D)
4. Encoder:                  O(E_g2m·D²)
5. Processor:                O(L·E_mesh·D²)  [Dominant!]
6. Decoder:                  O(E_m2g·D²)
7. Output:                   O(N·D)

Total: O(L·E_mesh·D²) where E_mesh ≈ 10M

Memory Complexity:
───────────────────────────────────────────────────────────────────────────────

1. Node Features:            O(N·D + M·D)
2. Edge Features:            O(E·D)  where E = E_g2m + E_mesh + E_m2g
3. Intermediate Activations: O(L·M·D)  [Processor depth]
4. Gradients:                O(Parameters) ≈ O(D²·L)

Total: O(N·D + L·M·D + E·D)

For N=50k, M=800, D=384, L=12, E≈45k:
  • Forward memory: ~2-3 GB
  • Backward memory: ~4-6 GB


═══════════════════════════════════════════════════════════════════════════════
                      KEY ARCHITECTURAL FEATURES
═══════════════════════════════════════════════════════════════════════════════

1. Multi-Scale Representation:
   • Surface points (N ≈ 50k-100k): Local, high-resolution
   • Latent mesh (M ≈ 800-1200): Global, compressed representation
   • Enables efficient long-range communication

2. Residual Connections:
   • All edge and node updates use residuals
   • Enables deep networks (12-16 layers)
   • Improves gradient flow

3. Message Passing Structure:
   • Edge updates: Combine edge, source, and destination features
   • Node updates: Aggregate edge messages with aggregation function
   • Flexible aggregation (sum or mean)

4. Bipartite Graphs:
   • Grid-to-Mesh: Information encoding
   • Mesh-to-Grid: Information decoding
   • k-NN connectivity for locality

5. Layer Normalization:
   • Applied after each linear layer
   • Stabilizes training
   • Enables higher learning rates


═══════════════════════════════════════════════════════════════════════════════
                      COMPARISON WITH ORIGINAL GRAPHCAST
═══════════════════════════════════════════════════════════════════════════════

┌────────────────────┬─────────────────────────┬─────────────────────────────┐
│ Aspect             │ Original GraphCast      │ DrivAerNet Adaptation       │
├────────────────────┼─────────────────────────┼─────────────────────────────┤
│ Input Domain       │ Lat-lon grid (721×1440) │ Irregular surface mesh      │
│ Input Size         │ ~1M grid points         │ ~50k-100k points           │
│ Mesh Type          │ Icosahedral hierarchy   │ k-NN sampled mesh          │
│ Mesh Nodes         │ ~40k (level 6)          │ ~800-1200                  │
│ Processor Layers   │ 16-36                   │ 12-16                      │
│ Hidden Dimension   │ 512                     │ 256-512                    │
│ Total Parameters   │ ~40M                    │ ~3-5M                      │
│ Application        │ Weather forecasting     │ Automotive aerodynamics    │
│ Dependencies       │ PhysicsNemo, DGL        │ PyTorch, PyG only          │
└────────────────────┴─────────────────────────┴─────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                            TRAINING TIPS
═══════════════════════════════════════════════════════════════════════════════

1. Learning Rate:
   • Start: 1e-4 to 5e-4
   • Warmup: Linear warmup over first 5-10 epochs
   • Scheduler: ReduceLROnPlateau with patience=10

2. Regularization:
   • Weight decay: 1e-5
   • Gradient clipping: max_norm=1.0
   • Dropout: Not typically needed due to heavy residual connections

3. Batch Size:
   • Typically: 1 (full vehicle per batch)
   • Memory permitting: 2-4
   • Use gradient accumulation if needed

4. Optimization:
   • AdamW optimizer
   • β1=0.9, β2=0.999
   • No momentum or Nesterov

5. Graph Caching:
   • Cache k-NN graphs if memory allows
   • Recompute only if input geometry changes
   • Speeds up training by 2-3×


═══════════════════════════════════════════════════════════════════════════════
                              REFERENCES
═══════════════════════════════════════════════════════════════════════════════

Primary Paper:
  Lam, R., et al. (2022). "GraphCast: Learning skillful medium-range global
  weather forecasting." arXiv:2212.12794
  Published in Nature, 2023.

Related Work:
  - Pfaff, T., et al. (2020). "Learning Mesh-Based Simulation with Graph Networks"
  - Sanchez-Gonzalez, A., et al. (2020). "Learning to Simulate Complex Physics"
  - Battaglia, P., et al. (2018). "Relational inductive biases, deep learning,
    and graph networks"

Implementation Reference:
  NVIDIA Modulus (PhysicsNemo)
  https://github.com/NVIDIA/modulus


═══════════════════════════════════════════════════════════════════════════════
                              END OF DIAGRAM
═══════════════════════════════════════════════════════════════════════════════

"""
    
    print(diagram)


if __name__ == "__main__":
    print_graphcast_architecture()
