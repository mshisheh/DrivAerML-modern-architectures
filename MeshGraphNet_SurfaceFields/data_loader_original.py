"""
Data loader for MeshGraphNet on DrivAerNet surface pressure prediction.

MeshGraphNet requires graph structure:
- Nodes: Surface points with features [x, y, z, nx, ny, nz, area]
- Edges: Connectivity based on k-nearest neighbors
- Edge features: [dx, dy, dz, distance]
- Target: Pressure values at each node

Reference: Pfaff, T. et al. Learning mesh-based simulation with graph networks. ICML 2021.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pyvista as pv
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from torch_geometric.data import Data


class MeshGraphDataset(Dataset):
    """
    Dataset for loading surface meshes as graphs for MeshGraphNet.
    
    Constructs k-NN graph from surface points:
    - Node features: [x, y, z, n_x, n_y, n_z, area] (7D)
    - Edge features: [dx, dy, dz, distance] (4D)
    - Target: Pressure at each node
    
    Parameters
    ----------
    data_dir : str
        Directory containing .vtk surface files
    split : str
        One of 'train', 'val', 'test'
    k_neighbors : int, optional
        Number of nearest neighbors for graph construction, by default 6
    normalize : bool, optional
        Whether to normalize features, by default True
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        k_neighbors: int = 6,
        normalize: bool = True
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.k_neighbors = k_neighbors
        self.normalize = normalize
        
        # Load design IDs for this split
        split_file = self.data_dir.parent / 'train_val_test_splits' / f'{split}_design_ids.txt'
        with open(split_file, 'r') as f:
            self.design_ids = [line.strip() for line in f if line.strip()]
        
        # Find corresponding .vtk files
        self.file_paths = []
        for design_id in self.design_ids:
            # Look for files matching pattern (e.g., Design_1234.vtk)
            vtk_files = list(self.data_dir.glob(f'{design_id}.vtk'))
            if not vtk_files:
                # Try alternative pattern
                vtk_files = list(self.data_dir.glob(f'*{design_id}*.vtk'))
            if vtk_files:
                self.file_paths.append(vtk_files[0])
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No .vtk files found in {self.data_dir} for {split} split")
        
        print(f"Loaded {len(self.file_paths)} files for {split} split")
        
        # Compute normalization statistics from training set
        if self.normalize and split == 'train':
            self._compute_normalization_stats()
        elif self.normalize:
            # Load stats computed from training set
            stats_file = self.data_dir / 'normalization_stats.npz'
            if stats_file.exists():
                stats = np.load(stats_file)
                self.pos_mean = stats['pos_mean']
                self.pos_std = stats['pos_std']
                self.normal_mean = stats['normal_mean']
                self.normal_std = stats['normal_std']
                self.area_mean = stats['area_mean']
                self.area_std = stats['area_std']
                self.pressure_mean = stats['pressure_mean']
                self.pressure_std = stats['pressure_std']
    
    def _compute_normalization_stats(self):
        """Compute dataset-wide normalization statistics."""
        print("Computing normalization statistics from training set...")
        
        all_pos = []
        all_normals = []
        all_areas = []
        all_pressures = []
        
        # Sample a subset for efficiency
        sample_indices = np.linspace(0, len(self.file_paths) - 1, min(100, len(self.file_paths)), dtype=int)
        
        for idx in sample_indices:
            mesh = pv.read(self.file_paths[idx])
            points = np.array(mesh.points)
            
            # Compute normals and areas
            mesh = mesh.compute_normals(point_normals=True, cell_normals=False)
            normals = np.array(mesh.point_data['Normals'])
            
            # Compute approximate area per point
            areas = self._compute_point_areas(mesh)
            
            # Get pressure
            pressure = np.array(mesh.point_data['p'])
            
            all_pos.append(points)
            all_normals.append(normals)
            all_areas.append(areas)
            all_pressures.append(pressure)
        
        all_pos = np.concatenate(all_pos, axis=0)
        all_normals = np.concatenate(all_normals, axis=0)
        all_areas = np.concatenate(all_areas, axis=0)
        all_pressures = np.concatenate(all_pressures, axis=0)
        
        self.pos_mean = np.mean(all_pos, axis=0)
        self.pos_std = np.std(all_pos, axis=0) + 1e-6
        self.normal_mean = np.mean(all_normals, axis=0)
        self.normal_std = np.std(all_normals, axis=0) + 1e-6
        self.area_mean = np.mean(all_areas)
        self.area_std = np.std(all_areas) + 1e-6
        self.pressure_mean = np.mean(all_pressures)
        self.pressure_std = np.std(all_pressures) + 1e-6
        
        # Save stats
        np.savez(
            self.data_dir / 'normalization_stats.npz',
            pos_mean=self.pos_mean,
            pos_std=self.pos_std,
            normal_mean=self.normal_mean,
            normal_std=self.normal_std,
            area_mean=self.area_mean,
            area_std=self.area_std,
            pressure_mean=self.pressure_mean,
            pressure_std=self.pressure_std
        )
        print("Normalization statistics saved.")
    
    def _compute_point_areas(self, mesh: pv.PolyData) -> np.ndarray:
        """
        Compute approximate area associated with each point.
        
        Each point gets 1/3 of the area of each triangle it belongs to.
        """
        n_points = mesh.n_points
        areas = np.zeros(n_points)
        
        # Compute cell (triangle) areas
        mesh = mesh.compute_cell_sizes(area=True, length=False, volume=False)
        cell_areas = mesh.cell_data['Area']
        
        # Distribute cell areas to points
        for i, face in enumerate(mesh.faces.reshape(-1, 4)):
            # face format: [n_vertices, v0, v1, v2]
            vertices = face[1:]
            area_per_point = cell_areas[i] / 3.0
            areas[vertices] += area_per_point
        
        return areas
    
    def _build_knn_graph(self, points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build k-nearest neighbor graph.
        
        Returns
        -------
        edge_index : np.ndarray
            Edge connectivity, shape (2, num_edges)
        edge_attr : np.ndarray
            Edge features [dx, dy, dz, distance], shape (num_edges, 4)
        """
        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # Build edge list (exclude self-loops)
        edge_list = []
        edge_features = []
        
        for i in range(len(points)):
            for j in range(1, k + 1):  # Skip first neighbor (self)
                neighbor_idx = indices[i, j]
                
                # Directed edge i -> neighbor
                edge_list.append([i, neighbor_idx])
                
                # Edge features: relative position + distance
                dx = points[neighbor_idx] - points[i]
                dist = distances[i, j]
                edge_features.append(np.concatenate([dx, [dist]]))
        
        edge_index = np.array(edge_list, dtype=np.int64).T  # Shape: (2, num_edges)
        edge_attr = np.array(edge_features, dtype=np.float32)  # Shape: (num_edges, 4)
        
        return edge_index, edge_attr
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Data:
        """
        Load mesh and convert to graph.
        
        Returns
        -------
        data : torch_geometric.data.Data
            Graph with:
            - x: node features [x, y, z, n_x, n_y, n_z, area] (7D)
            - edge_index: connectivity (2, num_edges)
            - edge_attr: edge features [dx, dy, dz, distance] (4D)
            - y: target pressure values
            - pos: original positions (for visualization)
        """
        # Load mesh
        mesh = pv.read(self.file_paths[idx])
        points = np.array(mesh.points, dtype=np.float32)
        
        # Compute normals
        mesh = mesh.compute_normals(point_normals=True, cell_normals=False)
        normals = np.array(mesh.point_data['Normals'], dtype=np.float32)
        
        # Compute areas
        areas = self._compute_point_areas(mesh).astype(np.float32)
        
        # Get pressure
        pressure = np.array(mesh.point_data['p'], dtype=np.float32)
        
        # Normalize if requested
        if self.normalize:
            points_norm = (points - self.pos_mean) / self.pos_std
            normals_norm = (normals - self.normal_mean) / self.normal_std
            areas_norm = (areas - self.area_mean) / self.area_std
            pressure_norm = (pressure - self.pressure_mean) / self.pressure_std
        else:
            points_norm = points
            normals_norm = normals
            areas_norm = areas
            pressure_norm = pressure
        
        # Build k-NN graph
        edge_index, edge_attr = self._build_knn_graph(points, self.k_neighbors)
        
        # Construct node features: [x, y, z, n_x, n_y, n_z, area]
        node_features = np.concatenate([
            points_norm,
            normals_norm,
            areas_norm.reshape(-1, 1)
        ], axis=1)
        
        # Create PyG Data object
        data = Data(
            x=torch.from_numpy(node_features).float(),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(edge_attr).float(),
            y=torch.from_numpy(pressure_norm).float(),
            pos=torch.from_numpy(points).float()  # Keep original positions
        )
        
        return data


def create_dataloaders(
    data_dir: str,
    batch_size: int = 1,
    k_neighbors: int = 6,
    num_workers: int = 0,
    normalize: bool = True
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train, validation, and test dataloaders.
    
    Note: Batch size is typically 1 for variable-size graphs.
    """
    from torch_geometric.loader import DataLoader
    
    train_dataset = MeshGraphDataset(data_dir, split='train', k_neighbors=k_neighbors, normalize=normalize)
    val_dataset = MeshGraphDataset(data_dir, split='val', k_neighbors=k_neighbors, normalize=normalize)
    test_dataset = MeshGraphDataset(data_dir, split='test', k_neighbors=k_neighbors, normalize=normalize)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test data loading
    data_dir = 'path/to/drivaeernet/surface_fields'  # Update this path
    
    print("Testing MeshGraphDataset...")
    dataset = MeshGraphDataset(data_dir, split='train', k_neighbors=6)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load first sample
    data = dataset[0]
    print(f"\nSample graph:")
    print(f"  Nodes: {data.x.shape[0]}")
    print(f"  Node features: {data.x.shape[1]} (x, y, z, nx, ny, nz, area)")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Edge features: {data.edge_attr.shape[1]} (dx, dy, dz, distance)")
    print(f"  Target shape: {data.y.shape}")
    print(f"\nNode feature statistics:")
    print(f"  Min: {data.x.min(dim=0).values}")
    print(f"  Max: {data.x.max(dim=0).values}")
