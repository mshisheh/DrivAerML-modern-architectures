"""
Data Loader for MeshGraphNet on DrivAerML

Loads VTP surface mesh files and constructs k-NN graphs for MeshGraphNet training.
Follows XAeroNet preprocessor pattern for data extraction.

Key Pattern (from XAeroNet preprocessor.py):
1. Load VTP file
2. Convert to triangular mesh
3. Convert cell_data to point_data (CRITICAL)
4. Extract from point_data: points, normals, pMeanTrim
5. Compute KNN edges on point data

MeshGraphNet Structure:
- Nodes: Surface points with features [x, y, z, nx, ny, nz]
- Edges: K-nearest neighbors connectivity
- Edge features: [dx, dy, dz, distance]
- Target: Pressure at each node

Reference: Pfaff, T. et al. Learning mesh-based simulation with graph networks. ICML 2021.

Author: Implementation for DrivAerML benchmark
Date: February 2026
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from typing import List, Tuple, Optional
import pyvista as pv
import vtk
from scipy.spatial import cKDTree


class MeshGraphDataset(Dataset):
    """
    Dataset for MeshGraphNet surface pressure prediction on DrivAerML.
    
    Data structure:
        - Nodes: Surface points with features [x, y, z, nx, ny, nz]
        - Edges: K-nearest neighbors connectivity
        - Edge features: [dx, dy, dz, distance]
        - Geometry parameters: 16 design variables from CSV
        - Target: Pressure (pMeanTrim) at points
        
    Following XAeroNet pattern:
        - Load VTP → Triangulate → cell_data_to_point_data → Extract
        - Use point_data["pMeanTrim"] (not cell_data["CpMeanTrim"])
        - Use mesh vertices (not cell centers)
        - Use point normals (not cell normals)
        - Compute KNN edges on mesh vertices
    """
    
    def __init__(
        self,
        data_dir: str,
        run_ids: List[int],
        k_neighbors: int = 6,
        normalize: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            data_dir: Root directory containing DrivAerML data (run_X folders)
            run_ids: List of run IDs to load (1-500)
            k_neighbors: Number of nearest neighbors for graph construction
            normalize: Whether to normalize features
            verbose: Print loading information
        """
        self.data_dir = data_dir
        self.run_ids = run_ids
        self.k_neighbors = k_neighbors
        self.normalize = normalize
        self.verbose = verbose
        
        # Normalization statistics
        self.coords_mean = None
        self.coords_std = None
        self.normals_mean = None
        self.normals_std = None
        self.geo_params_mean = None
        self.geo_params_std = None
        self.pressure_mean = None
        self.pressure_std = None
        self.edge_mean = None
        self.edge_std = None
        
        # Verify data exists
        self._verify_data()
        
        if self.verbose:
            print(f"MeshGraphDataset initialized with {len(self.run_ids)} runs, k={k_neighbors}")
    
    def _verify_data(self):
        """Check that data files exist"""
        missing_runs = []
        missing_csvs = []
        
        for run_id in self.run_ids[:min(5, len(self.run_ids))]:
            vtp_file = os.path.join(self.data_dir, f"run_{run_id}", f"boundary_{run_id}.vtp")
            csv_file = os.path.join(self.data_dir, f"geo_parameters_{run_id}.csv")
            
            if not os.path.exists(vtp_file):
                missing_runs.append(run_id)
            if not os.path.exists(csv_file):
                missing_csvs.append(run_id)
        
        if missing_runs:
            raise FileNotFoundError(
                f"Missing VTP files for runs: {missing_runs}\n"
                f"Expected location: {self.data_dir}/run_X/boundary_X.vtp"
            )
        if missing_csvs:
            raise FileNotFoundError(
                f"Missing geometry CSV files for runs: {missing_csvs}\n"
                f"Expected location: {self.data_dir}/geo_parameters_X.csv"
            )
    
    def _load_vtp_with_point_data(self, vtp_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load VTP file and extract point-based data.
        
        Follows XAeroNet preprocessor.py pattern (lines 155-161):
        1. Load VTP
        2. Convert to triangular mesh
        3. Convert cell_data to point_data (CRITICAL)
        4. Extract points, normals, pressure from point_data
        
        Args:
            vtp_file: Path to VTP file
            
        Returns:
            points: [N, 3] - mesh vertices (x, y, z)
            normals: [N, 3] - point normals (nx, ny, nz)
            pressure: [N] - pressure at points (pMeanTrim)
        """
        # Step 1: Load VTP
        surf = pv.read(vtp_file)
        
        # Step 2: Convert to triangular mesh (if not already)
        if surf.n_cells > 0 and surf.get_cell(0).type != vtk.VTK_TRIANGLE:
            tet_filter = vtk.vtkDataSetTriangleFilter()
            tet_filter.SetInputData(surf)
            tet_filter.Update()
            surf = pv.wrap(tet_filter.GetOutput())
        
        # Step 3: CRITICAL - Convert cell_data to point_data
        # This is the key step from XAeroNet preprocessor.py
        surf = surf.cell_data_to_point_data()
        
        # Step 4: Extract from point_data
        points = surf.points  # Mesh vertices (not cell centers)
        
        # Compute normals at points (not cells)
        surf_with_normals = surf.compute_normals(point_normals=True, cell_normals=False)
        normals = surf_with_normals.point_data["Normals"]
        
        # Extract pressure from point_data
        # Priority: pMeanTrim > CpMeanTrim > pressure > p
        pressure = None
        for field_name in ["pMeanTrim", "CpMeanTrim", "pressure", "p"]:
            if field_name in surf.point_data:
                pressure = surf.point_data[field_name]
                if self.verbose:
                    print(f"Using field: {field_name}")
                break
        
        if pressure is None:
            available = list(surf.point_data.keys())
            raise ValueError(f"No pressure field found in VTP. Available: {available}")
        
        return points, normals, pressure
    
    def _load_geometry_parameters(self, run_id: int) -> np.ndarray:
        """
        Load geometry parameters from CSV.
        
        Args:
            run_id: Run ID
            
        Returns:
            geo_params: [16] - design variables
        """
        csv_file = os.path.join(self.data_dir, f"geo_parameters_{run_id}.csv")
        df = pd.read_csv(csv_file)
        
        # Expected 16 parameters (excluding ID column)
        geo_params = df.iloc[0, 1:].values.astype(np.float32)
        
        if len(geo_params) != 16:
            raise ValueError(f"Expected 16 geometry parameters, got {len(geo_params)}")
        
        return geo_params
    
    def _build_knn_graph(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build k-nearest neighbor graph on point data.
        
        Args:
            points: [N, 3] - point coordinates
            
        Returns:
            edge_index: [2, E] - edge connectivity (COO format)
            edge_attr: [E, 4] - edge features [dx, dy, dz, distance]
        """
        # Build KDTree for efficient nearest neighbor search
        tree = cKDTree(points)
        
        # Query k+1 nearest neighbors (includes self)
        distances, indices = tree.query(points, k=self.k_neighbors + 1)
        
        # Build edge list (exclude self-loops)
        edge_list = []
        edge_features = []
        
        for i in range(len(points)):
            for j in range(1, self.k_neighbors + 1):  # Skip first (self)
                neighbor_idx = indices[i, j]
                
                # Directed edge: i -> neighbor
                edge_list.append([i, neighbor_idx])
                
                # Edge features: relative position + distance
                dx = points[neighbor_idx] - points[i]
                dist = distances[i, j]
                edge_features.append(np.concatenate([dx, [dist]]))
        
        edge_index = np.array(edge_list, dtype=np.int64).T  # [2, E]
        edge_attr = np.array(edge_features, dtype=np.float32)  # [E, 4]
        
        return edge_index, edge_attr
    
    def compute_normalization_stats(self):
        """Compute mean and std for normalization"""
        if self.verbose:
            print("Computing normalization statistics...")
        
        all_coords = []
        all_normals = []
        all_pressures = []
        all_geo_params = []
        all_edge_attrs = []
        
        # Sample subset for stats (first 50 or all if less)
        sample_size = min(50, len(self))
        for idx in range(sample_size):
            run_id = self.run_ids[idx]
            vtp_file = os.path.join(self.data_dir, f"run_{run_id}", f"boundary_{run_id}.vtp")
            
            points, normals, pressure = self._load_vtp_with_point_data(vtp_file)
            geo_params = self._load_geometry_parameters(run_id)
            _, edge_attr = self._build_knn_graph(points)
            
            all_coords.append(points)
            all_normals.append(normals)
            all_pressures.append(pressure)
            all_geo_params.append(geo_params)
            all_edge_attrs.append(edge_attr)
        
        # Concatenate
        all_coords = np.concatenate(all_coords, axis=0)
        all_normals = np.concatenate(all_normals, axis=0)
        all_pressures = np.concatenate(all_pressures, axis=0)
        all_geo_params = np.stack(all_geo_params, axis=0)
        all_edge_attrs = np.concatenate(all_edge_attrs, axis=0)
        
        # Compute statistics
        self.coords_mean = torch.tensor(all_coords.mean(axis=0), dtype=torch.float32)
        self.coords_std = torch.tensor(all_coords.std(axis=0) + 1e-8, dtype=torch.float32)
        self.normals_mean = torch.tensor(all_normals.mean(axis=0), dtype=torch.float32)
        self.normals_std = torch.tensor(all_normals.std(axis=0) + 1e-8, dtype=torch.float32)
        self.geo_params_mean = torch.tensor(all_geo_params.mean(axis=0), dtype=torch.float32)
        self.geo_params_std = torch.tensor(all_geo_params.std(axis=0) + 1e-8, dtype=torch.float32)
        self.pressure_mean = torch.tensor(all_pressures.mean(), dtype=torch.float32)
        self.pressure_std = torch.tensor(all_pressures.std() + 1e-8, dtype=torch.float32)
        self.edge_mean = torch.tensor(all_edge_attrs.mean(axis=0), dtype=torch.float32)
        self.edge_std = torch.tensor(all_edge_attrs.std(axis=0) + 1e-8, dtype=torch.float32)
        
        if self.verbose:
            print(f"Coordinates - mean: {self.coords_mean.numpy()}, std: {self.coords_std.numpy()}")
            print(f"Normals - mean: {self.normals_mean.numpy()}, std: {self.normals_std.numpy()}")
            print(f"Edge features - mean: {self.edge_mean.numpy()}, std: {self.edge_std.numpy()}")
            print(f"Pressure - mean: {self.pressure_mean:.4f}, std: {self.pressure_std:.4f}")
    
    def __len__(self) -> int:
        return len(self.run_ids)
    
    def __getitem__(self, idx: int) -> Data:
        """
        Load and return a single graph sample.
        
        Returns:
            PyTorch Geometric Data object with:
                - x: [num_nodes, 6] - node features [x, y, z, nx, ny, nz]
                - pos: [num_nodes, 3] - node positions (for visualization)
                - edge_index: [2, num_edges] - edge connectivity
                - edge_attr: [num_edges, 4] - edge features [dx, dy, dz, dist]
                - y: [num_nodes] - target pressure
                - geo_params: [16] - geometry parameters
                - run_id: int - run identifier
        """
        run_id = self.run_ids[idx]
        vtp_file = os.path.join(self.data_dir, f"run_{run_id}", f"boundary_{run_id}.vtp")
        
        # Load VTP with point-based data (XAeroNet pattern)
        points, normals, pressure = self._load_vtp_with_point_data(vtp_file)
        
        # Load geometry parameters
        geo_params = self._load_geometry_parameters(run_id)
        
        # Build KNN graph on point data
        edge_index, edge_attr = self._build_knn_graph(points)
        
        # Convert to tensors
        positions = torch.tensor(points, dtype=torch.float32)
        normals_tensor = torch.tensor(normals, dtype=torch.float32)
        pressures = torch.tensor(pressure, dtype=torch.float32)
        geo_params_tensor = torch.tensor(geo_params, dtype=torch.float32)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
        
        # Normalize
        if self.normalize:
            if self.coords_mean is None:
                self.compute_normalization_stats()
            
            positions_norm = (positions - self.coords_mean) / self.coords_std
            normals_norm = (normals_tensor - self.normals_mean) / self.normals_std
            geo_params_norm = (geo_params_tensor - self.geo_params_mean) / self.geo_params_std
            pressures_norm = (pressures - self.pressure_mean) / self.pressure_std
            edge_attr_norm = (edge_attr_tensor - self.edge_mean) / self.edge_std
            
            # Node features: concatenate normalized coordinates and normals
            node_features = torch.cat([positions_norm, normals_norm], dim=-1)  # [N, 6]
            
            # Create PyG Data object
            data = Data(
                x=node_features,
                pos=positions,  # Original positions (for visualization)
                edge_index=edge_index_tensor,
                edge_attr=edge_attr_norm,
                y=pressures_norm,
            )
            data.geo_params = geo_params_norm
            data.run_id = run_id
            
        else:
            # Node features: concatenate coordinates and normals
            node_features = torch.cat([positions, normals_tensor], dim=-1)  # [N, 6]
            
            # Create PyG Data object
            data = Data(
                x=node_features,
                pos=positions,
                edge_index=edge_index_tensor,
                edge_attr=edge_attr_tensor,
                y=pressures,
            )
            data.geo_params = geo_params_tensor
            data.run_id = run_id
        
        return data


def create_dataloaders(
    data_dir: str,
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    k_neighbors: int = 6,
    batch_size: int = 1,
    num_workers: int = 4,
    normalize: bool = True,
    verbose: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root directory containing DrivAerML data
        train_ids: List of training run IDs (1-500)
        val_ids: List of validation run IDs
        test_ids: List of test run IDs
        k_neighbors: Number of nearest neighbors for graph construction
        batch_size: Batch size
        num_workers: Number of data loading workers
        normalize: Whether to normalize features
        verbose: Print information
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = MeshGraphDataset(
        data_dir=data_dir,
        run_ids=train_ids,
        k_neighbors=k_neighbors,
        normalize=normalize,
        verbose=verbose,
    )
    
    val_dataset = MeshGraphDataset(
        data_dir=data_dir,
        run_ids=val_ids,
        k_neighbors=k_neighbors,
        normalize=normalize,
        verbose=verbose,
    )
    
    test_dataset = MeshGraphDataset(
        data_dir=data_dir,
        run_ids=test_ids,
        k_neighbors=k_neighbors,
        normalize=normalize,
        verbose=False,
    )
    
    # Compute normalization stats on training set
    if normalize:
        train_dataset.compute_normalization_stats()
        
        # Share stats with val and test
        val_dataset.coords_mean = train_dataset.coords_mean
        val_dataset.coords_std = train_dataset.coords_std
        val_dataset.normals_mean = train_dataset.normals_mean
        val_dataset.normals_std = train_dataset.normals_std
        val_dataset.geo_params_mean = train_dataset.geo_params_mean
        val_dataset.geo_params_std = train_dataset.geo_params_std
        val_dataset.pressure_mean = train_dataset.pressure_mean
        val_dataset.pressure_std = train_dataset.pressure_std
        val_dataset.edge_mean = train_dataset.edge_mean
        val_dataset.edge_std = train_dataset.edge_std
        
        test_dataset.coords_mean = train_dataset.coords_mean
        test_dataset.coords_std = train_dataset.coords_std
        test_dataset.normals_mean = train_dataset.normals_mean
        test_dataset.normals_std = train_dataset.normals_std
        test_dataset.geo_params_mean = train_dataset.geo_params_mean
        test_dataset.geo_params_std = train_dataset.geo_params_std
        test_dataset.pressure_mean = train_dataset.pressure_mean
        test_dataset.pressure_std = train_dataset.pressure_std
        test_dataset.edge_mean = train_dataset.edge_mean
        test_dataset.edge_std = train_dataset.edge_std
    
    # Create dataloaders
    # Note: MeshGraphNet can batch graphs with PyG's default collate
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Load train/val/test splits
    data_dir = "C:/Learning/Scientific/CARBENCH/DrivAerML"
    
    with open("../train_val_test_splits/train_run_ids.txt", "r") as f:
        train_ids = [int(line.strip()) for line in f]
    
    with open("../train_val_test_splits/val_run_ids.txt", "r") as f:
        val_ids = [int(line.strip()) for line in f]
    
    with open("../train_val_test_splits/test_run_ids.txt", "r") as f:
        test_ids = [int(line.strip()) for line in f]
    
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        k_neighbors=6,
        batch_size=2,
        num_workers=0,
        normalize=True,
        verbose=True,
    )
    
    # Test loading
    print("\nTesting data loading...")
    for batch in train_loader:
        print(f"Batch size: {batch.num_graphs}")
        print(f"Node features shape: {batch.x.shape}")
        print(f"Edge index shape: {batch.edge_index.shape}")
        print(f"Edge attr shape: {batch.edge_attr.shape}")
        print(f"Positions shape: {batch.pos.shape}")
        print(f"Target shape: {batch.y.shape}")
        print(f"Run IDs: {batch.run_id}")
        break
    
    print("\nMeshGraphNet data loader ready!")
