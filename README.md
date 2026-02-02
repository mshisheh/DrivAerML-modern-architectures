# DrivAerML — dataset and model adapters

This folder contains the DrivAerML dataset adapters, verification scripts, and several model implementations adapted to work with the DrivAerML VTP surface format.

Contents (top-level)
- `ABUPT_SurfaceFields/` — ABUPT model and its data loader
- `GraphCast_SurfaceFields/` — GraphCast model & data loader
- `FIGConvNet_SurfaceFields/` — FIGConvNet model & data loader
- `Transolver_SurfaceFields/` — Transolver model & data loader
- `Transolver++_SurfaceFields/` — Transolver++ model & data loader
- `MeshGraphNet_SurfaceFields/` — MeshGraphNet model & data loader
- `RegDGCNN_SurfaceFields/` — RegDGCNN model & data loader
- `NeuralOperator_SurfaceFields/` — NeuralOperator model & data loader
- `train_val_test_splits/` — split files (train_run_ids.txt, val_run_ids.txt, test_run_ids.txt and design_id variants)
- `verify_dataloaders.py` — dependency & basic loader checks
- `verify_train_compatibility.py` — checks that `train.py` imports match `data_loader.py` exports
- `DRIVAERML_ADAPTATION_COMPLETE.md` — adaptation notes and decisions
- `TRAIN_COMPATIBILITY_UPDATES.md` — details of compatibility changes applied
- `INSTALLATION_GUIDE.md` — installation and setup steps
- `PROGRESS_SUMMARY.md`, `FINAL_VERIFICATION.md`, `VERIFICATION_CHECKLIST.md` — project notes and final verification
- `requirements.txt` — Python dependencies used in this folder

Quick notes
- Dataset: DrivAerML uses VTP (VTK PolyData) surface files. The adapted loaders follow the XAeroNet pattern: triangulate meshes if needed, call `cell_data_to_point_data()`, and read pressure from `point_data['pMeanTrim']`.
- Splits: `train_val_test_splits/` contains both `*_run_ids.txt` (numeric run IDs 1..500) and `*_design_ids.txt` (alphanumeric design ids). The adapted loaders for DrivAerML use the numeric `*_run_ids.txt` files by default.
- Verification: Run `python verify_train_compatibility.py` to check `train.py` ↔ `data_loader.py` compatibility. Run `python verify_dataloaders.py` to confirm dependencies and basic loader behavior.

How to run a verification check (PowerShell)
```powershell
cd C:\Learning\Scientific\CARBENCH\DrivAerML
python verify_train_compatibility.py
python verify_dataloaders.py
```

How to run training (example)
1. Ensure DrivAerML run folders (e.g. `run_1/`, `run_2/`, ...) are placed under this folder or point `--data_dir` to their location.
2. Choose a model directory (for example `GraphCast_SurfaceFields`) and run its `train.py` with the `--split_dir` pointing to `train_val_test_splits`.

Example (PowerShell):
```powershell
cd GraphCast_SurfaceFields
python train.py --data_dir ..\run_* --split_dir ..\train_val_test_splits
```

Notes
- All 8 models now have complete training scripts ready to use.
- If you edit data loader normalization or file locations, re-run `verify_train_compatibility.py`.

---

## Per-Model API Examples

### 1. GraphCast
**Architecture:** Graph neural network with multi-scale message passing  
**Input:** Node features [x, y, z, nx, ny, nz, area], KNN edges  
**Output:** Pressure at each node

```powershell
cd GraphCast_SurfaceFields
python train.py --data_dir ..\run_* --split_dir ..\train_val_test_splits --batch_size 8 --epochs 100
```

Key parameters:
- `--hidden_dim 128` — hidden layer dimension
- `--num_layers 6` — number of message passing layers
- `--k_neighbors 16` — k-nearest neighbors for graph construction

---

### 2. FIGConvNet
**Architecture:** Fast Iterative Graph Convolutional Network  
**Input:** Node features + graph structure  
**Output:** Pressure field prediction

```powershell
cd FIGConvNet_SurfaceFields
python train.py --data_dir ..\run_* --split_dir ..\train_val_test_splits --batch_size 8
```

Key parameters:
- `--hidden_channels 128` — hidden channel dimension
- `--num_layers 6` — number of graph conv layers
- `--dropout 0.1` — dropout rate

---

### 3. Transolver
**Architecture:** Physics-aware transformer for PDEs  
**Input:** Point cloud [x, y, z] + features [nx, ny, nz, area]  
**Output:** Per-point pressure prediction

```powershell
cd Transolver_SurfaceFields
python train.py --data_dir ..\run_* --split_dir ..\train_val_test_splits --batch_size 2 --epochs 100
```

Key parameters:
- `--d_model 208` — model dimension (for ~2.47M params)
- `--n_layers 6` — number of transformer layers
- `--lr 1e-4` — learning rate

---

### 4. Transolver++
**Architecture:** Advanced transformer with slicing attention for variable-sized geometries  
**Input:** Variable-sized point clouds with normals  
**Output:** Per-point pressure (handles different mesh sizes)

```powershell
cd Transolver++_SurfaceFields
python train.py --data_dir ..\run_* --split_dir ..\train_val_test_splits --batch_size 4 --epochs 100
```

Key parameters:
- `--d_model 256` — model dimension
- `--n_layers 6` — number of layers
- `--n_slices 8` — number of slices for attention mechanism
- Handles variable-sized point clouds without padding

---

### 5. MeshGraphNet
**Architecture:** Graph network for mesh-based simulation (DeepMind ICML 2021)  
**Input:** Mesh with node/edge features, KNN connectivity  
**Output:** Node-level pressure prediction

```powershell
cd MeshGraphNet_SurfaceFields
python train.py --data_dir ..\run_* --split_dir ..\train_val_test_splits --batch_size 8 --epochs 100
```

Key parameters:
- `--hidden_dim 128` — hidden dimension
- `--num_layers 6` — number of message passing layers
- `--k_neighbors 16` — k-nearest neighbors for graph edges

---

### 6. RegDGCNN
**Architecture:** Dynamic Graph CNN with regularization  
**Input:** Point cloud with dynamic graph construction  
**Output:** Per-point pressure

```powershell
cd RegDGCNN_SurfaceFields
python train.py --data_dir ..\run_* --split_dir ..\train_val_test_splits --batch_size 8
```

Key parameters:
- `--k_neighbors 16` — k for dynamic graph
- `--hidden_dim 128` — feature dimension
- Uses dynamic edge computation during forward pass

---

### 7. NeuralOperator
**Architecture:** Fourier Neural Operator for continuous functions  
**Input:** Voxelized grid (32×32×32) with geometry and boundary conditions  
**Output:** Pressure field on voxel grid

```powershell
cd NeuralOperator_SurfaceFields
python train.py --data_dir ..\run_* --split_dir ..\train_val_test_splits --batch_size 8
```

Key parameters:
- `--voxel_size 32 32 32` — grid resolution
- `--modes 12` — Fourier modes to use
- `--width 64` — channel width

---

### 8. ABUPT (Anchored-Branched Universal Physics Transformer)
**Architecture:** Multi-branch transformer for surface + volume fields  
**Input:** Point cloud + optional wall shear stress (WSS)  
**Output:** Pressure field (can also predict WSS)

```powershell
cd ABUPT_SurfaceFields
python train.py --data_dir ..\run_* --split_dir ..\train_val_test_splits --batch_size 8 --load_wss
```

Key parameters:
- `--load_wss` — also load wall shear stress data
- `--num_points 10000` — subsample to fixed number of points (optional)
- `--d_model 256` — model dimension
- Multi-output capable (pressure + WSS)

---

If anything in this README is out-of-date or you want a different format (more examples, API reference), tell me what to include and I will update it.
