#!/usr/bin/env python3
"""
Training script for Transolver on DrivAerNet surface pressure prediction.

This script demonstrates how to train the Transolver model using the 
DrivAerNet dataset with proper data loading and normalization.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import create_transolver
from data_loader import TransolverDataset, collate_fn


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in loader:
        # batch is a list of Data objects
        features_list = []
        coords_list = []
        targets_list = []
        
        for data in batch:
            # data.x is [N, 7] with [x, y, z, nx, ny, nz, area]
            # Transolver model separates coords and features:
            # - coords: [x, y, z] passed to positional encoding
            # - features: [nx, ny, nz, area, x, y] (6D) for input MLP
            # Drop z from features since it's in coords
            feat = torch.cat([data.x[:, 3:], data.x[:, :2]], dim=1)  # [N, 6] = [nx,ny,nz,area,x,y]
            coords = data.pos  # [N, 3] = [x, y, z]  
            target = data.y    # [N, 1]
            
            features_list.append(feat.to(device))
            coords_list.append(coords.to(device))
            targets_list.append(target.to(device))
        
        # Forward pass
        predictions = model(features_list, coords=coords_list)
        
        # Compute loss
        loss = sum(loss_fn(pred, target) for pred, target in zip(predictions, targets_list))
        loss = loss / len(batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(model, loader, loss_fn, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            features_list = []
            coords_list = []
            targets_list = []
            
            for data in batch:
                feat = torch.cat([data.x[:, 3:], data.x[:, :2]], dim=1)  # [N, 6] = [nx,ny,nz,area,x,y]
                coords = data.pos  # [N, 3]
                target = data.y    # [N, 1]
                
                features_list.append(feat.to(device))
                coords_list.append(coords.to(device))
                targets_list.append(target.to(device))
            
            predictions = model(features_list, coords=coords_list)
            loss = sum(loss_fn(pred, target) for pred, target in zip(predictions, targets_list))
            loss = loss / len(batch)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def main():
    """Main training loop"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    d_model = 208  # For 2.47M params (benchmark target)
    n_layers = 6
    batch_size = 2
    lr = 1e-4
    epochs = 100
    
    # Create model
    model = create_transolver(d_model=d_model, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # NOTE: Replace these with actual DrivAerNet data paths and design IDs
    # Example usage with real data:
    """
    from data_loader import create_dataloaders
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir='/path/to/DrivAerNet/PressureVTK',
        train_ids=['design_001', 'design_002', ...],
        val_ids=['design_101', 'design_102', ...],
        test_ids=['design_201', 'design_202', ...],
        batch_size=batch_size,
        num_workers=4,
    )
    
    # Training loop
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transolver.pth')
    """
    
    print("\nTo train on real data:")
    print("1. Update data_dir to point to DrivAerNet PressureVTK folder")
    print("2. Load train/val/test design IDs from split files")
    print("3. Uncomment the training loop above")
    print("\nFor now, running a quick synthetic test...")
    
    # Synthetic test
    n_points = 1024
    features = torch.randn(n_points, 6).to(device)
    coords = torch.randn(n_points, 3).to(device)
    target = torch.randn(n_points, 1).to(device)
    
    model.train()
    preds = model(features, coords=coords)
    loss = loss_fn(preds, target)
    loss.backward()
    optimizer.step()
    
    print(f"Synthetic test completed. Loss: {loss.item():.6f}")


if __name__ == '__main__':
    main()
