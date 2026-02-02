#!/usr/bin/env python3
"""
Training script for MeshGraphNet on DrivAerML surface pressure prediction.

MeshGraphNet uses graph neural networks with edge features for learning
mesh-based physics simulations on irregular geometries.

Reference: Pfaff, T. et al. Learning mesh-based simulation with graph networks. ICML 2021.
"""
import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from model import MeshGraphNet, count_parameters
from data_loader import create_dataloaders


def compute_metrics(predictions, targets):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: [N, 1] predicted values
        targets: [N, 1] target values
    
    Returns:
        dict with MSE, RMSE, MAE, R²
    """
    mse = torch.mean((predictions - targets) ** 2).item()
    rmse = mse ** 0.5
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    # R² score
    ss_res = torch.sum((targets - predictions) ** 2).item()
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
    }


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, writer=None):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    start_time = time.time()
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        
        # Forward pass
        predictions = model(data)  # [N, 1]
        
        # Compute loss
        loss = criterion(predictions, data.y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions and targets for metrics
        all_predictions.append(predictions.detach().cpu())
        all_targets.append(data.y.cpu())
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader)
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_predictions, all_targets)
    
    print(f"Epoch {epoch} - Train Loss: {avg_loss:.6f}, RMSE: {metrics['rmse']:.6f}, "
          f"MAE: {metrics['mae']:.6f}, R²: {metrics['r2']:.4f}, Time: {epoch_time:.1f}s")
    
    if writer:
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Train/RMSE', metrics['rmse'], epoch)
        writer.add_scalar('Train/MAE', metrics['mae'], epoch)
        writer.add_scalar('Train/R2', metrics['r2'], epoch)
    
    return avg_loss, metrics


def validate(model, val_loader, criterion, device, epoch, writer=None):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            
            predictions = model(data)
            loss = criterion(predictions, data.y)
            
            total_loss += loss.item()
            
            all_predictions.append(predictions.cpu())
            all_targets.append(data.y.cpu())
    
    avg_loss = total_loss / len(val_loader)
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_predictions, all_targets)
    
    print(f"Validation - Loss: {avg_loss:.6f}, RMSE: {metrics['rmse']:.6f}, "
          f"MAE: {metrics['mae']:.6f}, R²: {metrics['r2']:.4f}")
    
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/RMSE', metrics['rmse'], epoch)
        writer.add_scalar('Val/MAE', metrics['mae'], epoch)
        writer.add_scalar('Val/R2', metrics['r2'], epoch)
    
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Train MeshGraphNet on DrivAerML')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to DrivAerML data directory (containing run_* folders)')
    parser.add_argument('--split_dir', type=str, required=True,
                        help='Path to train/val/test split directory')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--k_neighbors', type=int, default=16,
                        help='Number of k-nearest neighbors (default: 16)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension (default: 128)')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of message passing layers (default: 6)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for logs and checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load split files
    print(f"\nLoading data from {args.data_dir}")
    print(f"Using splits from {args.split_dir}")
    
    with open(os.path.join(args.split_dir, 'train_run_ids.txt'), 'r') as f:
        train_ids = [int(line.strip()) for line in f if line.strip() and not line.startswith('#')]
    
    with open(os.path.join(args.split_dir, 'val_run_ids.txt'), 'r') as f:
        val_ids = [int(line.strip()) for line in f if line.strip() and not line.startswith('#')]
    
    with open(os.path.join(args.split_dir, 'test_run_ids.txt'), 'r') as f:
        test_ids = [int(line.strip()) for line in f if line.strip() and not line.startswith('#')]
    
    print(f"Train: {len(train_ids)} samples, Val: {len(val_ids)} samples, Test: {len(test_ids)} samples")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        k_neighbors=args.k_neighbors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Get sample to determine dimensions
    sample = next(iter(train_loader))
    node_feat_dim = sample.x.shape[1]
    edge_feat_dim = sample.edge_attr.shape[1]
    
    print(f"\nData dimensions:")
    print(f"  Node features: {node_feat_dim}")
    print(f"  Edge features: {edge_feat_dim}")
    print(f"  K neighbors: {args.k_neighbors}")
    
    # Create model
    print(f"\nCreating MeshGraphNet model:")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num layers: {args.num_layers}")
    
    model = MeshGraphNet(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    
    n_params = count_parameters(model)
    print(f"  Total parameters: {n_params:,}")
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    
    # TensorBoard
    writer = SummaryWriter(log_dir=output_dir / 'logs')
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, writer
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
        }
        
        # Save latest
        torch.save(checkpoint, output_dir / 'latest.pth')
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / 'best.pth')
            print(f"✓ New best model saved! Val Loss: {val_loss:.6f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    writer.close()
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
