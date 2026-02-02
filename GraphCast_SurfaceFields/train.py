"""
Training Script for GraphCast on DrivAerNet

Trains GraphCast model for surface pressure prediction.

Author: Implementation for DrivAerNet benchmark
Date: February 2026
"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import create_graphcast, count_parameters
from data_loader import create_dataloaders, load_design_ids


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
    
    for batch_idx, batch in enumerate(train_loader):
        data = batch[0].to(device)  # GraphCast uses batch_size=1
        
        # Forward pass
        predictions = model(data)
        loss = criterion(predictions, data.y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate
        total_loss += loss.item()
        all_predictions.append(predictions.detach().cpu())
        all_targets.append(data.y.detach().cpu())
        
        # Log
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
    
    # Compute metrics
    avg_loss = total_loss / len(train_loader)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_predictions, all_targets)
    
    elapsed_time = time.time() - start_time
    
    # Log to tensorboard
    if writer is not None:
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/r2', metrics['r2'], epoch)
        writer.add_scalar('train/rmse', metrics['rmse'], epoch)
    
    print(f"Train Epoch {epoch}: Loss={avg_loss:.6f}, R²={metrics['r2']:.4f}, "
          f"RMSE={metrics['rmse']:.6f}, Time={elapsed_time:.1f}s")
    
    return avg_loss, metrics


def validate(model, val_loader, criterion, device, epoch, writer=None):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in val_loader:
            data = batch[0].to(device)
            
            # Forward pass
            predictions = model(data)
            loss = criterion(predictions, data.y)
            
            # Accumulate
            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(data.y.cpu())
    
    # Compute metrics
    avg_loss = total_loss / len(val_loader)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_predictions, all_targets)
    
    elapsed_time = time.time() - start_time
    
    # Log to tensorboard
    if writer is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/r2', metrics['r2'], epoch)
        writer.add_scalar('val/rmse', metrics['rmse'], epoch)
    
    print(f"Val Epoch {epoch}: Loss={avg_loss:.6f}, R²={metrics['r2']:.4f}, "
          f"RMSE={metrics['rmse']:.6f}, Time={elapsed_time:.1f}s")
    
    return avg_loss, metrics


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    device,
    num_epochs,
    save_dir,
    writer=None,
):
    """Main training loop"""
    best_val_r2 = -float('inf')
    best_epoch = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, writer
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
            if writer is not None:
                writer.add_scalar('train/lr', current_lr, epoch)
        
        # Save best model
        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            best_epoch = epoch
            
            checkpoint_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r2': best_val_r2,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, checkpoint_path)
            print(f"✓ Saved best model (R²={best_val_r2:.4f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r2': val_metrics['r2'],
                'val_loss': val_loss,
            }, checkpoint_path)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation R²: {best_val_r2:.4f} (epoch {best_epoch})")
    print(f"{'='*60}")
    
    return best_val_r2


def main():
    parser = argparse.ArgumentParser(description='Train GraphCast on DrivAerNet')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing surface field data')
    parser.add_argument('--split_dir', type=str, required=True,
                        help='Directory containing train/val/test splits')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=384,
                        help='Hidden dimension (default: 384 for ~3M params)')
    parser.add_argument('--num_mesh_nodes', type=int, default=800,
                        help='Number of mesh nodes')
    parser.add_argument('--num_processor_layers', type=int, default=12,
                        help='Number of processor layers')
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='Number of MLP layers in each block')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (GraphCast typically uses 1)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='./checkpoints/graphcast',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs/graphcast',
                        help='Directory for tensorboard logs')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load design IDs
    print("\nLoading design IDs...")
    train_ids, val_ids, test_ids = load_design_ids(args.split_dir)
    print(f"  Train: {len(train_ids)} designs")
    print(f"  Val:   {len(val_ids)} designs")
    print(f"  Test:  {len(test_ids)} designs")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=True,
        verbose=True,
    )
    
    # Create model
    print("\nCreating GraphCast model...")
    model = create_graphcast(
        hidden_dim=args.hidden_dim,
        num_mesh_nodes=args.num_mesh_nodes,
        num_processor_layers=args.num_processor_layers,
        num_mlp_layers=args.num_mlp_layers,
    ).to(device)
    
    total_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True,
    )
    
    # Tensorboard writer
    writer = SummaryWriter(args.log_dir)
    
    # Train
    print("\nStarting training...")
    best_val_r2 = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        writer=writer,
    )
    
    writer.close()
    
    print(f"\nTraining finished. Best validation R²: {best_val_r2:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")
    print(f"Logs saved to: {args.log_dir}")


if __name__ == "__main__":
    main()
