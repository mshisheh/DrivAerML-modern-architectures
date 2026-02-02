#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for Fourier Neural Operator on DrivAerNet++ surface pressure prediction.

Usage:
    python train.py --dataset_path /path/to/PressureVTK --subset_dir /path/to/splits

Reference: Li, Z. et al. Fourier neural operator for parametric partial differential equations. 
           arXiv preprint arXiv:2010.08895 (2020).

@author: NeuralOperator Implementation for DrivAerNet
"""

import os
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

from data_loader import get_dataloaders, PRESSURE_MEAN, PRESSURE_STD
from model import FNOSurfaceFieldPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compute_metrics(predictions, targets):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: List of (N_i, 1) tensors
        targets: List of (N_i,) tensors
        
    Returns:
        Dictionary of metrics
    """
    all_preds = torch.cat([p.squeeze() for p in predictions])
    all_targets = torch.cat(targets)
    
    mse = torch.mean((all_preds - all_targets) ** 2).item()
    mae = torch.mean(torch.abs(all_preds - all_targets)).item()
    
    # R² score
    ss_res = torch.sum((all_targets - all_preds) ** 2)
    ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
    r2 = (1 - ss_res / ss_tot).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'rmse': np.sqrt(mse),
    }


def train_epoch(model, train_loader, optimizer, criterion, device, normalize=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        if batch is None:
            continue
        
        voxel_grids = batch['voxel_grid'].to(device)
        positions = [pos.to(device) for pos in batch['positions']]
        pressures = [p.to(device) for p in batch['pressures']]
        bboxes = batch['bboxes']
        
        # Normalize targets
        if normalize:
            pressures = [(p - PRESSURE_MEAN) / PRESSURE_STD for p in pressures]
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(voxel_grids, positions, bboxes)
        
        # Compute loss
        loss = 0
        for pred, target in zip(predictions, pressures):
            loss += criterion(pred.squeeze(), target)
        loss = loss / len(predictions)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect for metrics (denormalize)
        if normalize:
            predictions = [p * PRESSURE_STD + PRESSURE_MEAN for p in predictions]
            pressures = [p * PRESSURE_STD + PRESSURE_MEAN for p in pressures]
        
        all_predictions.extend([p.detach() for p in predictions])
        all_targets.extend([p.detach() for p in pressures])
        
        pbar.set_postfix({'loss': loss.item()})
    
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['loss'] = total_loss / len(train_loader)
    
    return metrics


@torch.no_grad()
def validate(model, val_loader, criterion, device, normalize=True):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(val_loader, desc='Validation')
    for batch in pbar:
        if batch is None:
            continue
        
        voxel_grids = batch['voxel_grid'].to(device)
        positions = [pos.to(device) for pos in batch['positions']]
        pressures = [p.to(device) for p in batch['pressures']]
        bboxes = batch['bboxes']
        
        # Normalize targets
        if normalize:
            pressures_norm = [(p - PRESSURE_MEAN) / PRESSURE_STD for p in pressures]
        else:
            pressures_norm = pressures
        
        # Forward pass
        predictions = model(voxel_grids, positions, bboxes)
        
        # Compute loss
        loss = 0
        for pred, target in zip(predictions, pressures_norm):
            loss += criterion(pred.squeeze(), target)
        loss = loss / len(predictions)
        
        total_loss += loss.item()
        
        # Collect for metrics (denormalized)
        if normalize:
            predictions = [p * PRESSURE_STD + PRESSURE_MEAN for p in predictions]
        
        all_predictions.extend([p.detach() for p in predictions])
        all_targets.extend([p.detach() for p in pressures])
        
        pbar.set_postfix({'loss': loss.item()})
    
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    weight_decay,
    device,
    save_dir,
    normalize=True,
):
    """Complete training loop."""
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_val_r2 = -float('inf')
    best_epoch = 0
    
    logging.info("Starting training...")
    logging.info(f"Model parameters: {model.count_parameters():,}")
    
    for epoch in range(num_epochs):
        logging.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, normalize)
        logging.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                    f"MSE: {train_metrics['mse']:.2f}, "
                    f"MAE: {train_metrics['mae']:.2f}, "
                    f"R²: {train_metrics['r2']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, normalize)
        logging.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                    f"MSE: {val_metrics['mse']:.2f}, "
                    f"MAE: {val_metrics['mae']:.2f}, "
                    f"R²: {val_metrics['r2']:.4f}")
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logging.info(f"Learning rate: {current_lr:.6f}")
        
        # Save best model
        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            best_epoch = epoch + 1
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r2': best_val_r2,
                'val_metrics': val_metrics,
            }, checkpoint_path)
            logging.info(f"Saved best model with R² = {best_val_r2:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, checkpoint_path)
    
    logging.info(f"\nTraining complete!")
    logging.info(f"Best validation R²: {best_val_r2:.4f} at epoch {best_epoch}")
    
    return best_val_r2


def main():
    parser = argparse.ArgumentParser(description='Train FNO for surface pressure prediction')
    
    # Data parameters
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to VTK files')
    parser.add_argument('--subset_dir', type=str, required=True,
                       help='Directory with train/val/test ID files')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Directory for cached voxelized data')
    
    # Model parameters
    parser.add_argument('--grid_resolution', type=int, default=32,
                       help='Voxel grid resolution (default: 32)')
    parser.add_argument('--fno_modes', type=int, default=8,
                       help='Number of Fourier modes (default: 8)')
    parser.add_argument('--fno_width', type=int, default=16,
                       help='FNO hidden dimension (default: 16)')
    parser.add_argument('--fno_layers', type=int, default=4,
                       help='Number of FNO layers (default: 4)')
    parser.add_argument('--refine_hidden', type=int, default=64,
                       help='Refinement network hidden size (default: 64)')
    
    # Training parameters
    parser.add_argument('--num_points', type=int, default=10000,
                       help='Number of surface points (default: 10000)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                       help='Learning rate (default: 2e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--no_normalize', action='store_true',
                       help='Disable target normalization')
    
    # Other
    parser.add_argument('--save_dir', type=str, default='./checkpoints_fno',
                       help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.save_dir, 'config.txt')
    with open(config_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_path=args.dataset_path,
        subset_dir=args.subset_dir,
        grid_resolution=args.grid_resolution,
        num_points=args.num_points,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
    )
    
    # Create model
    model = FNOSurfaceFieldPredictor(
        grid_resolution=args.grid_resolution,
        fno_modes=args.fno_modes,
        fno_width=args.fno_width,
        fno_layers=args.fno_layers,
        refine_hidden=args.refine_hidden,
    ).to(device)
    
    # Train
    best_val_r2 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        save_dir=args.save_dir,
        normalize=not args.no_normalize,
    )
    
    # Test on best model
    logging.info("\nEvaluating on test set...")
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, nn.MSELoss(), device, normalize=not args.no_normalize)
    logging.info(f"Test - MSE: {test_metrics['mse']:.2f}, "
                f"MAE: {test_metrics['mae']:.2f}, "
                f"R²: {test_metrics['r2']:.4f}")
    
    # Save test results
    results_path = os.path.join(args.save_dir, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Test Results\n")
        f.write(f"============\n")
        f.write(f"MSE: {test_metrics['mse']:.2f}\n")
        f.write(f"MAE: {test_metrics['mae']:.2f}\n")
        f.write(f"RMSE: {test_metrics['rmse']:.2f}\n")
        f.write(f"R²: {test_metrics['r2']:.4f}\n")


if __name__ == '__main__':
    main()
