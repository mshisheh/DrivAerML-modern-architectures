#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for AB-UPT on DrivAerNet++ surface field prediction.

This script trains AB-UPT to predict surface pressure and wall shear stress fields.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import logging

# Import modules
from data_loader import get_dataloaders, PRESSURE_MEAN, PRESSURE_STD
from collator import ABUPTSurfaceFieldCollator
from model import ABUPTSurfaceFieldPredictor, ABUPTSurfaceFieldPredictorLite

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Configuration
config = {
    'exp_name': 'ABUPT_SurfaceFields_DrivAerNet',
    'cuda': True,
    'seed': 42,
    
    # Model architecture
    'model_size': 'base',  # 'lite', 'base', 'large'
    'dim': 256,
    'geometry_depth': 2,
    'num_heads': 8,
    'blocks': 'pscs',
    'num_surface_blocks': 6,
    'num_volume_blocks': 2,
    'radius': 0.25,
    'predict_wss': False,  # Set to True if WSS data is available
    
    # Collator settings
    'num_geometry_points': 8192,
    'num_surface_anchors': 4096,
    'num_geometry_supernodes': 512,
    'use_query_positions': False,
    
    # Training
    'batch_size': 2,
    'epochs': 150,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'optimizer': 'adamw',
    'scheduler': 'cosine',
    
    # Data
    'dataset_path': '../PressureVTK',
    'subset_dir': '../train_val_test_splits',
    'cache_dir': './cached_data_abupt',
    'num_points': 10000,
    'num_workers': 4,
    
    # Checkpointing
    'checkpoint_dir': './experiments',
    'save_every': 10,
    'log_every': 10,
}


def setup_seed(seed: int):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_metrics(pred, target):
    """
    Compute evaluation metrics for field prediction.
    
    Args:
        pred: Predictions (B, N, D)
        target: Ground truth (B, N, D)
        
    Returns:
        Dictionary of metrics
    """
    mse = F.mse_loss(pred, target)
    mae = F.l1_loss(pred, target)
    
    # Relative error
    relative_error = torch.abs(pred - target) / (torch.abs(target) + 1e-8)
    mean_relative_error = relative_error.mean()
    
    # R2 score
    target_mean = target.mean()
    ss_tot = ((target - target_mean) ** 2).sum()
    ss_res = ((target - pred) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    
    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'relative_error': mean_relative_error.item(),
        'r2': r2.item(),
    }


def initialize_model(config: dict, device: torch.device) -> nn.Module:
    """Initialize AB-UPT model."""
    model_size = config['model_size']
    
    if model_size == 'lite':
        model = ABUPTSurfaceFieldPredictorLite(
            predict_wss=config['predict_wss']
        )
    elif model_size == 'base':
        model = ABUPTSurfaceFieldPredictor(
            dim=config['dim'],
            geometry_depth=config['geometry_depth'],
            num_heads=config['num_heads'],
            blocks=config['blocks'],
            num_surface_blocks=config['num_surface_blocks'],
            num_volume_blocks=config['num_volume_blocks'],
            radius=config['radius'],
            predict_wss=config['predict_wss'],
        )
    elif model_size == 'large':
        model = ABUPTSurfaceFieldPredictor(
            dim=512,
            geometry_depth=3,
            num_heads=16,
            blocks='pscscs',
            num_surface_blocks=8,
            num_volume_blocks=2,
            radius=0.25,
            predict_wss=config['predict_wss'],
        )
    else:
        raise ValueError(f"Unknown model_size: {model_size}")
    
    model = model.to(device)
    
    # Multi-GPU support
    if config['cuda'] and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    return model


def create_collator_fn(config: dict):
    """Create collator function."""
    collator = ABUPTSurfaceFieldCollator(
        num_geometry_points=config['num_geometry_points'],
        num_surface_anchors=config['num_surface_anchors'],
        num_geometry_supernodes=config['num_geometry_supernodes'],
        use_query_positions=config['use_query_positions'],
        seed=config['seed'],
    )
    return collator


def train_epoch(model, train_loader, collator, optimizer, device, epoch, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_metrics = {
        'pressure_mse': 0.0,
        'pressure_mae': 0.0,
        'pressure_r2': 0.0,
    }
    if config['predict_wss']:
        total_metrics['wss_mse'] = 0.0
        total_metrics['wss_mae'] = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, raw_batch in enumerate(pbar):
        # Collate batch
        batch = collator(raw_batch)
        if batch is None:
            continue
        
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch)
        
        # Compute loss
        loss = 0.0
        
        # Pressure loss
        pred_pressure = outputs['surface_anchor_pressure']
        target_pressure = batch['surface_anchor_pressure'].unsqueeze(-1)
        pressure_loss = F.mse_loss(pred_pressure, target_pressure)
        loss += pressure_loss
        
        # WSS loss (if applicable)
        if config['predict_wss'] and 'surface_anchor_wallshearstress' in outputs:
            pred_wss = outputs['surface_anchor_wallshearstress']
            target_wss = batch['surface_anchor_wallshearstress']
            wss_loss = F.mse_loss(pred_wss, target_wss)
            loss += wss_loss
            total_metrics['wss_mse'] += wss_loss.item()
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        total_metrics['pressure_mse'] += pressure_loss.item()
        
        with torch.no_grad():
            pressure_metrics = compute_metrics(pred_pressure, target_pressure)
            total_metrics['pressure_mae'] += pressure_metrics['mae']
            total_metrics['pressure_r2'] += pressure_metrics['r2']
        
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    # Average metrics
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    return avg_loss, avg_metrics


@torch.no_grad()
def evaluate(model, data_loader, collator, device, config):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    total_metrics = {
        'pressure_mse': 0.0,
        'pressure_mae': 0.0,
        'pressure_r2': 0.0,
    }
    if config['predict_wss']:
        total_metrics['wss_mse'] = 0.0
        total_metrics['wss_mae'] = 0.0
    
    for raw_batch in tqdm(data_loader, desc="Evaluating"):
        batch = collator(raw_batch)
        if batch is None:
            continue
        
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = model(batch)
        
        # Compute loss
        loss = 0.0
        
        pred_pressure = outputs['surface_anchor_pressure']
        target_pressure = batch['surface_anchor_pressure'].unsqueeze(-1)
        pressure_loss = F.mse_loss(pred_pressure, target_pressure)
        loss += pressure_loss
        
        if config['predict_wss'] and 'surface_anchor_wallshearstress' in outputs:
            pred_wss = outputs['surface_anchor_wallshearstress']
            target_wss = batch['surface_anchor_wallshearstress']
            wss_loss = F.mse_loss(pred_wss, target_wss)
            loss += wss_loss
            total_metrics['wss_mse'] += wss_loss.item()
        
        total_loss += loss.item()
        total_metrics['pressure_mse'] += pressure_loss.item()
        
        pressure_metrics = compute_metrics(pred_pressure, target_pressure)
        total_metrics['pressure_mae'] += pressure_metrics['mae']
        total_metrics['pressure_r2'] += pressure_metrics['r2']
    
    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    return avg_loss, avg_metrics


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, filename):
    """Save checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'config': config,
    }
    
    exp_dir = os.path.join(config['checkpoint_dir'], config['exp_name'])
    os.makedirs(exp_dir, exist_ok=True)
    checkpoint_path = os.path.join(exp_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Saved checkpoint: {checkpoint_path}")


def main(config):
    """Main training function."""
    # Setup
    setup_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() and config['cuda'] else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create collator
    collator = create_collator_fn(config)
    
    # Create dataloaders (without collator, we'll collate manually)
    logging.info("\nLoading datasets...")
    from data_loader import SurfaceFieldDataset, create_subset
    from torch.utils.data import DataLoader
    
    full_dataset = SurfaceFieldDataset(
        root_dir=config['dataset_path'],
        num_points=config['num_points'],
        preprocess=True,
        cache_dir=config['cache_dir'],
        load_wss=config['predict_wss'],
    )
    
    train_dataset = create_subset(full_dataset, os.path.join(config['subset_dir'], 'train_design_ids.txt'))
    val_dataset = create_subset(full_dataset, os.path.join(config['subset_dir'], 'val_design_ids.txt'))
    test_dataset = create_subset(full_dataset, os.path.join(config['subset_dir'], 'test_design_ids.txt'))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    
    # Initialize model
    logging.info("\nInitializing model...")
    model = initialize_model(config, device)
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {num_params:,}")
    
    # Optimizer
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    # Scheduler
    if config['scheduler'] == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    elif config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    else:
        scheduler = None
    
    # Training loop
    logging.info("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, config['epochs'] + 1):
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, collator, optimizer, device, epoch, config)
        logging.info(f"Epoch {epoch}/{config['epochs']}: train_loss = {train_loss:.6f}, "
                    f"pressure_r2 = {train_metrics['pressure_r2']:.4f}")
        
        # Validate
        val_loss, val_metrics = evaluate(model, val_loader, collator, device, config)
        logging.info(f"Epoch {epoch}/{config['epochs']}: val_loss = {val_loss:.6f}, "
                    f"pressure_r2 = {val_metrics['pressure_r2']:.4f}")
        
        # Scheduler step
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, 'best_model.pth')
        
        if epoch % config['save_every'] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, f'checkpoint_epoch_{epoch}.pth')
    
    # Final evaluation on test set
    logging.info("\nEvaluating on test set...")
    test_loss, test_metrics = evaluate(model, test_loader, collator, device, config)
    logging.info(f"Test: loss = {test_loss:.6f}, pressure_r2 = {test_metrics['pressure_r2']:.4f}")
    
    logging.info("\nTraining completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AB-UPT for surface field prediction')
    parser.add_argument('--model_size', type=str, choices=['lite', 'base', 'large'], help='Model size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--dataset_path', type=str, help='Path to VTK dataset')
    parser.add_argument('--predict_wss', action='store_true', help='Predict wall shear stress')
    
    args = parser.parse_args()
    
    # Update config
    if args.model_size:
        config['model_size'] = args.model_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['lr'] = args.lr
    if args.dataset_path:
        config['dataset_path'] = args.dataset_path
    if args.predict_wss:
        config['predict_wss'] = True
    
    main(config)
