"""
Main training and testing script for Hybrid SciML Potential Field Prediction.
Supports local and HPC (PBSPro) environments with distributed training.
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime
import shutil

from src.models.hybrid import HybridWrapper
from src.utils.data_utils import create_dataloaders
from src.utils.hpc_utils import setup_distributed, cleanup_distributed, get_device, is_master_process
from src.utils.physics import compute_total_loss


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric values are properly typed
    def convert_numeric(obj):
        """Recursively convert numeric strings to floats/ints."""
        if isinstance(obj, dict):
            return {k: convert_numeric(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numeric(item) for item in obj]
        elif isinstance(obj, str):
            # Try to convert to float or int
            try:
                if '.' in obj or 'e' in obj.lower():
                    return float(obj)
                else:
                    return int(obj)
            except ValueError:
                return obj
        else:
            return obj
    
    config = convert_numeric(config)
    
    return config


def setup_model(config: dict, device: torch.device) -> nn.Module:
    """Initialize model from configuration."""
    model_config = config['model']
    
    # Prepare analytical config
    analytical_config = {
        'current_I': config['physics']['analytical']['current_I'],
        'epsilon_factor': config['physics']['analytical']['epsilon_factor'],
        'coord_range': config['grid']['coord_range']
    }
    
    # Prepare backbone config
    backbone_type = model_config['backbone']
    if backbone_type == 'fno':
        backbone_config = model_config['fno']
    else:
        backbone_config = model_config['unet']
    
    # Create model
    model = HybridWrapper(
        backbone=backbone_type,
        geometry_encoder=model_config.get('geometry_encoder'),
        backbone_config=backbone_config,
        analytical_config=analytical_config
    )
    
    model = model.to(device)
    
    return model


def setup_optimizer(model: nn.Module, config: dict) -> tuple:
    """Setup optimizer and scheduler."""
    train_config = config['training']
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config.get('weight_decay', 1e-5)
    )
    
    # Setup scheduler
    scheduler_type = train_config.get('scheduler', 'cosine')
    scheduler_params = train_config.get('scheduler_params', {})
    
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get('T_max', 100),
            eta_min=scheduler_params.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params.get('step_size', 30),
            gamma=scheduler_params.get('gamma', 0.1)
        )
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_params.get('factor', 0.5),
            patience=scheduler_params.get('patience', 10)
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    loss: float,
    config: dict,
    checkpoint_dir: str,
    is_best: bool = False
):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    torch.save(checkpoint, latest_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pt')
        torch.save(checkpoint, best_path)
    
    # Save periodic checkpoint
    periodic_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, periodic_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    checkpoint_path: str,
    device: torch.device
) -> tuple:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint.get('loss', float('inf'))
    
    return start_epoch, best_loss


def visualize_slices(
    pred: torch.Tensor,
    target: torch.Tensor,
    correction: torch.Tensor,
    epoch,  # Can be int or str for test mode
    batch_idx: int,
    vis_dir: str,
    vis_slices: list,
    is_train: bool = True
):
    """
    Generate and save 2D slice visualizations.
    
    Args:
        pred: Predicted potential (B, 1, D, H, W)
        target: Ground truth potential (B, 1, D, H, W)
        correction: Neural correction (B, 1, D, H, W)
        epoch: Current epoch
        batch_idx: Batch index
        vis_dir: Visualization directory
        vis_slices: List of slice types to visualize
        is_train: Whether this is training or validation
    """
    os.makedirs(vis_dir, exist_ok=True)
    
    # Take first sample from batch
    pred = pred[0, 0].detach().cpu().numpy()
    target = target[0, 0].detach().cpu().numpy()
    correction = correction[0, 0].detach().cpu().numpy()
    error = np.abs(pred - target)
    
    D, H, W = pred.shape
    
    # Get center slices
    d_center = D // 2
    h_center = H // 2
    w_center = W // 2
    
    split = 'train' if is_train else 'val'
    
    saved_paths = []
    for slice_type in vis_slices:
        if slice_type == 'axial':
            # XY plane at center Z
            pred_slice = pred[:, :, w_center]
            target_slice = target[:, :, w_center]
            correction_slice = correction[:, :, w_center]
            error_slice = error[:, :, w_center]
            plane = 'XY'
        elif slice_type == 'sagittal':
            # YZ plane at center X
            pred_slice = pred[d_center, :, :]
            target_slice = target[d_center, :, :]
            correction_slice = correction[d_center, :, :]
            error_slice = error[d_center, :, :]
            plane = 'YZ'
        elif slice_type == 'coronal':
            # XZ plane at center Y
            pred_slice = pred[:, h_center, :]
            target_slice = target[:, h_center, :]
            correction_slice = correction[:, h_center, :]
            error_slice = error[:, h_center, :]
            plane = 'XZ'
        else:
            continue
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        vmin = min(pred_slice.min(), target_slice.min())
        vmax = max(pred_slice.max(), target_slice.max())
        
        im1 = axes[0].imshow(target_slice, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        axes[0].set_title(f'Ground Truth ({plane})')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(pred_slice, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        axes[1].set_title(f'Predicted Total ({plane})')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(correction_slice, cmap='RdBu', aspect='auto')
        axes[2].set_title(f'Neural Correction ({plane})')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])
        
        im4 = axes[3].imshow(error_slice, cmap='hot', aspect='auto')
        axes[3].set_title(f'Error Map ({plane})')
        axes[3].axis('off')
        plt.colorbar(im4, ax=axes[3])
        
        plt.tight_layout()
        
        filename = f'{split}_epoch_{epoch}_batch_{batch_idx}_{slice_type}.png'
        filepath = os.path.join(vis_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        saved_paths.append(filepath)
    
    return saved_paths  # Return list of paths for potential TensorBoard logging


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: dict,
    epoch: int,
    rank: int,
    scaler: GradScaler = None,
    writer: SummaryWriter = None
) -> dict:
    """Train for one epoch."""
    model.train()
    total_losses = {'total': 0.0, 'mse': 0.0, 'pde': 0.0, 'charge': 0.0}
    num_batches = 0
    
    use_amp = config['training'].get('use_amp', False)
    loss_weights = config['training']['loss_weights']
    grid_resolution = tuple(config['grid']['train_resolution'])
    vis_freq = config['logging'].get('vis_freq', 10)
    vis_dir = config['logging'].get('vis_dir', 'visualizations')
    vis_slices = config['logging'].get('vis_slices', ['axial', 'sagittal'])
    
    for batch_idx, batch in enumerate(train_loader):
        input_tensor = batch['input'].to(device)  # (B, 5, D, H, W)
        target = batch['potential'].to(device)  # (B, 1, D, H, W)
        conductivity = batch['conductivity'].to(device)
        source = batch['source'].to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            # Forward pass
            pred, phi_analytical, correction = model(input_tensor, return_components=True)
            
            # Compute losses
            losses = compute_total_loss(
                pred=pred,
                target=target,
                conductivity=conductivity,
                source=source,
                grid_resolution=grid_resolution,
                loss_weights=loss_weights,
                config=config
            )
        
        # Backward pass
        if use_amp and scaler is not None:
            scaler.scale(losses['total']).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses['total'].backward()
            optimizer.step()
        
        # Accumulate losses
        for key in total_losses:
            if key in losses:
                total_losses[key] += losses[key].item()
        
        num_batches += 1
        
        # Visualization (only on master process, every vis_freq epochs, first batch only)
        if is_master_process(rank) and epoch % vis_freq == 0 and batch_idx == 0:
            # Use no_grad for visualization to save memory
            with torch.no_grad():
                vis_paths = visualize_slices(
                    pred, target, correction, epoch, batch_idx,
                    vis_dir, vis_slices, is_train=True
                )
                # Log images to TensorBoard
                if writer is not None and vis_paths:
                    import torchvision.transforms as transforms
                    from PIL import Image
                    
                    # Log first slice type (usually axial)
                    if len(vis_paths) > 0:
                        img = Image.open(vis_paths[0])
                        img_tensor = transforms.ToTensor()(img)
                        writer.add_image('Train/Visualization', img_tensor, epoch, dataformats='CHW')
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    # Logging (only on master process)
    if is_master_process(rank) and writer is not None:
        for key, value in avg_losses.items():
            writer.add_scalar(f'Train/{key}_loss', value, epoch)
    
    return avg_losses


def validate(
    model: nn.Module,
    val_loader,
    device: torch.device,
    config: dict,
    epoch: int,
    rank: int,
    writer: SummaryWriter = None
) -> dict:
    """Validate model."""
    model.eval()
    total_losses = {'total': 0.0, 'mse': 0.0, 'pde': 0.0, 'charge': 0.0}
    num_batches = 0
    
    loss_weights = config['training']['loss_weights']
    grid_resolution = tuple(config['grid']['train_resolution'])
    vis_freq = config['logging'].get('vis_freq', 10)
    vis_dir = config['logging'].get('vis_dir', 'visualizations')
    vis_slices = config['logging'].get('vis_slices', ['axial', 'sagittal'])
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_tensor = batch['input'].to(device)
            target = batch['potential'].to(device)
            conductivity = batch['conductivity'].to(device)
            source = batch['source'].to(device)
            
            # Forward pass
            pred, phi_analytical, correction = model(input_tensor, return_components=True)
            
            # Compute losses
            losses = compute_total_loss(
                pred=pred,
                target=target,
                conductivity=conductivity,
                source=source,
                grid_resolution=grid_resolution,
                loss_weights=loss_weights,
                config=config
            )
            
            # Accumulate losses
            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key].item()
            
            num_batches += 1
            
            # Visualization (only on master process, every vis_freq epochs)
            if is_master_process(rank) and epoch % vis_freq == 0 and batch_idx == 0:
                vis_paths = visualize_slices(
                    pred, target, correction, epoch, batch_idx,
                    vis_dir, vis_slices, is_train=False
                )
                # Log images to TensorBoard
                if writer is not None and vis_paths:
                    import torchvision.transforms as transforms
                    from PIL import Image
                    
                    # Log first slice type (usually axial)
                    if len(vis_paths) > 0:
                        img = Image.open(vis_paths[0])
                        img_tensor = transforms.ToTensor()(img)
                        writer.add_image('Val/Visualization', img_tensor, epoch, dataformats='CHW')
    
    # Average losses
    if num_batches > 0:
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    else:
        # Return zero losses if no validation data
        avg_losses = {k: 0.0 for k in total_losses.keys()}
    
    # Logging (only on master process)
    if is_master_process(rank) and writer is not None:
        for key, value in avg_losses.items():
            writer.add_scalar(f'Val/{key}_loss', value, epoch)
    
    return avg_losses


def setup_experiment_dir(config: dict, rank: int = None) -> dict:
    """
    Setup experiment directory structure and save config.
    Returns updated config with experiment paths.
    """
    if not is_master_process(rank):
        return config
    
    # Create experiments directory
    experiments_dir = 'experiments'
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Generate experiment ID with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = config['logging'].get('experiment_name', 'hybrid_potential_prediction')
    experiment_id = f"{experiment_name}_{timestamp}"
    experiment_dir = os.path.join(experiments_dir, experiment_id)
    
    # Create experiment directory structure
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'configs'), exist_ok=True)
    
    # Save config to experiment directory
    config_path = os.path.join(experiment_dir, 'configs', 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # Update config paths to point to experiment directory
    config['experiment'] = {
        'id': experiment_id,
        'dir': experiment_dir,
        'timestamp': timestamp
    }
    config['training']['checkpoint_dir'] = os.path.join(experiment_dir, 'checkpoints')
    config['logging']['log_dir'] = os.path.join(experiment_dir, 'logs')
    config['logging']['vis_dir'] = os.path.join(experiment_dir, 'visualizations')
    
    print(f"Experiment directory: {experiment_dir}")
    
    return config


def train(
    config: dict,
    rank: int = None,
    world_size: int = None,
    local_rank: int = None
):
    """Main training loop."""
    # Setup device
    device = get_device(rank, local_rank)
    
    # Setup experiment directory (only on master)
    if is_master_process(rank):
        config = setup_experiment_dir(config, rank)
    
    # Setup logging (only on master)
    writer = None
    if is_master_process(rank):
        log_dir = config['logging'].get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    
    # Setup model
    model = setup_model(config, device)
    
    # Wrap in DDP if distributed
    if rank is not None:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if local_rank is not None else None,
            output_device=device
        )
        model_module = model.module
    else:
        model_module = model
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer(model_module, config)
    
    # Setup mixed precision
    scaler = None
    if config['training'].get('use_amp', False):
        scaler = GradScaler()
    
    # Setup dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config['data']['data_dir'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        resolution=tuple(config['grid']['train_resolution']),
        log_conductivity=config['grid']['log_conductivity'],
        conductivity_epsilon=config['grid']['conductivity_epsilon'],
        coord_range=tuple(config['grid']['coord_range']),
        use_distributed=(rank is not None),
        rank=rank or 0,
        world_size=world_size or 1,
        use_scratch_dir=config['data'].get('use_scratch_dir', False),
        scratch_dir=config['data'].get('scratch_dir', '/scratch')
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = config['training'].get('resume_from')
    checkpoint_dir = config['training']['checkpoint_dir']
    
    # If resume_from is not specified but we're in an experiment dir, try latest checkpoint
    if not checkpoint_path and 'experiment' in config:
        latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        if os.path.exists(latest_checkpoint):
            checkpoint_path = latest_checkpoint
            if is_master_process(rank):
                print(f"Found latest checkpoint in experiment directory: {checkpoint_path}")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        if is_master_process(rank):
            print(f"Loading checkpoint from {checkpoint_path}")
        start_epoch, best_val_loss = load_checkpoint(
            model_module, optimizer, scheduler, checkpoint_path, device
        )
    
    # Training loop
    for epoch in range(start_epoch, config['training']['epochs']):
        if rank is not None:
            train_loader.sampler.set_epoch(epoch)
        
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, device, config, epoch, rank, scaler, writer
        )
        
        # Validate
        val_losses = validate(model, val_loader, device, config, epoch, rank, writer)
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_losses['total'])
            else:
                scheduler.step()
        
        # Checkpointing (only on master)
        if is_master_process(rank):
            is_best = val_losses['total'] < best_val_loss
            if is_best:
                best_val_loss = val_losses['total']
            
            # Save checkpoint
            if epoch % config['training'].get('checkpoint_freq', 10) == 0 or is_best:
                save_checkpoint(
                    model_module, optimizer, scheduler, epoch, val_losses['total'],
                    config, checkpoint_dir, is_best
                )
            
            # Print progress
            print(f"Epoch {epoch}/{config['training']['epochs']}")
            print(f"  Train Loss: {train_losses['total']:.6f}")
            print(f"  Val Loss: {val_losses['total']:.6f}")
            print(f"  Val MSE: {val_losses.get('mse', 0):.6f}")
            print(f"  Val PDE: {val_losses.get('pde', 0):.6f}")
            print(f"  Val Charge: {val_losses.get('charge', 0):.6f}")
    
    if writer is not None:
        writer.close()
    
    if is_master_process(rank):
        print("Training completed!")


def test(
    config: dict,
    checkpoint_path: str,
    rank: int = None,
    local_rank: int = None
):
    """Test model on test set and optionally on different resolutions."""
    device = get_device(rank, local_rank)
    
    # Setup experiment directory for test run (only on master)
    if is_master_process(rank):
        # If checkpoint has experiment info, use it; otherwise create new
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'config' in checkpoint and 'experiment' in checkpoint['config']:
            # Use the original experiment directory
            exp_config = checkpoint['config']
            config['experiment'] = exp_config.get('experiment', {})
            if 'dir' in config['experiment']:
                config['logging']['vis_dir'] = os.path.join(config['experiment']['dir'], 'visualizations', 'test')
        else:
            # Create new test experiment directory
            config = setup_experiment_dir(config, rank)
            config['logging']['vis_dir'] = os.path.join(config['experiment']['dir'], 'visualizations', 'test')
    
    # Setup model
    model = setup_model(config, device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test resolutions
    test_resolutions = config['evaluation'].get('test_resolutions', [config['grid']['test_resolution']])
    
    for resolution in test_resolutions:
        if is_master_process(rank):
            print(f"\nTesting on resolution: {resolution}")
        
        # Create test dataloader
        _, _, test_loader = create_dataloaders(
            data_dir=config['data']['data_dir'],
            train_split=config['data']['train_split'],
            val_split=config['data']['val_split'],
            test_split=config['data']['test_split'],
            batch_size=1,  # Test with batch size 1 for visualization
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
            resolution=tuple(resolution),
            log_conductivity=config['grid']['log_conductivity'],
            conductivity_epsilon=config['grid']['conductivity_epsilon'],
            coord_range=tuple(config['grid']['coord_range']),
            use_distributed=False,
            rank=0,
            world_size=1
        )
        
        # Evaluate
        total_losses = {'total': 0.0, 'mse': 0.0, 'pde': 0.0, 'charge': 0.0}
        num_batches = 0
        
        vis_dir = os.path.join(config['logging'].get('vis_dir', 'visualizations'), 'test')
        vis_slices = config['logging'].get('vis_slices', ['axial', 'sagittal'])
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                input_tensor = batch['input'].to(device)
                target = batch['potential'].to(device)
                conductivity = batch['conductivity'].to(device)
                source = batch['source'].to(device)
                
                # Forward pass
                pred, phi_analytical, correction = model(input_tensor, return_components=True)
                
                # Compute losses
                losses = compute_total_loss(
                    pred=pred,
                    target=target,
                    conductivity=conductivity,
                    source=source,
                    grid_resolution=tuple(resolution),
                    loss_weights=config['training']['loss_weights'],
                    config=config
                )
                
                # Accumulate
                for key in total_losses:
                    if key in losses:
                        total_losses[key] += losses[key].item()
                
                num_batches += 1
                
                # Visualize first few samples
                if is_master_process(rank) and batch_idx < 5:
                    # Create resolution-specific directory
                    res_dir = os.path.join(vis_dir, f"res_{resolution[0]}x{resolution[1]}x{resolution[2]}")
                    visualize_slices(
                        pred, target, correction, 
                        epoch=f"test_{resolution[0]}x{resolution[1]}x{resolution[2]}",
                        batch_idx=batch_idx, 
                        vis_dir=res_dir, 
                        vis_slices=vis_slices, 
                        is_train=False
                    )
        
        # Print results
        if is_master_process(rank):
            avg_losses = {k: v / num_batches for k, v in total_losses.items()}
            print(f"Resolution {resolution}:")
            print(f"  Test Loss: {avg_losses['total']:.6f}")
            print(f"  Test MSE: {avg_losses.get('mse', 0):.6f}")
            print(f"  Test PDE: {avg_losses.get('pde', 0):.6f}")
            print(f"  Test Charge: {avg_losses.get('charge', 0):.6f}")


def main():
    parser = argparse.ArgumentParser(description='Hybrid SciML Potential Field Prediction')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Mode: train or test')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path for testing')
    parser.add_argument('--experiment-dir', type=str, default=None, help='Use existing experiment directory (for resuming)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # If resuming from experiment directory, load its config
    if args.experiment_dir and os.path.exists(args.experiment_dir):
        exp_config_path = os.path.join(args.experiment_dir, 'configs', 'config.yaml')
        if os.path.exists(exp_config_path):
            config = load_config(exp_config_path)
            print(f"Resuming experiment from: {args.experiment_dir}")
    
    # Setup distributed training if needed
    rank, world_size, local_rank = None, None, None
    if config['hpc'].get('detect_automatically', True):
        rank, world_size, local_rank = setup_distributed(
            backend=config['hpc'].get('backend', 'nccl'),
            master_addr=config['hpc'].get('master_addr'),
            master_port=config['hpc'].get('master_port', '29500')
        )
    
    try:
        if args.mode == 'train':
            train(config, rank, world_size, local_rank)
        else:
            if args.checkpoint is None:
                checkpoint_path = os.path.join(config['training']['checkpoint_dir'], 'best_checkpoint.pt')
            else:
                checkpoint_path = args.checkpoint
            test(config, checkpoint_path, rank, local_rank)
    finally:
        if rank is not None:
            cleanup_distributed()


if __name__ == '__main__':
    main()
