#!/usr/bin/env python3
"""Main training script for 3D potential field prediction."""

import argparse
import os
import sys
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_dataloaders
from src.models.wrapper import build_model, get_device, count_parameters
from src.utils.masking import CombinedLoss, create_combined_mask
from src.utils.metrics import compute_all_metrics, gradient_norm, parameter_norm
from src.utils.visualization import save_validation_figure


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_experiment_dir(config: dict) -> str:
    """Create experiment directory for checkpoints and logs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config['logging'].get('experiment_name', 'experiment')
    backbone = config['model'].get('backbone', 'fno')

    exp_dir = Path("experiments") / f"{experiment_name}_{backbone}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "visualizations").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)

    # Save config to experiment directory
    with open(exp_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return str(exp_dir)


def train_epoch(
    model: torch.nn.Module,
    train_loader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    config: dict,
) -> dict:
    """Train for one epoch.

    Returns:
        Dictionary with training metrics
    """
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_grad = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        sigma = batch['sigma'].to(device)
        source = batch['source'].to(device)
        coords = batch['coords'].to(device)
        spacing = batch['spacing'].to(device)
        target = batch['target'].to(device)
        source_point = batch.get('source_point', None)
        if source_point is not None:
            source_point = source_point.to(device)

        # Forward pass
        optimizer.zero_grad()
        pred = model(sigma, source, coords, spacing)

        # Compute loss
        loss, loss_dict = criterion(pred, target, sigma, source, spacing, source_point)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate metrics
        total_loss += loss_dict['loss']
        total_mse += loss_dict['mse_loss']
        total_grad += loss_dict['grad_loss']
        num_batches += 1

        # Log batch metrics
        global_step = epoch * len(train_loader) + batch_idx
        if batch_idx % 10 == 0:
            writer.add_scalar('train/batch_loss', loss_dict['loss'], global_step)

    # Average metrics
    avg_metrics = {
        'loss': total_loss / num_batches,
        'mse_loss': total_mse / num_batches,
        'grad_loss': total_grad / num_batches,
    }

    return avg_metrics


def validate(
    model: torch.nn.Module,
    val_loader,
    criterion: CombinedLoss,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    config: dict,
    exp_dir: str,
) -> dict:
    """Validate the model.

    Returns:
        Dictionary with validation metrics
    """
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_grad = 0.0
    num_batches = 0
    all_metrics = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move data to device
            sigma = batch['sigma'].to(device)
            source = batch['source'].to(device)
            coords = batch['coords'].to(device)
            spacing = batch['spacing'].to(device)
            target = batch['target'].to(device)
            source_point = batch.get('source_point', None)
            if source_point is not None:
                source_point = source_point.to(device)

            # Forward pass
            pred = model(sigma, source, coords, spacing)

            # Compute loss
            loss, loss_dict = criterion(pred, target, sigma, source, spacing, source_point)

            # Compute metrics
            mask = create_combined_mask(sigma, source)
            metrics = compute_all_metrics(pred, target, mask)
            all_metrics.append(metrics)

            # Accumulate loss
            total_loss += loss_dict['loss']
            total_mse += loss_dict['mse_loss']
            total_grad += loss_dict['grad_loss']
            num_batches += 1

            # Save visualization for first batch on visualization epochs
            vis_freq = config['logging'].get('vis_freq', 10)
            if batch_idx == 0 and (epoch + 1) % vis_freq == 0:
                vis_dir = os.path.join(exp_dir, "visualizations")
                slice_types = config['logging'].get('vis_slices', ['axial', 'sagittal'])
                save_validation_figure(
                    pred[0], target[0], mask[0],
                    epoch=epoch,
                    save_dir=vis_dir,
                    slice_types=slice_types,
                )

    # Average metrics
    avg_loss = {
        'loss': total_loss / num_batches,
        'mse_loss': total_mse / num_batches,
        'grad_loss': total_grad / num_batches,
    }

    # Average evaluation metrics
    avg_eval_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }

    return {**avg_loss, **avg_eval_metrics}


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train 3D potential field prediction model")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set seed
    seed = args.seed
    set_seed(seed)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Create experiment directory
    exp_dir = create_experiment_dir(config)
    print(f"Experiment directory: {exp_dir}")

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(config, seed=seed)
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # Build model
    print("Building model...")
    model = build_model(config)
    model = model.to(device)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    print(f"Backbone: {config['model']['backbone']}")

    # Setup optimizer
    train_config = config['training']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
    )

    # Setup scheduler
    scheduler = None
    if train_config.get('scheduler') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config['scheduler_params'].get('T_max', train_config['epochs']),
            eta_min=train_config['scheduler_params'].get('eta_min', 1e-6),
        )

    # Setup loss function
    loss_weights = train_config.get('loss_weights', {})
    physics_config = config.get('physics', {})
    mse_config = physics_config.get('mse', {})

    criterion = CombinedLoss(
        mse_weight=loss_weights.get('mse', 1.0),
        grad_weight=loss_weights.get('pde', 0.1),
        singularity_radius=mse_config.get('singularity_mask_radius', 3),
        use_muscle_mask=True,
    )

    # Setup TensorBoard writer
    log_dir = os.path.join(exp_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)

    # Log configuration
    writer.add_text('config', yaml.dump(config), 0)
    writer.add_scalar('model/parameters', num_params, 0)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Training loop
    epochs = train_config['epochs']
    checkpoint_freq = train_config.get('checkpoint_freq', 10)

    print(f"\nStarting training for {epochs} epochs...")

    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, writer, config
        )
        train_losses.append(train_metrics['loss'])

        # Log training metrics
        for key, value in train_metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)

        # Log gradient and parameter norms periodically
        if epoch % 5 == 0:
            writer.add_scalar('train/gradient_norm', gradient_norm(model), epoch)
            writer.add_scalar('train/parameter_norm', parameter_norm(model), epoch)

        # Validate
        val_metrics = validate(
            model, val_loader, criterion,
            device, epoch, writer, config, exp_dir
        )
        val_losses.append(val_metrics['loss'])

        # Log validation metrics
        for key, value in val_metrics.items():
            writer.add_scalar(f'val/{key}', value, epoch)

        # Print metrics
        print(f"  Train Loss: {train_metrics['loss']:.6f} (MSE: {train_metrics['mse_loss']:.6f}, Grad: {train_metrics['grad_loss']:.6f})")
        print(f"  Val Loss: {val_metrics['loss']:.6f} | RMSE: {val_metrics['rmse']:.6f} | Rel L2: {val_metrics['relative_l2']:.6f}")

        # Update scheduler
        if scheduler:
            scheduler.step()

        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']

        if (epoch + 1) % checkpoint_freq == 0 or is_best:
            checkpoint_path = os.path.join(
                exp_dir, "checkpoints",
                f"checkpoint_epoch_{epoch + 1:04d}.pt" if not is_best else "best_model.pt"
            )
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'best_val_loss': best_val_loss,
                'config': config,
            }
            if scheduler:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = os.path.join(exp_dir, "checkpoints", "final_model.pt")
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")

    # Close writer
    writer.close()

    return exp_dir


if __name__ == "__main__":
    main()
