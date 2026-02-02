#!/usr/bin/env python3
"""Main training script for 3D potential field prediction.

Features:
- Early stopping with configurable patience
- 3D surface visualizations
- Region-wise error metrics and diagnostic metrics (scale, smoothness)
- Multiple loss types (MSE, normalized, log-cosh)
- Optional PDE residual loss and spectral smoothing regularization
- Configurable singularity and muscle masking
- Loss curve plotting with component breakdown
- Experiment tracking
"""

import argparse
import json
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

from src.data.loader import get_dataloaders, inverse_log_transform
from src.models.wrapper import build_model, get_device, count_parameters
from src.utils.masking import CombinedLoss, create_combined_mask
from src.utils.metrics import (
    compute_all_metrics, compute_all_metrics_extended,
    gradient_norm, parameter_norm
)
from src.utils.visualization import (
    save_validation_figure, plot_loss_curves,
    create_comprehensive_visualization
)
from scripts.visualize_fiber_potential import create_fiber_visualization
from src.utils.training import EarlyStopping, ExperimentTracker


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


def create_experiment_dir(config: dict, experiment_name: str = None) -> str:
    """Create experiment directory for checkpoints and logs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name is None:
        experiment_name = config['logging'].get('experiment_name', 'experiment')
    backbone = config['model'].get('backbone', 'fno')

    exp_dir = Path("experiments") / f"{experiment_name}_{backbone}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "visualizations").mkdir(exist_ok=True)
    (exp_dir / "visualizations" / "train").mkdir(exist_ok=True)
    (exp_dir / "visualizations" / "val").mkdir(exist_ok=True)
    (exp_dir / "visualizations" / "test").mkdir(exist_ok=True)
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
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_grad = 0.0
    total_pde = 0.0
    total_spectral = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        sigma = batch['sigma'].to(device)
        source = batch['source'].to(device)
        coords = batch['coords'].to(device)
        spacing = batch['spacing'].to(device)
        target = batch['target'].to(device)
        source_point = batch.get('source_point', None)
        if source_point is not None:
            source_point = source_point.to(device)

        # Handle spacing channels if present
        if 'spacing_channels' in batch:
            spacing_channels = batch['spacing_channels'].to(device)
            coords = torch.cat([coords, spacing_channels], dim=1)

        # Handle analytical solution if present
        analytical = batch.get('analytical', None)
        if analytical is not None:
            analytical = analytical.to(device)

        # Handle distance field if present
        distance_field = batch.get('distance_field', None)
        if distance_field is not None:
            distance_field = distance_field.to(device)

        # Get mask for geometry attention backbone
        mask = batch.get('mask', None)
        if mask is not None:
            mask = mask.to(device)

        optimizer.zero_grad()
        pred = model(sigma, source, coords, spacing, analytical=analytical, distance_field=distance_field, mask=mask)

        loss, loss_dict = criterion(pred, target, sigma, source, spacing, source_point)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_dict['loss']
        total_mse += loss_dict['mse_loss']
        total_grad += loss_dict['grad_loss']
        total_pde += loss_dict.get('pde_loss', 0.0)
        total_spectral += loss_dict.get('spectral_loss', 0.0)
        num_batches += 1

        global_step = epoch * len(train_loader) + batch_idx
        if batch_idx % 10 == 0:
            writer.add_scalar('train/batch_loss', loss_dict['loss'], global_step)

    avg_metrics = {
        'loss': total_loss / num_batches,
        'mse_loss': total_mse / num_batches,
        'grad_loss': total_grad / num_batches,
    }

    # Include optional loss components if non-zero
    if total_pde > 0:
        avg_metrics['pde_loss'] = total_pde / num_batches
    if total_spectral > 0:
        avg_metrics['spectral_loss'] = total_spectral / num_batches

    return avg_metrics


def validate(
    model: torch.nn.Module,
    data_loader,
    criterion: CombinedLoss,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    config: dict,
    exp_dir: str,
    split_name: str = "val",
    save_visualizations: bool = False,
) -> dict:
    """Validate/test the model with extended metrics."""
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_grad = 0.0
    total_pde = 0.0
    total_spectral = 0.0
    num_batches = 0
    all_metrics = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            sigma = batch['sigma'].to(device)
            source = batch['source'].to(device)
            coords = batch['coords'].to(device)
            spacing = batch['spacing'].to(device)
            target = batch['target'].to(device)
            source_point = batch.get('source_point', None)
            if source_point is not None:
                source_point = source_point.to(device)

            # Handle spacing channels if present
            if 'spacing_channels' in batch:
                spacing_channels = batch['spacing_channels'].to(device)
                coords = torch.cat([coords, spacing_channels], dim=1)

            # Handle analytical solution if present
            analytical = batch.get('analytical', None)
            if analytical is not None:
                analytical = analytical.to(device)

            # Handle distance field if present
            distance_field = batch.get('distance_field', None)
            if distance_field is not None:
                distance_field = distance_field.to(device)

            # Get mask for geometry attention backbone
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)

            pred = model(sigma, source, coords, spacing, analytical=analytical, distance_field=distance_field, mask=mask)

            loss, loss_dict = criterion(pred, target, sigma, source, spacing, source_point)

            # Apply inverse log transform if data was log-transformed
            log_transformed = batch.get('log_transformed', torch.tensor(False))
            if log_transformed.any():
                pred_for_metrics = inverse_log_transform(pred)
                target_for_metrics = inverse_log_transform(target)
            else:
                pred_for_metrics = pred
                target_for_metrics = target

            # Compute extended metrics including region-wise and diagnostics
            metrics = compute_all_metrics_extended(pred_for_metrics, target_for_metrics, sigma, source, spacing)
            all_metrics.append(metrics)

            total_loss += loss_dict['loss']
            total_mse += loss_dict['mse_loss']
            total_grad += loss_dict['grad_loss']
            total_pde += loss_dict.get('pde_loss', 0.0)
            total_spectral += loss_dict.get('spectral_loss', 0.0)
            num_batches += 1

            # Save comprehensive visualizations (use inverse-transformed values)
            if save_visualizations and batch_idx < 3:  # Save for first 3 samples
                vis_dir = os.path.join(exp_dir, "visualizations", split_name)
                mask = create_combined_mask(sigma, source)
                save_validation_figure(
                    pred_for_metrics[0], target_for_metrics[0],
                    mask=mask[0],
                    source=source[0],
                    epoch=epoch,
                    save_dir=vis_dir,
                    sample_idx=batch_idx,
                    comprehensive=True,
                )

    avg_loss = {
        'loss': total_loss / num_batches,
        'mse_loss': total_mse / num_batches,
        'grad_loss': total_grad / num_batches,
    }

    # Include optional loss components if non-zero
    if total_pde > 0:
        avg_loss['pde_loss'] = total_pde / num_batches
    if total_spectral > 0:
        avg_loss['spectral_loss'] = total_spectral / num_batches

    # Average all metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        avg_metrics[key] = np.mean(values) if values else float('nan')

    return {**avg_loss, **avg_metrics}


def evaluate_on_split(
    model: torch.nn.Module,
    data_loader,
    criterion: CombinedLoss,
    device: torch.device,
    exp_dir: str,
    split_name: str,
) -> dict:
    """Evaluate model on a data split and save visualizations."""
    model.eval()

    all_metrics = []
    total_loss = 0.0
    num_batches = 0

    vis_dir = os.path.join(exp_dir, "visualizations", split_name)
    os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            sigma = batch['sigma'].to(device)
            source = batch['source'].to(device)
            coords = batch['coords'].to(device)
            spacing = batch['spacing'].to(device)
            target = batch['target'].to(device)

            # Handle spacing channels if present
            if 'spacing_channels' in batch:
                spacing_channels = batch['spacing_channels'].to(device)
                coords = torch.cat([coords, spacing_channels], dim=1)

            # Handle analytical solution if present
            analytical = batch.get('analytical', None)
            if analytical is not None:
                analytical = analytical.to(device)

            # Handle distance field if present
            distance_field = batch.get('distance_field', None)
            if distance_field is not None:
                distance_field = distance_field.to(device)

            # Get mask for geometry attention backbone
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)

            pred = model(sigma, source, coords, spacing, analytical=analytical, distance_field=distance_field, mask=mask)

            loss, loss_dict = criterion(pred, target, sigma, source, spacing)

            # Apply inverse log transform if data was log-transformed
            # Metrics should be computed in original space for fair comparison
            log_transformed = batch.get('log_transformed', torch.tensor(False))
            if log_transformed.any():
                pred_for_metrics = inverse_log_transform(pred)
                target_for_metrics = inverse_log_transform(target)
            else:
                pred_for_metrics = pred
                target_for_metrics = target

            metrics = compute_all_metrics_extended(pred_for_metrics, target_for_metrics, sigma, source, spacing)
            all_metrics.append(metrics)
            total_loss += loss_dict['loss']
            num_batches += 1

            # Save visualization for all test samples (use inverse-transformed values)
            mask = create_combined_mask(sigma, source)

            # Comprehensive 3D visualization for first 5 samples
            if batch_idx < 5:
                create_comprehensive_visualization(
                    pred_for_metrics[0], target_for_metrics[0],
                    source=source[0],
                    mask=mask[0],
                    title_prefix=f"{split_name.capitalize()} Sample {batch_idx}",
                    save_dir=vis_dir,
                    sample_idx=batch_idx,
                )

            # Fiber potential visualization for ALL samples - DISABLED for faster runs
            # try:
            #     create_fiber_visualization(
            #         pred_for_metrics, target_for_metrics, source,
            #         sample_idx=batch_idx,
            #         resolution=f"{split_name}",
            #         output_dir=os.path.join(vis_dir, "fiber_potentials"),
            #         num_fibers=10,
            #         mask=mask,
            #     )
            # except Exception as e:
            #     print(f"Warning: Could not create fiber visualization for sample {batch_idx}: {e}")

    # Aggregate metrics
    avg_metrics = {'loss': total_loss / num_batches}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        avg_metrics[key] = np.mean(values) if values else float('nan')

    return avg_metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train 3D potential field prediction model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Custom experiment name")
    parser.add_argument("--description", type=str, default="",
                        help="Experiment description for tracking")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Disable early stopping")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Create experiment directory
    exp_dir = create_experiment_dir(config, args.experiment_name)
    exp_name = Path(exp_dir).name
    print(f"Experiment directory: {exp_dir}")

    # Create dataloaders with FIXED seed for reproducible splits across experiments
    print("Loading data...")
    DATA_SPLIT_SEED = 42  # Always use same split for fair comparison
    train_loader, val_loader, test_loader = get_dataloaders(config, seed=DATA_SPLIT_SEED)

    # Now set the model seed (can vary for ensemble training)
    set_seed(args.seed)
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples, Test: {len(test_loader.dataset)} samples")

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
    loss_config = config.get('loss', {})

    use_singularity_mask = config.get('experiment', {}).get('use_singularity_mask', True)
    experiment_config = config.get('experiment', {})
    singularity_mode = experiment_config.get('singularity_mask_mode', 'radius')
    singularity_percentile = experiment_config.get('singularity_percentile', 99.0)
    distance_weight_alpha = loss_config.get('distance_weight_alpha', 0.0)
    use_muscle_mask = experiment_config.get('use_muscle_mask', False)

    criterion = CombinedLoss(
        mse_weight=loss_weights.get('mse', 1.0),
        grad_weight=loss_weights.get('pde', 0.5),
        singularity_radius=mse_config.get('singularity_mask_radius', 3),
        pde_weight=loss_config.get('pde_weight', 0.0),
        tv_weight=loss_config.get('tv_weight', 0.01),
        gradient_matching_weight=loss_config.get('gradient_matching_weight', 0.0),
        laplacian_matching_weight=loss_config.get('laplacian_matching_weight', 0.0),
        use_singularity_mask=use_singularity_mask,
        singularity_mode=singularity_mode,
        singularity_percentile=singularity_percentile,
        distance_weight_alpha=distance_weight_alpha,
        use_muscle_mask=use_muscle_mask,
    )

    # Log loss configuration
    print(f"Loss config: tv_weight={loss_config.get('tv_weight', 0.01)}, "
          f"gradient_matching_weight={loss_config.get('gradient_matching_weight', 0.0)}, "
          f"use_singularity_mask={use_singularity_mask}, "
          f"use_muscle_mask={use_muscle_mask}, "
          f"distance_weight_alpha={distance_weight_alpha}")

    # Setup TensorBoard writer
    log_dir = os.path.join(exp_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)

    # Log configuration
    writer.add_text('config', yaml.dump(config), 0)
    writer.add_scalar('model/parameters', num_params, 0)

    # Setup early stopping
    early_stopping = None
    if not args.no_early_stopping:
        patience = config.get('training', {}).get('early_stopping_patience', 10)
        min_delta = config.get('training', {}).get('early_stopping_delta', 1e-6)
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        print(f"Early stopping enabled: patience={patience}, min_delta={min_delta}")

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

    # Training history
    history = {
        'train_loss': [], 'train_mse': [], 'train_grad': [],
        'val_loss': [], 'val_mse': [], 'val_grad': [],
        'val_relative_l2': [], 'val_rmse': [],
    }

    # Training loop
    epochs = train_config['epochs']
    checkpoint_freq = train_config.get('checkpoint_freq', 10)
    vis_freq = config['logging'].get('vis_freq', 10)

    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, writer, config
        )

        # Record training history
        history['train_loss'].append(train_metrics['loss'])
        history['train_mse'].append(train_metrics['mse_loss'])
        history['train_grad'].append(train_metrics['grad_loss'])

        # Log training metrics
        for key, value in train_metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)

        if epoch % 5 == 0:
            writer.add_scalar('train/gradient_norm', gradient_norm(model), epoch)
            writer.add_scalar('train/parameter_norm', parameter_norm(model), epoch)

        # Validate
        save_vis = (epoch + 1) % vis_freq == 0
        val_metrics = validate(
            model, val_loader, criterion,
            device, epoch, writer, config, exp_dir,
            split_name="val",
            save_visualizations=save_vis,
        )

        # Record validation history
        history['val_loss'].append(val_metrics['loss'])
        history['val_mse'].append(val_metrics['mse_loss'])
        history['val_grad'].append(val_metrics['grad_loss'])
        history['val_relative_l2'].append(val_metrics.get('relative_l2', float('nan')))
        history['val_rmse'].append(val_metrics.get('rmse', float('nan')))

        # Log validation metrics
        for key, value in val_metrics.items():
            if not np.isnan(value):
                writer.add_scalar(f'val/{key}', value, epoch)

        # Print metrics
        print(f"  Train Loss: {train_metrics['loss']:.6f} (MSE: {train_metrics['mse_loss']:.6f}, Grad: {train_metrics['grad_loss']:.6f})")
        print(f"  Val Loss: {val_metrics['loss']:.6f} | RMSE: {val_metrics.get('rmse', 0):.6f} | Rel L2: {val_metrics.get('relative_l2', 0):.6f}")

        # Print region-wise metrics if available
        if 'mse_muscle' in val_metrics:
            print(f"  Region-wise MSE - Muscle: {val_metrics['mse_muscle']:.6f}, Non-muscle: {val_metrics['mse_non_muscle']:.6f}")
            print(f"  Region-wise MSE - Near sing: {val_metrics['mse_near_singularity']:.6f}, Far sing: {val_metrics['mse_far_singularity']:.6f}")

        # Update scheduler
        if scheduler:
            scheduler.step()

        # Check early stopping
        if early_stopping is not None:
            if early_stopping(val_metrics['loss'], epoch):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

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
            if is_best:
                print(f"  Saved best model: {checkpoint_path}")

    # Save training history
    history_path = os.path.join(exp_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Plot and save loss curves
    loss_curve_path = os.path.join(exp_dir, "loss_curves.png")
    plot_loss_curves(
        history['train_loss'], history['val_loss'],
        train_mse=history['train_mse'], train_grad=history['train_grad'],
        val_mse=history['val_mse'], val_grad=history['val_grad'],
        save_path=loss_curve_path,
        title=f"Training Progress - {exp_name}",
    )
    print(f"\nLoss curves saved to {loss_curve_path}")

    # Load best model for final evaluation
    best_checkpoint_path = os.path.join(exp_dir, "checkpoints", "best_model.pt")
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model from epoch {checkpoint['epoch'] + 1}")

    # Evaluate on all splits
    print("\nFinal evaluation on all splits...")

    print("\n--- Train Set ---")
    train_final_metrics = evaluate_on_split(model, train_loader, criterion, device, exp_dir, "train")
    for key, value in train_final_metrics.items():
        if not np.isnan(value):
            print(f"  {key}: {value:.6f}")

    print("\n--- Validation Set ---")
    val_final_metrics = evaluate_on_split(model, val_loader, criterion, device, exp_dir, "val")
    for key, value in val_final_metrics.items():
        if not np.isnan(value):
            print(f"  {key}: {value:.6f}")

    print("\n--- Test Set ---")
    test_final_metrics = evaluate_on_split(model, test_loader, criterion, device, exp_dir, "test")
    for key, value in test_final_metrics.items():
        if not np.isnan(value):
            print(f"  {key}: {value:.6f}")

    # Save final results
    final_results = {
        'experiment_name': exp_name,
        'description': args.description,
        'config': config,
        'train_metrics': train_final_metrics,
        'val_metrics': val_final_metrics,
        'test_metrics': test_final_metrics,
        'best_epoch': checkpoint.get('epoch', -1) + 1 if os.path.exists(best_checkpoint_path) else -1,
        'total_epochs': len(history['train_loss']),
        'early_stopped': early_stopping.early_stop if early_stopping else False,
    }

    results_path = os.path.join(exp_dir, "final_results.json")
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\nTraining complete. Results saved to {results_path}")

    # Close writer
    writer.close()

    return exp_dir, final_results


if __name__ == "__main__":
    main()
