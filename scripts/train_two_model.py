#!/usr/bin/env python3
"""Two-model approach for potential field prediction.

This script implements a dual-model strategy:
- Model A (near-source): Trained on raw targets with singularity mask
- Model B (far-source): Trained on normalized targets for better far-field accuracy

The predictions are combined based on distance from the source using a smooth blend.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_dataloaders
from src.models.wrapper import build_model, get_device, count_parameters
from src.utils.masking import CombinedLoss, create_combined_mask
from src.utils.metrics import (
    compute_all_metrics_extended,
    create_singularity_distance_mask,
)
from src.utils.visualization import save_validation_figure, plot_loss_curves


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_blend_mask(source: torch.Tensor, near_radius: int = 10, blend_width: int = 5) -> torch.Tensor:
    """Create smooth blend mask for combining near and far predictions.

    Args:
        source: Source field (B, 1, D, H, W)
        near_radius: Distance within which we use near-source model exclusively
        blend_width: Width of smooth transition zone

    Returns:
        Weight for far-source model (0 = near model, 1 = far model)
    """
    B, _, D, H, W = source.shape
    device = source.device

    blend_masks = []

    for b in range(B):
        # Find source peak
        source_flat = source[b, 0].reshape(-1)
        max_idx = torch.argmax(torch.abs(source_flat))
        z_idx = max_idx // (H * W)
        y_idx = (max_idx % (H * W)) // W
        x_idx = max_idx % W

        # Create coordinate grids
        z_coords = torch.arange(D, device=device).float()
        y_coords = torch.arange(H, device=device).float()
        x_coords = torch.arange(W, device=device).float()
        ZZ, YY, XX = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

        # Compute distance from source
        dist = torch.sqrt((ZZ - z_idx.float())**2 + (YY - y_idx.float())**2 + (XX - x_idx.float())**2)

        # Create smooth blend: 0 for near, 1 for far, smooth transition in between
        blend = torch.clamp((dist - near_radius) / blend_width, 0, 1)
        blend_masks.append(blend)

    return torch.stack(blend_masks, dim=0).unsqueeze(1)


def combine_predictions(
    pred_near: torch.Tensor,
    pred_far: torch.Tensor,
    source: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    near_radius: int = 10,
    blend_width: int = 5,
) -> torch.Tensor:
    """Combine near and far model predictions.

    Args:
        pred_near: Near-source model prediction (raw scale)
        pred_far: Far-source model prediction (normalized scale)
        source: Source field for computing blend mask
        target_mean: Mean for denormalizing far prediction
        target_std: Std for denormalizing far prediction
        near_radius: Radius for near model exclusivity
        blend_width: Transition zone width

    Returns:
        Combined prediction in raw scale
    """
    # Denormalize far prediction
    pred_far_denorm = pred_far * target_std.view(-1, 1, 1, 1, 1) + target_mean.view(-1, 1, 1, 1, 1)

    # Create blend mask
    blend = create_blend_mask(source, near_radius, blend_width)

    # Combine: (1-blend)*near + blend*far
    combined = (1 - blend) * pred_near + blend * pred_far_denorm

    return combined


def train_single_model(
    config: dict,
    train_loader,
    val_loader,
    device: torch.device,
    exp_dir: str,
    model_name: str,
    writer: SummaryWriter,
) -> torch.nn.Module:
    """Train a single model."""
    print(f"\n{'='*50}")
    print(f"Training {model_name} model...")
    print(f"{'='*50}")

    # Build model
    model = build_model(config).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['scheduler_params'].get('eta_min', 1e-6),
    )

    # Setup loss
    criterion = CombinedLoss(
        tv_weight=config['loss'].get('tv_weight', 0.01),
        gradient_matching_weight=config['loss'].get('gradient_matching_weight', 0.0),
        use_singularity_mask=config['experiment'].get('use_singularity_mask', True),
        singularity_radius=config['physics']['mse'].get('singularity_mask_radius', 3),
    )

    best_val_loss = float('inf')
    patience_counter = 0
    patience = config['training'].get('early_stopping_patience', 10)

    for epoch in range(config['training']['epochs']):
        # Training
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            sigma = batch['sigma'].to(device)
            source = batch['source'].to(device)
            coords = batch['coords'].to(device)
            spacing = batch['spacing'].to(device)
            target = batch['target'].to(device)

            analytical = batch.get('analytical', None)
            if analytical is not None:
                analytical = analytical.to(device)

            optimizer.zero_grad()
            pred = model(sigma, source, coords, spacing, analytical=analytical)

            loss, _ = criterion(pred, target, sigma, source, spacing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                sigma = batch['sigma'].to(device)
                source = batch['source'].to(device)
                coords = batch['coords'].to(device)
                spacing = batch['spacing'].to(device)
                target = batch['target'].to(device)

                analytical = batch.get('analytical', None)
                if analytical is not None:
                    analytical = analytical.to(device)

                pred = model(sigma, source, coords, spacing, analytical=analytical)
                loss, _ = criterion(pred, target, sigma, source, spacing)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        scheduler.step()

        # Logging
        writer.add_scalar(f'{model_name}/train_loss', avg_train_loss, epoch)
        writer.add_scalar(f'{model_name}/val_loss', avg_val_loss, epoch)

        # Early stopping
        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(exp_dir, f'{model_name}_best.pt'))
            print(f"Epoch {epoch+1}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f} (saved)")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f} (patience {patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load(os.path.join(exp_dir, f'{model_name}_best.pt')))
    return model


def evaluate_combined_proper(
    model_near: torch.nn.Module,
    model_far: torch.nn.Module,
    config: dict,
    device: torch.device,
    near_radius: int = 10,
    blend_width: int = 5,
) -> dict:
    """Evaluate combined two-model system with proper handling of targets.

    We need to:
    1. Get raw targets for evaluation
    2. Get normalized targets + stats for far model evaluation
    3. Compare all predictions against raw targets
    """
    model_near.eval()
    model_far.eval()

    # Create two test loaders - one raw, one normalized
    config_raw = config.copy()
    config_raw['data'] = config['data'].copy()
    config_raw['data']['normalize_target'] = False

    config_norm = config.copy()
    config_norm['data'] = config['data'].copy()
    config_norm['data']['normalize_target'] = True
    config_norm['data']['singularity_percentile'] = 99.0

    _, _, test_loader_raw = get_dataloaders(config_raw)
    _, _, test_loader_norm = get_dataloaders(config_norm)

    all_metrics_combined = []
    all_metrics_near_only = []
    all_metrics_far_only = []

    with torch.no_grad():
        for batch_raw, batch_norm in zip(test_loader_raw, test_loader_norm):
            # Common inputs
            sigma = batch_raw['sigma'].to(device)
            source = batch_raw['source'].to(device)
            coords = batch_raw['coords'].to(device)
            spacing = batch_raw['spacing'].to(device)

            # Raw target for evaluation
            target_raw = batch_raw['target'].to(device)

            # Normalization stats for denormalizing far model output
            target_mean = batch_norm.get('target_mean', torch.zeros(1)).to(device)
            target_std = batch_norm.get('target_std', torch.ones(1)).to(device)

            analytical = batch_raw.get('analytical', None)
            if analytical is not None:
                analytical = analytical.to(device)

            # Get predictions from both models
            pred_near = model_near(sigma, source, coords, spacing, analytical=analytical)
            pred_far_norm = model_far(sigma, source, coords, spacing, analytical=analytical)

            # Denormalize far prediction
            pred_far = pred_far_norm * target_std.view(-1, 1, 1, 1, 1) + target_mean.view(-1, 1, 1, 1, 1)

            # Combined prediction using blend mask
            blend = create_blend_mask(source, near_radius, blend_width)
            pred_combined = (1 - blend) * pred_near + blend * pred_far

            # Compute metrics against RAW targets
            metrics_combined = compute_all_metrics_extended(pred_combined, target_raw, sigma, source, spacing)
            metrics_near = compute_all_metrics_extended(pred_near, target_raw, sigma, source, spacing)
            metrics_far = compute_all_metrics_extended(pred_far, target_raw, sigma, source, spacing)

            all_metrics_combined.append(metrics_combined)
            all_metrics_near_only.append(metrics_near)
            all_metrics_far_only.append(metrics_far)

    # Average metrics
    def avg_metrics(metrics_list):
        avg = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if not np.isnan(m[key])]
            avg[key] = np.mean(values) if values else float('nan')
        return avg

    return {
        'combined': avg_metrics(all_metrics_combined),
        'near_only': avg_metrics(all_metrics_near_only),
        'far_only': avg_metrics(all_metrics_far_only),
    }


def main():
    parser = argparse.ArgumentParser(description='Two-model training for potential field prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--experiment-name', type=str, default='two_model', help='Experiment name')
    parser.add_argument('--near-radius', type=int, default=10, help='Near model radius')
    parser.add_argument('--blend-width', type=int, default=5, help='Blend transition width')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Load config
    config = load_config(args.config)

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("experiments") / f"{args.experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=str(exp_dir / "logs"))

    # Save config
    with open(exp_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    # ===== STEP 1: Train near-source model (raw targets) =====
    print("\n" + "="*60)
    print("STEP 1: Training NEAR-SOURCE model (raw targets)")
    print("="*60)

    config_near = config.copy()
    config_near['data'] = config['data'].copy()
    config_near['data']['normalize_target'] = False  # Raw targets

    train_loader_near, val_loader_near, test_loader_near = get_dataloaders(
        config_near,
        seed=args.seed,
    )
    print(f"Near model - Train: {len(train_loader_near.dataset)}, Val: {len(val_loader_near.dataset)}, Test: {len(test_loader_near.dataset)}")

    model_near = train_single_model(
        config_near, train_loader_near, val_loader_near,
        device, str(exp_dir), 'near', writer
    )

    # ===== STEP 2: Train far-source model (normalized targets) =====
    print("\n" + "="*60)
    print("STEP 2: Training FAR-SOURCE model (normalized targets)")
    print("="*60)

    config_far = config.copy()
    config_far['data'] = config['data'].copy()
    config_far['data']['normalize_target'] = True  # Normalized targets
    config_far['data']['singularity_percentile'] = 99.0  # Top 1% as singularity

    train_loader_far, val_loader_far, test_loader_far = get_dataloaders(
        config_far,
        seed=args.seed,
    )
    print(f"Far model - Train: {len(train_loader_far.dataset)}, Val: {len(val_loader_far.dataset)}, Test: {len(test_loader_far.dataset)}")

    model_far = train_single_model(
        config_far, train_loader_far, val_loader_far,
        device, str(exp_dir), 'far', writer
    )

    # ===== STEP 3: Evaluate combined system =====
    print("\n" + "="*60)
    print("STEP 3: Evaluating COMBINED system")
    print("="*60)

    # Create a test loader with both raw targets AND normalization info
    # We need raw targets for evaluation but norm info for denormalizing far predictions
    # Use a custom evaluation approach
    results = evaluate_combined_proper(
        model_near, model_far, config, device,
        near_radius=args.near_radius, blend_width=args.blend_width
    )

    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)

    print("\n--- Overall Relative L2 ---")
    print(f"  Near-only model:  {results['near_only']['relative_l2']:.4f}")
    print(f"  Far-only model:   {results['far_only']['relative_l2']:.4f}")
    print(f"  Combined model:   {results['combined']['relative_l2']:.4f}")

    print("\n--- Far-from-source Relative L2 ---")
    print(f"  Near-only model:  {results['near_only']['rel_l2_far_singularity']:.4f}")
    print(f"  Far-only model:   {results['far_only']['rel_l2_far_singularity']:.4f}")
    print(f"  Combined model:   {results['combined']['rel_l2_far_singularity']:.4f}")

    print("\n--- Near-source Relative L2 ---")
    print(f"  Near-only model:  {results['near_only']['rel_l2_near_singularity']:.4f}")
    print(f"  Far-only model:   {results['far_only']['rel_l2_near_singularity']:.4f}")
    print(f"  Combined model:   {results['combined']['rel_l2_near_singularity']:.4f}")

    print("\n--- Muscle-far Relative L2 ---")
    print(f"  Near-only model:  {results['near_only']['rel_l2_muscle_far']:.4f}")
    print(f"  Far-only model:   {results['far_only']['rel_l2_muscle_far']:.4f}")
    print(f"  Combined model:   {results['combined']['rel_l2_muscle_far']:.4f}")

    # Save results
    results_path = exp_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'near_radius': args.near_radius,
            'blend_width': args.blend_width,
            'combined': results['combined'],
            'near_only': results['near_only'],
            'far_only': results['far_only'],
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    writer.close()


if __name__ == '__main__':
    main()
