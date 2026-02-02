#!/usr/bin/env python3
"""
Evaluate trained models on both low-res and high-res test sets.

This script:
1. Loads a trained model from checkpoint
2. Evaluates on low-res test set (from 70/15/15 split of 901 samples)
3. Evaluates on high-res test set (50 samples in data/highres_test_samples/)
4. Saves metrics to JSON
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import get_dataloaders, PotentialFieldDataset
from src.models.wrapper import build_model, get_device
from src.utils.masking import CombinedLoss, create_combined_mask
from src.utils.metrics import compute_all_metrics_extended
from torch.utils.data import DataLoader


def evaluate_on_dataset(model, data_loader, criterion, device, config):
    """Evaluate model on a dataset and return metrics."""
    model.eval()

    all_metrics = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Move batch to device
            sigma = batch['sigma'].to(device)
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            coords = batch['coords'].to(device)
            spacing = batch['spacing'].to(device)

            # Get optional inputs
            analytical = batch.get('analytical')
            if analytical is not None:
                analytical = analytical.to(device)

            # Forward pass
            mask = create_combined_mask(sigma, source)
            if hasattr(model.backbone, 'needs_geometry') and model.backbone.needs_geometry:
                pred = model(sigma, source, coords, spacing, analytical=analytical,
                           sigma_raw=sigma, mask=mask)
            else:
                pred = model(sigma, source, coords, spacing, analytical=analytical)

            # Compute loss (need to pass sigma, source, spacing for masking)
            loss_value, loss_dict = criterion(pred, target, sigma, source, spacing)

            # Compute extended metrics
            metrics = compute_all_metrics_extended(pred, target, sigma, source, spacing)
            all_metrics.append(metrics)
            total_loss += loss_value.item() if hasattr(loss_value, 'item') else loss_value
            num_batches += 1

    # Aggregate metrics
    avg_metrics = {'loss': total_loss / num_batches if num_batches > 0 else float('nan')}
    if all_metrics:
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if not np.isnan(m[key])]
            avg_metrics[key] = np.mean(values) if values else float('nan')

    return avg_metrics


def load_highres_test_data(config):
    """Load high-res test dataset."""
    highres_test_dir = PROJECT_ROOT / "data" / "highres_test_samples"

    if not highres_test_dir.exists():
        raise ValueError(f"High-res test directory not found: {highres_test_dir}")

    # Get all sample files in high-res test dir
    sample_files = sorted(highres_test_dir.glob("*.npz"))
    if not sample_files:
        raise ValueError(f"No .npz files found in {highres_test_dir}")

    # Create dataset for high-res test samples
    dataset = PotentialFieldDataset(
        data_dir=str(highres_test_dir),
        sample_indices=None,  # Use all samples
        add_analytical_solution=config.get('model', {}).get('add_analytical_solution', True),
    )

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return data_loader


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test sets")
    parser.add_argument("--experiment-dir", type=str, required=True,
                        help="Path to experiment directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: experiment_dir/evaluation_results.json)")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        # Try to find it in experiments folder
        exp_base = PROJECT_ROOT / "experiments"
        matches = list(exp_base.glob(f"*{args.experiment_dir}*"))
        if matches:
            exp_dir = matches[0]
        else:
            raise ValueError(f"Experiment directory not found: {args.experiment_dir}")

    print(f"Evaluating: {exp_dir.name}")

    # Load config
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        raise ValueError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load checkpoint
    checkpoint_path = exp_dir / "checkpoints" / "best_model.pt"
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    print("Loading model...")
    device = get_device()
    model = build_model(config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Build criterion
    criterion = CombinedLoss(
        pde_weight=config.get('loss', {}).get('pde_weight', 0.0),
        tv_weight=config.get('loss', {}).get('tv_weight', 0.01),
        gradient_matching_weight=config.get('loss', {}).get('gradient_matching_weight', 0.0),
        use_singularity_mask=config.get('experiment', {}).get('use_singularity_mask', True),
        singularity_radius=config.get('physics', {}).get('mse', {}).get('singularity_mask_radius', 3),
    )

    # Evaluate on low-res test set
    print("\nLoading low-res test data...")
    DATA_SPLIT_SEED = 42
    _, _, lowres_test_loader = get_dataloaders(config, seed=DATA_SPLIT_SEED)
    print(f"Low-res test samples: {len(lowres_test_loader.dataset)}")

    print("Evaluating on low-res test set...")
    lowres_metrics = evaluate_on_dataset(model, lowres_test_loader, criterion, device, config)

    # Evaluate on high-res test set
    print("\nLoading high-res test data...")
    try:
        highres_test_loader = load_highres_test_data(config)
        print(f"High-res test samples: {len(highres_test_loader.dataset)}")

        print("Evaluating on high-res test set...")
        highres_metrics = evaluate_on_dataset(model, highres_test_loader, criterion, device, config)
    except Exception as e:
        print(f"Warning: Could not evaluate on high-res test set: {e}")
        highres_metrics = {}

    # Print results
    print("\n" + "="*60)
    print("LOW-RES TEST SET RESULTS (48x48x96)")
    print("="*60)
    for key, value in lowres_metrics.items():
        if not np.isnan(value):
            print(f"  {key}: {value:.6f}")

    if highres_metrics:
        print("\n" + "="*60)
        print("HIGH-RES TEST SET RESULTS (96x96x192)")
        print("="*60)
        for key, value in highres_metrics.items():
            if not np.isnan(value):
                print(f"  {key}: {value:.6f}")

    # Save results
    results = {
        'experiment_name': exp_dir.name,
        'lowres_test_metrics': {k: float(v) if not np.isnan(v) else None
                                for k, v in lowres_metrics.items()},
        'highres_test_metrics': {k: float(v) if not np.isnan(v) else None
                                 for k, v in highres_metrics.items()} if highres_metrics else {},
    }

    output_path = args.output or (exp_dir / "evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
