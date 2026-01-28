#!/usr/bin/env python3
"""Evaluate models on the SAME region definitions for fair comparison.

Compares:
1. Baseline model (no normalization)
2. Normalized 99pct model
3. Normalized 90pct model
4. Two-model approach (near model in near region, far model in far region)

All evaluated on the SAME region definitions based on 90th percentile threshold.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_dataloaders, PotentialFieldDataset
from src.models.wrapper import build_model, get_device
from src.utils.metrics import (
    compute_all_metrics_extended,
    mse, relative_l2_error,
)


def create_percentile_masks(target: torch.Tensor, muscle_mask: torch.Tensor, percentile: float = 90.0):
    """Create near/far masks based on percentile threshold.

    Args:
        target: Target potential (B, 1, D, H, W)
        muscle_mask: Muscle region mask (B, 1, D, H, W)
        percentile: Percentile for threshold (e.g., 90 means top 10% is "near")

    Returns:
        near_mask: Mask for high-value (near-source) region
        far_mask: Mask for low-value (far-from-source) region
    """
    B = target.shape[0]
    near_masks = []
    far_masks = []

    for b in range(B):
        muscle_bool = muscle_mask[b, 0] > 0.5
        muscle_values = torch.abs(target[b, 0][muscle_bool])

        if muscle_values.numel() > 0:
            threshold = torch.quantile(muscle_values, percentile / 100.0)
            near_mask = (torch.abs(target[b, 0]) >= threshold).float()
            far_mask = (torch.abs(target[b, 0]) < threshold).float() * muscle_bool.float()
        else:
            near_mask = torch.zeros_like(target[b, 0])
            far_mask = torch.zeros_like(target[b, 0])

        near_masks.append(near_mask)
        far_masks.append(far_mask)

    near_mask = torch.stack(near_masks, dim=0).unsqueeze(1)
    far_mask = torch.stack(far_masks, dim=0).unsqueeze(1)

    return near_mask, far_mask


def evaluate_model_on_regions(
    model,
    test_loader_raw,
    test_loader_norm,
    device,
    percentile: float = 90.0,
    needs_denorm: bool = False,
):
    """Evaluate a model on specific regions defined by percentile threshold.

    Args:
        model: Model to evaluate
        test_loader_raw: Loader with raw (unnormalized) targets for evaluation
        test_loader_norm: Loader with normalization stats for denormalization
        device: Compute device
        percentile: Percentile for near/far split
        needs_denorm: Whether model predictions need to be denormalized
    """
    model.eval()

    near_mses = []
    far_mses = []
    near_rel_l2s = []
    far_rel_l2s = []
    overall_rel_l2s = []

    with torch.no_grad():
        for batch_raw, batch_norm in zip(test_loader_raw, test_loader_norm):
            sigma = batch_raw['sigma'].to(device)
            source = batch_raw['source'].to(device)
            coords = batch_raw['coords'].to(device)
            spacing = batch_raw['spacing'].to(device)
            # ALWAYS use raw target for evaluation
            target_raw = batch_raw['target'].to(device)

            # Get muscle mask
            from src.utils.metrics import create_muscle_mask
            muscle_mask = create_muscle_mask(sigma)

            analytical = batch_raw.get('analytical', None)
            if analytical is not None:
                analytical = analytical.to(device)

            # Get prediction
            pred = model(sigma, source, coords, spacing, analytical=analytical)

            # Denormalize if model was trained on normalized targets
            if needs_denorm:
                target_mean = batch_norm.get('target_mean', torch.zeros(1)).to(device)
                target_std = batch_norm.get('target_std', torch.ones(1)).to(device)
                pred = pred * target_std.view(-1, 1, 1, 1, 1) + target_mean.view(-1, 1, 1, 1, 1)

            # Create region masks based on RAW target values
            near_mask, far_mask = create_percentile_masks(target_raw, muscle_mask, percentile)

            # Compute metrics against RAW targets
            if near_mask.sum() > 0:
                near_mses.append(mse(pred, target_raw, near_mask).item())
                near_rel_l2s.append(relative_l2_error(pred, target_raw, near_mask).item())

            if far_mask.sum() > 0:
                far_mses.append(mse(pred, target_raw, far_mask).item())
                far_rel_l2s.append(relative_l2_error(pred, target_raw, far_mask).item())

            # Overall on muscle
            overall_rel_l2s.append(relative_l2_error(pred, target_raw, muscle_mask).item())

    return {
        'near_mse': np.mean(near_mses) if near_mses else float('nan'),
        'far_mse': np.mean(far_mses) if far_mses else float('nan'),
        'near_rel_l2': np.mean(near_rel_l2s) if near_rel_l2s else float('nan'),
        'far_rel_l2': np.mean(far_rel_l2s) if far_rel_l2s else float('nan'),
        'overall_rel_l2': np.mean(overall_rel_l2s) if overall_rel_l2s else float('nan'),
    }


def evaluate_two_model_combined(
    model_near,
    model_far,
    test_loader_raw,
    test_loader_norm,
    device,
    percentile: float = 90.0,
):
    """Evaluate two-model approach: use near model in near region, far model in far region."""
    model_near.eval()
    model_far.eval()

    combined_mses = []
    combined_rel_l2s = []
    near_region_rel_l2s = []
    far_region_rel_l2s = []

    with torch.no_grad():
        for batch_raw, batch_norm in zip(test_loader_raw, test_loader_norm):
            sigma = batch_raw['sigma'].to(device)
            source = batch_raw['source'].to(device)
            coords = batch_raw['coords'].to(device)
            spacing = batch_raw['spacing'].to(device)
            target_raw = batch_raw['target'].to(device)

            # Normalization stats for far model
            target_mean = batch_norm.get('target_mean', torch.zeros(1)).to(device)
            target_std = batch_norm.get('target_std', torch.ones(1)).to(device)

            # Get muscle mask
            from src.utils.metrics import create_muscle_mask
            muscle_mask = create_muscle_mask(sigma)

            analytical = batch_raw.get('analytical', None)
            if analytical is not None:
                analytical = analytical.to(device)

            # Get predictions from both models
            pred_near = model_near(sigma, source, coords, spacing, analytical=analytical)
            pred_far_norm = model_far(sigma, source, coords, spacing, analytical=analytical)

            # Denormalize far prediction
            pred_far = pred_far_norm * target_std.view(-1, 1, 1, 1, 1) + target_mean.view(-1, 1, 1, 1, 1)

            # Create region masks based on RAW target values
            near_mask, far_mask = create_percentile_masks(target_raw, muscle_mask, percentile)

            # Combine: use near model in near region, far model in far region
            pred_combined = near_mask * pred_near + far_mask * pred_far

            # Also need to handle regions outside both masks (use average or one of them)
            # For simplicity, use near model outside both specific regions
            other_mask = muscle_mask * (1 - near_mask) * (1 - far_mask)
            pred_combined = pred_combined + other_mask * pred_near

            # Compute metrics on muscle region
            if near_mask.sum() > 0:
                near_region_rel_l2s.append(relative_l2_error(pred_combined, target_raw, near_mask).item())

            if far_mask.sum() > 0:
                far_region_rel_l2s.append(relative_l2_error(pred_combined, target_raw, far_mask).item())

            combined_rel_l2s.append(relative_l2_error(pred_combined, target_raw, muscle_mask).item())
            combined_mses.append(mse(pred_combined, target_raw, muscle_mask).item())

    return {
        'combined_mse': np.mean(combined_mses),
        'combined_rel_l2': np.mean(combined_rel_l2s),
        'near_region_rel_l2': np.mean(near_region_rel_l2s) if near_region_rel_l2s else float('nan'),
        'far_region_rel_l2': np.mean(far_region_rel_l2s) if far_region_rel_l2s else float('nan'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-checkpoint', type=str, required=True)
    parser.add_argument('--norm99-checkpoint', type=str, required=True)
    parser.add_argument('--norm90-checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--percentile', type=float, default=90.0, help='Percentile for near/far split')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Create data loaders
    config_raw = config.copy()
    config_raw['data'] = config['data'].copy()
    config_raw['data']['normalize_target'] = False

    config_norm = config.copy()
    config_norm['data'] = config['data'].copy()
    config_norm['data']['normalize_target'] = True
    config_norm['data']['singularity_percentile'] = 99.0

    _, _, test_loader_raw = get_dataloaders(config_raw)
    _, _, test_loader_norm = get_dataloaders(config_norm)

    print(f"Test samples: {len(test_loader_raw.dataset)}")
    print(f"Using {args.percentile}th percentile for near/far split")
    print(f"  - Near region: top {100 - args.percentile}% of absolute values")
    print(f"  - Far region: bottom {args.percentile}% of absolute values (within muscle)")

    # Load models
    print("\nLoading models...")

    def load_checkpoint(model, path):
        checkpoint = torch.load(path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return model

    model_baseline = build_model(config).to(device)
    load_checkpoint(model_baseline, args.baseline_checkpoint)
    print(f"  Baseline: {args.baseline_checkpoint}")

    model_norm99 = build_model(config).to(device)
    load_checkpoint(model_norm99, args.norm99_checkpoint)
    print(f"  Norm99: {args.norm99_checkpoint}")

    model_norm90 = build_model(config).to(device)
    load_checkpoint(model_norm90, args.norm90_checkpoint)
    print(f"  Norm90: {args.norm90_checkpoint}")

    # Evaluate each model on the SAME regions
    print(f"\n{'='*60}")
    print("Evaluating all models on SAME region definitions")
    print(f"{'='*60}")

    print("\n1. Baseline (no normalization)...")
    results_baseline = evaluate_model_on_regions(
        model_baseline, test_loader_raw, test_loader_raw, device, args.percentile, needs_denorm=False
    )

    print("2. Normalized 99pct...")
    results_norm99 = evaluate_model_on_regions(
        model_norm99, test_loader_raw, test_loader_norm, device, args.percentile, needs_denorm=True
    )

    print("3. Normalized 90pct...")
    # Need test_loader with 90pct normalization stats for denormalizing
    config_norm90 = config.copy()
    config_norm90['data'] = config['data'].copy()
    config_norm90['data']['normalize_target'] = True
    config_norm90['data']['singularity_percentile'] = 90.0
    _, _, test_loader_norm90 = get_dataloaders(config_norm90)

    results_norm90 = evaluate_model_on_regions(
        model_norm90, test_loader_raw, test_loader_norm90, device, args.percentile, needs_denorm=True
    )

    # Two-model approach: use baseline in near region, norm99 in far region
    print("\n4. Two-model approach (baseline near + norm99 far)...")
    results_two_model = evaluate_two_model_combined(
        model_baseline, model_norm99,
        test_loader_raw, test_loader_norm,
        device, args.percentile
    )

    # Print comparison table
    print(f"\n{'='*60}")
    print("RESULTS: All models evaluated on SAME 90pct regions")
    print(f"{'='*60}")

    print(f"\n{'Model':<25} {'Near Rel L2':<15} {'Far Rel L2':<15} {'Overall':<15}")
    print("-" * 70)
    print(f"{'Baseline':<25} {results_baseline['near_rel_l2']:.4f}         {results_baseline['far_rel_l2']:.4f}         {results_baseline['overall_rel_l2']:.4f}")
    print(f"{'Normalized 99pct':<25} {results_norm99['near_rel_l2']:.4f}         {results_norm99['far_rel_l2']:.4f}         {results_norm99['overall_rel_l2']:.4f}")
    print(f"{'Normalized 90pct':<25} {results_norm90['near_rel_l2']:.4f}         {results_norm90['far_rel_l2']:.4f}         {results_norm90['overall_rel_l2']:.4f}")
    print(f"{'Two-model (base+norm99)':<25} {results_two_model['near_region_rel_l2']:.4f}         {results_two_model['far_region_rel_l2']:.4f}         {results_two_model['combined_rel_l2']:.4f}")

    # Best in each region
    print(f"\n--- Best in each region ---")
    near_vals = [
        ('Baseline', results_baseline['near_rel_l2']),
        ('Norm99', results_norm99['near_rel_l2']),
        ('Norm90', results_norm90['near_rel_l2']),
        ('Two-model', results_two_model['near_region_rel_l2']),
    ]
    far_vals = [
        ('Baseline', results_baseline['far_rel_l2']),
        ('Norm99', results_norm99['far_rel_l2']),
        ('Norm90', results_norm90['far_rel_l2']),
        ('Two-model', results_two_model['far_region_rel_l2']),
    ]

    best_near = min(near_vals, key=lambda x: x[1])
    best_far = min(far_vals, key=lambda x: x[1])

    print(f"  Near region: {best_near[0]} ({best_near[1]:.4f})")
    print(f"  Far region:  {best_far[0]} ({best_far[1]:.4f})")

    # Save results
    results = {
        'percentile': args.percentile,
        'baseline': results_baseline,
        'norm99': results_norm99,
        'norm90': results_norm90,
        'two_model': results_two_model,
    }

    output_path = Path('experiments') / 'region_comparison_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
