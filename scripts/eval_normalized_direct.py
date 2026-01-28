#!/usr/bin/env python3
"""Evaluate normalized models WITHOUT denormalization.

Compare predictions directly against normalized targets in the regions
the models were trained on.
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_dataloaders
from src.models.wrapper import build_model, get_device
from src.utils.metrics import relative_l2_error, mse, create_muscle_mask
import yaml


def compute_norm_stats_and_mask(target, muscle_mask, percentile):
    """Compute normalization stats and valid mask excluding top percentile."""
    B = target.shape[0]
    means = []
    stds = []
    valid_masks = []
    singularity_masks = []

    for b in range(B):
        muscle_bool = muscle_mask[b, 0] > 0.5
        muscle_values = target[b, 0][muscle_bool]

        # Threshold based on percentile of absolute values
        threshold = torch.quantile(torch.abs(muscle_values), percentile / 100.0)

        # Singularity = above threshold (excluded from normalization)
        singularity_mask = (torch.abs(target[b, 0]) > threshold)

        # Valid = muscle AND NOT singularity
        valid_mask = muscle_bool & ~singularity_mask

        valid_values = target[b, 0][valid_mask]
        mean = valid_values.mean()
        std = valid_values.std().clamp(min=1e-8)

        means.append(mean)
        stds.append(std)
        valid_masks.append(valid_mask.float())
        singularity_masks.append(singularity_mask.float())

    return (torch.stack(means), torch.stack(stds),
            torch.stack(valid_masks).unsqueeze(1),
            torch.stack(singularity_masks).unsqueeze(1))


def main():
    device = get_device()
    print(f"Using device: {device}")

    with open('configs/ablations/normalized_target_90pct.yaml') as f:
        config = yaml.safe_load(f)

    # Load raw data (no normalization in loader)
    config['data']['normalize_target'] = False
    _, _, test_loader = get_dataloaders(config, seed=42)
    print(f"Test samples: {len(test_loader.dataset)}")

    # Load models
    baseline_ckpt = 'experiments/layers6_analytical_fno_20260123_144632/checkpoints/best_model.pt'
    norm99_ckpt = 'experiments/normalized_target_fno_20260127_185441/checkpoints/best_model.pt'
    norm90_ckpt = 'experiments/normalized_target_90pct_fno_20260128_140719/checkpoints/best_model.pt'

    model_baseline = build_model(config).to(device)
    ckpt = torch.load(baseline_ckpt, map_location=device)
    model_baseline.load_state_dict(ckpt['model_state_dict'])
    model_baseline.eval()
    print(f"Loaded baseline: {baseline_ckpt}")

    model_norm99 = build_model(config).to(device)
    ckpt = torch.load(norm99_ckpt, map_location=device)
    model_norm99.load_state_dict(ckpt['model_state_dict'])
    model_norm99.eval()
    print(f"Loaded norm99: {norm99_ckpt}")

    model_norm90 = build_model(config).to(device)
    ckpt = torch.load(norm90_ckpt, map_location=device)
    model_norm90.load_state_dict(ckpt['model_state_dict'])
    model_norm90.eval()
    print(f"Loaded norm90: {norm90_ckpt}")

    # Results storage
    # Baseline: evaluated on raw targets, full muscle region
    baseline_results = {'muscle': [], 'far90': [], 'far99': []}

    # Norm99: evaluated on 99pct-normalized targets, in far99 region (valid region)
    norm99_results = {'normalized_far99': []}

    # Norm90: evaluated on 90pct-normalized targets, in far90 region (valid region)
    norm90_results = {'normalized_far90': []}

    with torch.no_grad():
        for batch in test_loader:
            sigma = batch['sigma'].to(device)
            source = batch['source'].to(device)
            coords = batch['coords'].to(device)
            spacing = batch['spacing'].to(device)
            target_raw = batch['target'].to(device)

            muscle_mask = create_muscle_mask(sigma)
            analytical = batch.get('analytical')
            if analytical is not None:
                analytical = analytical.to(device)

            # Compute normalization stats and masks for both percentiles
            mean_99, std_99, valid_mask_99, sing_mask_99 = compute_norm_stats_and_mask(
                target_raw, muscle_mask, 99.0)
            mean_90, std_90, valid_mask_90, sing_mask_90 = compute_norm_stats_and_mask(
                target_raw, muscle_mask, 90.0)

            # Normalized targets
            target_norm99 = (target_raw - mean_99.view(-1,1,1,1,1)) / std_99.view(-1,1,1,1,1)
            target_norm90 = (target_raw - mean_90.view(-1,1,1,1,1)) / std_90.view(-1,1,1,1,1)

            # Get predictions
            pred_baseline = model_baseline(sigma, source, coords, spacing, analytical=analytical)
            pred_norm99 = model_norm99(sigma, source, coords, spacing, analytical=analytical)
            pred_norm90 = model_norm90(sigma, source, coords, spacing, analytical=analytical)

            # Baseline: compare raw pred vs raw target
            baseline_results['muscle'].append(
                relative_l2_error(pred_baseline, target_raw, muscle_mask).item())
            baseline_results['far90'].append(
                relative_l2_error(pred_baseline, target_raw, valid_mask_90).item())
            baseline_results['far99'].append(
                relative_l2_error(pred_baseline, target_raw, valid_mask_99).item())

            # Norm99: compare normalized pred vs normalized target in valid region
            # The model outputs normalized predictions, compare to normalized targets
            norm99_results['normalized_far99'].append(
                relative_l2_error(pred_norm99, target_norm99, valid_mask_99).item())

            # Norm90: compare normalized pred vs normalized target in valid region
            norm90_results['normalized_far90'].append(
                relative_l2_error(pred_norm90, target_norm90, valid_mask_90).item())

    print("\n" + "="*70)
    print("RESULTS: Direct Comparison (No Denormalization)")
    print("="*70)

    print("\n--- Baseline (predicts raw values) ---")
    print(f"  Full muscle region:     Rel L2 = {np.mean(baseline_results['muscle']):.4f}")
    print(f"  Far-90 region (bottom 90%): Rel L2 = {np.mean(baseline_results['far90']):.4f}")
    print(f"  Far-99 region (bottom 99%): Rel L2 = {np.mean(baseline_results['far99']):.4f}")

    print("\n--- Norm99 (predicts 99pct-normalized values) ---")
    print(f"  Far-99 region (where it was trained): Rel L2 = {np.mean(norm99_results['normalized_far99']):.4f}")

    print("\n--- Norm90 (predicts 90pct-normalized values) ---")
    print(f"  Far-90 region (where it was trained): Rel L2 = {np.mean(norm90_results['normalized_far90']):.4f}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
The normalized models predict in normalized space. To fairly compare:
- Baseline on far-90 region (raw): {:.4f}
- Norm90 on far-90 region (normalized): {:.4f}

- Baseline on far-99 region (raw): {:.4f}
- Norm99 on far-99 region (normalized): {:.4f}

Note: These are in different spaces (raw vs normalized), so direct
comparison of numbers isn't meaningful. What matters is whether
the normalized models learn better patterns in their respective spaces.
""".format(
        np.mean(baseline_results['far90']),
        np.mean(norm90_results['normalized_far90']),
        np.mean(baseline_results['far99']),
        np.mean(norm99_results['normalized_far99'])
    ))


if __name__ == '__main__':
    main()
