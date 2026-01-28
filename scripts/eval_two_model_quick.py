#!/usr/bin/env python3
"""Quick script to evaluate two-model combination with correct denormalization."""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_dataloaders
from src.models.wrapper import build_model, get_device
from src.utils.metrics import relative_l2_error, mse, create_muscle_mask
import yaml


def compute_norm_stats(target, muscle_mask, percentile):
    """Compute normalization stats excluding top percentile."""
    B = target.shape[0]
    means = []
    stds = []
    for b in range(B):
        muscle_bool = muscle_mask[b, 0] > 0.5
        muscle_values = target[b, 0][muscle_bool]
        threshold = torch.quantile(torch.abs(muscle_values), percentile / 100.0)
        valid_mask = muscle_bool & (torch.abs(target[b, 0]) <= threshold)
        valid_values = target[b, 0][valid_mask]
        mean = valid_values.mean()
        std = valid_values.std().clamp(min=1e-8)
        means.append(mean)
        stds.append(std)
    return torch.stack(means), torch.stack(stds)


def create_percentile_masks(target, muscle_mask, percentile=90.0):
    """Create near/far masks based on percentile threshold."""
    B = target.shape[0]
    near_masks = []
    far_masks = []

    for b in range(B):
        muscle_bool = muscle_mask[b, 0] > 0.5
        muscle_values = torch.abs(target[b, 0][muscle_bool])
        threshold = torch.quantile(muscle_values, percentile / 100.0)
        near_mask = (torch.abs(target[b, 0]) >= threshold).float()
        far_mask = (torch.abs(target[b, 0]) < threshold).float() * muscle_bool.float()
        near_masks.append(near_mask)
        far_masks.append(far_mask)

    return torch.stack(near_masks, dim=0).unsqueeze(1), torch.stack(far_masks, dim=0).unsqueeze(1)


def main():
    device = get_device()
    print(f"Using device: {device}")

    with open('configs/ablations/normalized_target_90pct.yaml') as f:
        config = yaml.safe_load(f)

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
    print(f"Loaded baseline (layers6): {baseline_ckpt}")

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
    results = {
        'baseline': {'near': [], 'far': [], 'overall': []},
        'norm99': {'near': [], 'far': [], 'overall': []},
        'norm90': {'near': [], 'far': [], 'overall': []},
        'two_model_base_norm90': {'near': [], 'far': [], 'overall': []},
    }

    percentile = 90.0

    with torch.no_grad():
        for batch in test_loader:
            sigma = batch['sigma'].to(device)
            source = batch['source'].to(device)
            coords = batch['coords'].to(device)
            spacing = batch['spacing'].to(device)
            target = batch['target'].to(device)

            muscle_mask = create_muscle_mask(sigma)
            analytical = batch.get('analytical')
            if analytical is not None:
                analytical = analytical.to(device)

            # Get raw predictions
            pred_baseline = model_baseline(sigma, source, coords, spacing, analytical=analytical)
            pred_norm99_raw = model_norm99(sigma, source, coords, spacing, analytical=analytical)
            pred_norm90_raw = model_norm90(sigma, source, coords, spacing, analytical=analytical)

            # Compute norm stats and denormalize
            mean_99, std_99 = compute_norm_stats(target, muscle_mask, 99.0)
            mean_90, std_90 = compute_norm_stats(target, muscle_mask, 90.0)

            pred_norm99 = pred_norm99_raw * std_99.view(-1, 1, 1, 1, 1) + mean_99.view(-1, 1, 1, 1, 1)
            pred_norm90 = pred_norm90_raw * std_90.view(-1, 1, 1, 1, 1) + mean_90.view(-1, 1, 1, 1, 1)

            # Create region masks
            near_mask, far_mask = create_percentile_masks(target, muscle_mask, percentile)

            # Two-model: baseline in near, norm90 in far
            pred_combined = near_mask * pred_baseline + far_mask * pred_norm90

            # Compute metrics for each model
            models_preds = [
                ('baseline', pred_baseline),
                ('norm99', pred_norm99),
                ('norm90', pred_norm90),
                ('two_model_base_norm90', pred_combined),
            ]

            for name, pred in models_preds:
                if near_mask.sum() > 0:
                    results[name]['near'].append(relative_l2_error(pred, target, near_mask).item())
                if far_mask.sum() > 0:
                    results[name]['far'].append(relative_l2_error(pred, target, far_mask).item())
                results[name]['overall'].append(relative_l2_error(pred, target, muscle_mask).item())

    print("\n" + "="*70)
    print("RESULTS: All Models on SAME 90pct Region Definition")
    print("Region: top 10% absolute values = near, bottom 90% = far")
    print("="*70)

    print(f"\n{'Model':<30} {'Near Rel L2':<15} {'Far Rel L2':<15} {'Overall':<15}")
    print("-"*75)

    for name in ['baseline', 'norm99', 'norm90', 'two_model_base_norm90']:
        near = np.mean(results[name]['near']) if results[name]['near'] else float('nan')
        far = np.mean(results[name]['far']) if results[name]['far'] else float('nan')
        overall = np.mean(results[name]['overall']) if results[name]['overall'] else float('nan')
        display_name = {
            'baseline': 'Baseline (layers6)',
            'norm99': 'Norm99',
            'norm90': 'Norm90',
            'two_model_base_norm90': 'Two-model (base+norm90)',
        }[name]
        print(f"{display_name:<30} {near:.4f}          {far:.4f}          {overall:.4f}")

    print("\n--- Best in Each Region ---")
    near_results = [(name, np.mean(results[name]['near'])) for name in ['baseline', 'norm99', 'norm90']]
    far_results = [(name, np.mean(results[name]['far'])) for name in ['baseline', 'norm99', 'norm90']]
    overall_results = [(name, np.mean(results[name]['overall'])) for name in ['baseline', 'norm99', 'norm90', 'two_model_base_norm90']]

    best_near = min(near_results, key=lambda x: x[1])
    best_far = min(far_results, key=lambda x: x[1])
    best_overall = min(overall_results, key=lambda x: x[1])

    print(f"Near region:    {best_near[0]} ({best_near[1]:.4f})")
    print(f"Far region:     {best_far[0]} ({best_far[1]:.4f})")
    print(f"Overall:        {best_overall[0]} ({best_overall[1]:.4f})")

    # Optimal two-model combination
    print("\n--- Optimal Two-Model Combination ---")
    print(f"Use {best_near[0]} in near region, {best_far[0]} in far region")

    # If best near and best far are different, compute optimal combined
    if best_near[0] != best_far[0]:
        print("(This is already computed above as two_model_base_norm90)" if best_near[0] == 'baseline' and best_far[0] == 'norm90' else "")


if __name__ == '__main__':
    main()
