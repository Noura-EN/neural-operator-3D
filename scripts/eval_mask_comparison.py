#!/usr/bin/env python3
"""Compare different masking strategies: baseline (3 voxel radius), top 1%, top 10%.

All models predict raw values. Evaluate on 90pct region (bottom 90% of muscle).
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


def create_percentile_masks(target, muscle_mask, percentile=90.0):
    """Create near/far masks based on percentile threshold."""
    B = target.shape[0]
    near_masks = []
    far_masks = []

    for b in range(B):
        muscle_bool = muscle_mask[b, 0] > 0.5
        muscle_values = torch.abs(target[b, 0][muscle_bool])
        threshold = torch.quantile(muscle_values, percentile / 100.0)
        near_mask = (torch.abs(target[b, 0]) >= threshold).float() * muscle_bool.float()
        far_mask = (torch.abs(target[b, 0]) < threshold).float() * muscle_bool.float()
        near_masks.append(near_mask)
        far_masks.append(far_mask)

    return torch.stack(near_masks, dim=0).unsqueeze(1), torch.stack(far_masks, dim=0).unsqueeze(1)


def main():
    device = get_device()
    print(f"Using device: {device}")

    with open('configs/ablations/mask_top10pct.yaml') as f:
        config = yaml.safe_load(f)

    config['data']['normalize_target'] = False
    _, _, test_loader = get_dataloaders(config, seed=42)
    print(f"Test samples: {len(test_loader.dataset)}")

    # Load all models
    models = {
        'baseline': 'experiments/layers6_analytical_fno_20260123_144632/checkpoints/best_model.pt',
        'pde_0.1': 'experiments/pde_loss_0.1_fno_20260128_155433/checkpoints/best_model.pt',
    }

    loaded_models = {}
    for name, ckpt_path in models.items():
        model = build_model(config).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        loaded_models[name] = model
        print(f"Loaded {name}: {ckpt_path}")

    # Results storage
    results = {name: {'muscle': [], 'near10': [], 'far90': []} for name in models}

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

            # Create 90th percentile masks
            near_mask, far_mask = create_percentile_masks(target, muscle_mask, 90.0)

            for name, model in loaded_models.items():
                pred = model(sigma, source, coords, spacing, analytical=analytical)

                results[name]['muscle'].append(
                    relative_l2_error(pred, target, muscle_mask).item())
                results[name]['near10'].append(
                    relative_l2_error(pred, target, near_mask).item())
                results[name]['far90'].append(
                    relative_l2_error(pred, target, far_mask).item())

    print("\n" + "="*75)
    print("RESULTS: Different Masking Strategies (All Raw Targets)")
    print("Evaluated on 90th percentile region definition")
    print("="*75)

    print(f"\n{'Model':<20} {'Full Muscle':<15} {'Near 10%':<15} {'Far 90%':<15}")
    print("-"*65)

    for name in models:
        muscle = np.mean(results[name]['muscle'])
        near = np.mean(results[name]['near10'])
        far = np.mean(results[name]['far90'])
        print(f"{name:<20} {muscle:.4f}          {near:.4f}          {far:.4f}")

    print("\n--- Analysis ---")
    baseline_far = np.mean(results['baseline']['far90'])
    for name in models:
        if name != 'baseline':
            far = np.mean(results[name]['far90'])
            pct = (far - baseline_far) / baseline_far * 100
            print(f"  {name}: Far-90 = {far:.4f} ({pct:+.1f}% vs baseline {baseline_far:.4f})")


if __name__ == '__main__':
    main()
