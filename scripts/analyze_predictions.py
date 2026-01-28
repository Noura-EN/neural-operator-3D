#!/usr/bin/env python3
"""Analyze what the model is actually predicting in different regions."""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_dataloaders
from src.models.wrapper import build_model, get_device
from src.utils.metrics import create_muscle_mask
import yaml


def main():
    device = get_device()

    with open('configs/ablations/mask_top10pct.yaml') as f:
        config = yaml.safe_load(f)

    config['data']['normalize_target'] = False
    _, _, test_loader = get_dataloaders(config, seed=42)

    # Load baseline model
    model = build_model(config).to(device)
    ckpt = torch.load('experiments/layers6_analytical_fno_20260123_144632/checkpoints/best_model.pt',
                      map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Analyze first few samples
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 3:
                break

            sigma = batch['sigma'].to(device)
            source = batch['source'].to(device)
            coords = batch['coords'].to(device)
            spacing = batch['spacing'].to(device)
            target = batch['target'].to(device)
            analytical = batch.get('analytical')
            if analytical is not None:
                analytical = analytical.to(device)

            muscle_mask = create_muscle_mask(sigma)
            pred = model(sigma, source, coords, spacing, analytical=analytical)

            # Get muscle voxels
            muscle_bool = muscle_mask[0, 0] > 0.5
            target_muscle = target[0, 0][muscle_bool]
            pred_muscle = pred[0, 0][muscle_bool]

            # Sort by target magnitude
            abs_target = torch.abs(target_muscle)
            sorted_idx = torch.argsort(abs_target, descending=True)

            n = len(sorted_idx)
            top10_idx = sorted_idx[:n//10]
            bottom90_idx = sorted_idx[n//10:]

            # Analyze each region
            print(f"\n=== Sample {i} ===")
            print(f"Total muscle voxels: {n}")

            # Near region (top 10%)
            t_near = target_muscle[top10_idx]
            p_near = pred_muscle[top10_idx]
            print(f"\nNear region (top 10% by |target|):")
            print(f"  Target range: [{t_near.min():.4f}, {t_near.max():.4f}]")
            print(f"  Pred range:   [{p_near.min():.4f}, {p_near.max():.4f}]")
            print(f"  Target mean:  {t_near.mean():.4f}, std: {t_near.std():.4f}")
            print(f"  Pred mean:    {p_near.mean():.4f}, std: {p_near.std():.4f}")
            print(f"  Correlation:  {torch.corrcoef(torch.stack([t_near, p_near]))[0,1]:.4f}")

            # Far region (bottom 90%)
            t_far = target_muscle[bottom90_idx]
            p_far = pred_muscle[bottom90_idx]
            print(f"\nFar region (bottom 90% by |target|):")
            print(f"  Target range: [{t_far.min():.4f}, {t_far.max():.4f}]")
            print(f"  Pred range:   [{p_far.min():.4f}, {p_far.max():.4f}]")
            print(f"  Target mean:  {t_far.mean():.4f}, std: {t_far.std():.4f}")
            print(f"  Pred mean:    {p_far.mean():.4f}, std: {p_far.std():.4f}")
            print(f"  Correlation:  {torch.corrcoef(torch.stack([t_far, p_far]))[0,1]:.4f}")

            # What fraction of loss comes from each region?
            mse_near = ((t_near - p_near)**2).mean()
            mse_far = ((t_far - p_far)**2).mean()
            total_mse = ((target_muscle - pred_muscle)**2).mean()
            print(f"\nMSE contribution:")
            print(f"  Near MSE: {mse_near:.6f}")
            print(f"  Far MSE:  {mse_far:.6f}")
            print(f"  Ratio (near/far): {mse_near/mse_far:.1f}x")


if __name__ == '__main__':
    main()
