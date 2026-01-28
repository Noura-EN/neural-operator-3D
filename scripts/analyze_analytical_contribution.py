#!/usr/bin/env python3
"""Analyze how much the analytical solution contributes vs the learned correction."""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_dataloaders
from src.utils.metrics import create_muscle_mask, relative_l2_error
import yaml


def main():
    with open('configs/ablations/mask_top10pct.yaml') as f:
        config = yaml.safe_load(f)

    config['data']['normalize_target'] = False
    _, _, test_loader = get_dataloaders(config, seed=42)

    analytical_near_errs = []
    analytical_far_errs = []
    analytical_muscle_errs = []

    for batch in test_loader:
        target = batch['target']
        analytical = batch.get('analytical')
        sigma = batch['sigma']

        if analytical is None:
            print("No analytical solution in data!")
            return

        muscle_mask = create_muscle_mask(sigma)

        # Create near/far masks
        muscle_bool = muscle_mask[0, 0] > 0.5
        muscle_values = torch.abs(target[0, 0][muscle_bool])
        threshold = torch.quantile(muscle_values, 0.90)
        near_mask = ((torch.abs(target[0, 0]) >= threshold) & muscle_bool).float().unsqueeze(0).unsqueeze(0)
        far_mask = ((torch.abs(target[0, 0]) < threshold) & muscle_bool).float().unsqueeze(0).unsqueeze(0)

        # How well does analytical alone predict?
        analytical_near_errs.append(relative_l2_error(analytical, target, near_mask).item())
        analytical_far_errs.append(relative_l2_error(analytical, target, far_mask).item())
        analytical_muscle_errs.append(relative_l2_error(analytical, target, muscle_mask).item())

    print("=== Analytical Solution Performance ===")
    print(f"(How well does the monopole Φ=I/(4πσr) predict the FEM target?)\n")
    print(f"Near region (top 10%):  Rel L2 = {np.mean(analytical_near_errs):.4f}")
    print(f"Far region (bottom 90%): Rel L2 = {np.mean(analytical_far_errs):.4f}")
    print(f"Full muscle:            Rel L2 = {np.mean(analytical_muscle_errs):.4f}")

    print("\n=== Comparison ===")
    print(f"Analytical alone (far): {np.mean(analytical_far_errs):.4f}")
    print(f"Baseline model (far):   0.1746")
    print(f"Improvement from learning: {(np.mean(analytical_far_errs) - 0.1746) / np.mean(analytical_far_errs) * 100:.1f}%")


if __name__ == '__main__':
    main()
