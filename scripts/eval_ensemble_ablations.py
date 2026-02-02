#!/usr/bin/env python3
"""Evaluate ensemble from ablation experiments."""

import sys
from pathlib import Path
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import get_dataloaders, PotentialFieldDataset
from src.models.wrapper import build_model, get_device
from src.utils.metrics import compute_all_metrics_extended
from torch.utils.data import DataLoader


SEEDS = [42, 142, 242, 342, 442, 542, 642, 742, 842, 942]


def find_experiment_dir(model_name: str, training_type: str, seed: int) -> Path:
    """Find experiment directory with final_results.json."""
    exp_base = PROJECT_ROOT / "experiments"
    pattern = f"{model_name}_{training_type}_seed{seed}"

    for d in sorted(exp_base.iterdir(), key=lambda x: x.name, reverse=True):
        if d.is_dir() and d.name.startswith(pattern):
            if (d / "final_results.json").exists():
                return d
    return None


def load_highres_test_data():
    """Load high-res test dataset."""
    highres_test_dir = PROJECT_ROOT / "data" / "highres_test_samples"
    dataset = PotentialFieldDataset(
        data_dir=str(highres_test_dir),
        sample_indices=None,
        add_analytical_solution=True,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def main():
    device = get_device()
    print(f"Device: {device}")

    model_name = "fno_analytical"
    training_type = "mixed"

    # Find all experiment directories
    exp_dirs = []
    for seed in SEEDS:
        exp_dir = find_experiment_dir(model_name, training_type, seed)
        if exp_dir:
            exp_dirs.append((seed, exp_dir))
            print(f"Found seed {seed}: {exp_dir.name}")
        else:
            print(f"Missing seed {seed}")

    if len(exp_dirs) < 10:
        print(f"\nWarning: Only {len(exp_dirs)} seeds found")

    # Load config from first experiment
    with open(exp_dirs[0][1] / "config.yaml") as f:
        config = yaml.safe_load(f)

    # Load models
    models = []
    for seed, exp_dir in exp_dirs:
        checkpoint_path = exp_dir / "checkpoints" / "best_model.pt"
        model = build_model(config)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        models.append((seed, model))
        print(f"Loaded model for seed {seed}")

    # Get dataloaders
    _, _, lowres_test_loader = get_dataloaders(config, seed=42)
    highres_test_loader = load_highres_test_data()

    print(f"\nLow-res test: {len(lowres_test_loader.dataset)} samples")
    print(f"High-res test: {len(highres_test_loader.dataset)} samples")

    # Evaluate on both test sets
    for test_name, test_loader in [("Low-res", lowres_test_loader), ("High-res", highres_test_loader)]:
        print(f"\n{'='*60}")
        print(f"Evaluating on {test_name} test set")
        print(f"{'='*60}")

        ensemble_metrics = []
        individual_metrics = {seed: [] for seed, _ in models}

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                sigma = batch['sigma'].to(device)
                source = batch['source'].to(device)
                coords = batch['coords'].to(device)
                spacing = batch['spacing'].to(device)
                target = batch['target'].to(device)

                analytical = batch.get('analytical')
                if analytical is not None:
                    analytical = analytical.to(device)

                # Get predictions from all models
                preds = []
                for seed, model in models:
                    pred = model(sigma, source, coords, spacing, analytical=analytical)
                    preds.append(pred)

                    # Compute individual metrics
                    metrics = compute_all_metrics_extended(pred, target, sigma, source, spacing)
                    individual_metrics[seed].append(metrics['relative_l2'])

                # Ensemble prediction (average)
                ensemble_pred = torch.stack(preds).mean(dim=0)

                # Compute ensemble metrics
                metrics = compute_all_metrics_extended(ensemble_pred, target, sigma, source, spacing)
                ensemble_metrics.append(metrics)

                if (batch_idx + 1) % 25 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(test_loader)} samples")

        # Aggregate ensemble metrics
        avg_ensemble_rel_l2 = np.mean([m['relative_l2'] for m in ensemble_metrics])
        avg_ensemble_l2_norm = np.mean([m['l2_norm_ratio'] for m in ensemble_metrics])
        avg_ensemble_grad = np.mean([m['gradient_energy_ratio'] for m in ensemble_metrics])

        # Aggregate individual metrics
        individual_means = [np.mean(individual_metrics[seed]) for seed, _ in models]
        avg_individual_rel_l2 = np.mean(individual_means)
        std_individual_rel_l2 = np.std(individual_means)

        # Compute improvement
        improvement = (avg_individual_rel_l2 - avg_ensemble_rel_l2) / avg_individual_rel_l2 * 100

        print(f"\n{test_name} Results:")
        print(f"  Ensemble Rel L2:       {avg_ensemble_rel_l2:.4f}")
        print(f"  Mean Individual Rel L2: {avg_individual_rel_l2:.4f} ± {std_individual_rel_l2:.4f}")
        print(f"  Improvement:           {improvement:.1f}%")
        print(f"  Ensemble L2 Norm Ratio: {avg_ensemble_l2_norm:.4f}")
        print(f"  Ensemble Grad Energy:   {avg_ensemble_grad:.4f}")


if __name__ == "__main__":
    main()
