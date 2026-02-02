#!/usr/bin/env python3
"""Evaluate an ensemble of models by averaging predictions."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_dataloaders
from src.models.wrapper import build_model, get_device
from src.utils.metrics import compute_all_metrics_extended


def load_model(checkpoint_path, config, device):
    """Load a single model from checkpoint."""
    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble of models")
    parser.add_argument("--ensemble-dir", type=str, required=True, help="Directory containing ensemble models")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Data split to evaluate")
    parser.add_argument("--save-predictions", action="store_true", help="Save ensemble predictions as NPZ")
    args = parser.parse_args()

    ensemble_dir = Path(args.ensemble_dir)
    device = get_device()

    # Find all model directories
    model_dirs = sorted(ensemble_dir.glob("model_*"))
    if not model_dirs:
        print(f"No model directories found in {ensemble_dir}")
        return

    print(f"Found {len(model_dirs)} models in ensemble")
    print(f"Device: {device}")
    print("=" * 60)

    # Load config from first model (all should have same config except seed)
    first_model_dir = model_dirs[0].resolve()
    config_path = first_model_dir / "config.yaml"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create dataloader
    _, val_loader, test_loader = get_dataloaders(config, seed=42)
    data_loader = test_loader if args.split == "test" else val_loader

    # Load all models
    models = []
    for model_dir in model_dirs:
        model_dir = model_dir.resolve()
        checkpoint_path = model_dir / "checkpoints" / "best_model.pt"
        if checkpoint_path.exists():
            print(f"Loading: {model_dir.name}")
            model = load_model(checkpoint_path, config, device)
            models.append(model)
        else:
            print(f"Warning: No checkpoint found in {model_dir}")

    if not models:
        print("No models loaded!")
        return

    print(f"\nLoaded {len(models)} models")
    print(f"Evaluating on {args.split} set ({len(data_loader.dataset)} samples)")
    print("=" * 60)

    # Evaluate ensemble
    all_metrics = []
    predictions_to_save = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            sigma = batch['sigma'].to(device)
            source = batch['source'].to(device)
            coords = batch['coords'].to(device)
            spacing = batch['spacing'].to(device)
            target = batch['target'].to(device)
            mask = batch['mask'].to(device)

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

            # Get predictions from all models
            preds = []
            for model in models:
                pred = model(sigma, source, coords, spacing, analytical=analytical, distance_field=distance_field)
                preds.append(pred)

            # Average predictions (ensemble)
            ensemble_pred = torch.stack(preds).mean(dim=0)

            # Also compute individual model predictions for comparison
            individual_metrics = []
            for pred in preds:
                metrics = compute_all_metrics_extended(
                    pred, target, sigma, source, spacing
                )
                individual_metrics.append(metrics['rel_l2_muscle'])

            # Compute ensemble metrics
            metrics = compute_all_metrics_extended(
                ensemble_pred, target, sigma, source, spacing
            )
            metrics['individual_rel_l2'] = individual_metrics
            metrics['ensemble_improvement'] = np.mean(individual_metrics) - metrics['rel_l2_muscle']
            all_metrics.append(metrics)

            if args.save_predictions:
                predictions_to_save.append({
                    'prediction': ensemble_pred.cpu().numpy(),
                    'target': target.cpu().numpy(),
                    'sigma': sigma.cpu().numpy(),
                    'source': source.cpu().numpy(),
                    'spacing': spacing.cpu().numpy(),
                })

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(data_loader)} samples")

    # Aggregate metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if key == 'individual_rel_l2':
            continue
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = float(np.mean(values))

    # Compute average individual performance
    all_individual = [m['individual_rel_l2'] for m in all_metrics]
    avg_individual = np.mean(all_individual)
    avg_improvement = avg_individual - avg_metrics['rel_l2_muscle']

    print("\n" + "=" * 60)
    print("ENSEMBLE RESULTS")
    print("=" * 60)
    print(f"\nEnsemble Muscle Rel L2:     {avg_metrics['rel_l2_muscle']:.4f}")
    print(f"Avg Individual Muscle Rel L2: {avg_individual:.4f}")
    print(f"Ensemble Improvement:         {avg_improvement:.4f} ({avg_improvement/avg_individual*100:.1f}%)")
    print()
    print(f"Muscle Far Rel L2:          {avg_metrics.get('rel_l2_muscle_far', 0):.4f}")
    print(f"L2 Norm Ratio (muscle):     {avg_metrics.get('l2_norm_ratio_muscle', 0):.4f}")
    print(f"Gradient Energy Ratio:      {avg_metrics.get('gradient_energy_ratio_muscle', 0):.4f}")
    print(f"Laplacian Ratio (muscle):   {avg_metrics.get('laplacian_energy_ratio_muscle', 0):.2f}")

    # Save results
    results_path = ensemble_dir / f"ensemble_results_{args.split}.json"
    results = {
        'num_models': len(models),
        'split': args.split,
        'ensemble_metrics': avg_metrics,
        'avg_individual_rel_l2': float(avg_individual),
        'ensemble_improvement': float(avg_improvement),
        'improvement_percent': float(avg_improvement / avg_individual * 100),
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save predictions if requested
    if args.save_predictions:
        pred_dir = ensemble_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        for i, pred_data in enumerate(predictions_to_save):
            np.savez(pred_dir / f"sample_{i:04d}.npz", **pred_data)
        print(f"Predictions saved to: {pred_dir}")


if __name__ == "__main__":
    main()
