#!/usr/bin/env python3
"""Zero-shot super-resolution evaluation script.

This script evaluates a trained model (especially FNO) at different resolutions
to test resolution-independent generalization.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import PotentialFieldDataset, create_data_splits
from src.data.transforms import resample_batch
from src.models.wrapper import build_model, get_device
from src.utils.masking import create_combined_mask
from src.utils.metrics import compute_all_metrics
from src.utils.visualization import save_validation_figure, create_multi_slice_figure
import matplotlib.pyplot as plt


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


def evaluate_at_resolution(
    model: torch.nn.Module,
    dataset: PotentialFieldDataset,
    target_resolution: tuple,
    device: torch.device,
    config: dict,
    num_samples: int = None,
) -> dict:
    """Evaluate model at a specific resolution.

    Args:
        model: Trained model
        dataset: Dataset to evaluate on
        target_resolution: Target resolution (D, H, W)
        device: Device to run evaluation on
        config: Configuration dict
        num_samples: Number of samples to evaluate (default: all)

    Returns:
        Dictionary with evaluation metrics
    """
    coord_range = tuple(config['grid']['coord_range'])

    num_samples = num_samples or len(dataset)
    all_metrics = []

    with torch.no_grad():
        for idx in range(min(num_samples, len(dataset))):
            # Load sample
            batch = dataset[idx]

            # Add batch dimension
            batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Get original resolution
            original_shape = batch['sigma'].shape[-3:]

            # Check if we need to resample
            if original_shape != target_resolution:
                batch = resample_batch(batch, target_resolution, coord_range)

            # Move to device
            sigma = batch['sigma'].to(device)
            source = batch['source'].to(device)
            coords = batch['coords'].to(device)
            spacing = batch['spacing'].to(device)
            target = batch['target'].to(device)

            # Forward pass
            pred = model(sigma, source, coords, spacing)

            # Create mask and compute metrics
            mask = create_combined_mask(sigma, source)
            metrics = compute_all_metrics(pred, target, mask)
            all_metrics.append(metrics)

    # Aggregate metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    std_metrics = {
        f"{key}_std": np.std([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }

    return {**avg_metrics, **std_metrics, 'num_samples': len(all_metrics)}


def visualize_super_resolution(
    model: torch.nn.Module,
    dataset: PotentialFieldDataset,
    resolutions: list,
    device: torch.device,
    config: dict,
    save_dir: str,
    sample_idx: int = 0,
):
    """Create visualizations comparing predictions at different resolutions.

    Args:
        model: Trained model
        dataset: Dataset
        resolutions: List of resolutions to compare
        device: Device
        config: Configuration
        save_dir: Directory to save visualizations
        sample_idx: Index of sample to visualize
    """
    os.makedirs(save_dir, exist_ok=True)
    coord_range = tuple(config['grid']['coord_range'])

    # Load sample
    batch = dataset[sample_idx]
    batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    results = []

    with torch.no_grad():
        for res in resolutions:
            # Resample to target resolution
            resampled_batch = resample_batch(batch, res, coord_range)

            # Move to device
            sigma = resampled_batch['sigma'].to(device)
            source = resampled_batch['source'].to(device)
            coords = resampled_batch['coords'].to(device)
            spacing = resampled_batch['spacing'].to(device)
            target = resampled_batch['target'].to(device)

            # Forward pass
            pred = model(sigma, source, coords, spacing)

            # Store results
            results.append({
                'resolution': res,
                'pred': pred[0].cpu(),
                'target': target[0].cpu(),
                'mask': create_combined_mask(sigma, source)[0].cpu(),
            })

    # Create comparison figure
    n_res = len(resolutions)
    fig, axes = plt.subplots(n_res, 3, figsize=(15, 5 * n_res))

    if n_res == 1:
        axes = axes[np.newaxis, :]

    for row, result in enumerate(results):
        res = result['resolution']
        pred = result['pred'].squeeze().numpy()
        target = result['target'].squeeze().numpy()
        mask = result['mask'].squeeze().numpy()

        # Get middle axial slice
        D = pred.shape[0]
        slice_idx = D // 2

        pred_slice = pred[slice_idx, :, :]
        target_slice = target[slice_idx, :, :]
        mask_slice = mask[slice_idx, :, :]

        # Apply mask for visualization
        pred_masked = np.where(mask_slice > 0.5, pred_slice, np.nan)
        target_masked = np.where(mask_slice > 0.5, target_slice, np.nan)
        error = np.abs(pred_slice - target_slice)
        log_error = np.log10(error + 1e-10)

        # Color scale
        vmin = np.nanmin([np.nanmin(target_masked), np.nanmin(pred_masked)])
        vmax = np.nanmax([np.nanmax(target_masked), np.nanmax(pred_masked)])

        # Plot
        im1 = axes[row, 0].imshow(target_masked.T, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        axes[row, 0].set_title(f"Ground Truth ({res[0]}x{res[1]}x{res[2]})")
        plt.colorbar(im1, ax=axes[row, 0])

        im2 = axes[row, 1].imshow(pred_masked.T, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        axes[row, 1].set_title(f"Prediction ({res[0]}x{res[1]}x{res[2]})")
        plt.colorbar(im2, ax=axes[row, 1])

        im3 = axes[row, 2].imshow(log_error.T, cmap='hot', origin='lower')
        axes[row, 2].set_title(f"Log Error ({res[0]}x{res[1]}x{res[2]})")
        plt.colorbar(im3, ax=axes[row, 2])

    fig.suptitle("Zero-Shot Super-Resolution Comparison", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"super_resolution_comparison_sample_{sample_idx}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved visualization: {save_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate model at different resolutions for zero-shot super-resolution"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--resolutions", type=str, nargs="+",
        default=["96,48,48", "48,24,24", "192,96,96"],
        help="Resolutions to evaluate at (format: D,H,W)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Number of samples to evaluate (default: all test samples)"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate visualizations"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Parse resolutions
    resolutions = []
    for res_str in args.resolutions:
        res = tuple(map(int, res_str.split(',')))
        resolutions.append(res)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint, device)
    print(f"Model backbone: {config['model']['backbone']}")

    # Update data directory in config
    config['data']['data_dir'] = args.data_dir

    # Create test dataset
    _, _, test_indices = create_data_splits(
        args.data_dir,
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        seed=args.seed,
    )

    test_dataset = PotentialFieldDataset(
        data_dir=args.data_dir,
        sample_indices=test_indices,
        coord_range=tuple(config['grid']['coord_range']),
    )
    print(f"Test dataset: {len(test_dataset)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate at each resolution
    print("\n" + "=" * 60)
    print("Zero-Shot Super-Resolution Evaluation")
    print("=" * 60)

    results = {}
    for res in resolutions:
        print(f"\nEvaluating at resolution: {res[0]}x{res[1]}x{res[2]}")

        metrics = evaluate_at_resolution(
            model, test_dataset, res, device, config, args.num_samples
        )
        results[f"{res[0]}x{res[1]}x{res[2]}"] = metrics

        print(f"  MSE: {metrics['mse']:.6f} +/- {metrics['mse_std']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f} +/- {metrics['rmse_std']:.6f}")
        print(f"  Relative L2: {metrics['relative_l2']:.6f} +/- {metrics['relative_l2_std']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f} +/- {metrics['mae_std']:.6f}")
        print(f"  Max Error: {metrics['max_error']:.6f}")

    # Save results
    results_path = os.path.join(args.output_dir, "resolution_evaluation_results.yaml")
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nResults saved to: {results_path}")

    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        vis_dir = os.path.join(args.output_dir, "visualizations")
        visualize_super_resolution(
            model, test_dataset, resolutions,
            device, config, vis_dir, sample_idx=0
        )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
