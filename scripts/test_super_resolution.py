#!/usr/bin/env python3
"""Test zero-shot super-resolution capabilities of trained FNO models.

This script tests a model trained on 48x48x96 resolution on high-resolution
96x96x192 data without any interpolation. The FNO should generalize to
different resolutions due to its Fourier-space operations.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import PotentialFieldDataset
from src.models.wrapper import build_model, get_device
from src.utils.metrics import compute_all_metrics_extended
from src.utils.visualization import create_comprehensive_visualization


class HighResDataset(PotentialFieldDataset):
    """Dataset for loading high-resolution data for super-resolution testing."""

    def __init__(
        self,
        data_dir,
        holdout_indices,
        add_analytical_solution=False,
    ):
        """Initialize high-res dataset with specific holdout indices.

        Args:
            data_dir: Directory containing high-res samples
            holdout_indices: List of sample indices to use
            add_analytical_solution: If True, compute analytical solution
        """
        self.data_dir = Path(data_dir)
        self.add_spacing_channels = False
        self.add_analytical_solution = add_analytical_solution

        # Find all sample files and select holdout indices
        all_files = sorted(self.data_dir.glob("sample_*.npz"))
        self.sample_files = [all_files[i] for i in holdout_indices if i < len(all_files)]

        if len(self.sample_files) == 0:
            raise ValueError(f"No sample files found in {data_dir} for given indices")


def test_super_resolution(
    checkpoint_path: str,
    config_path: str,
    highres_data_dir: str,
    holdout_info_path: str,
    output_dir: str,
):
    """Test super-resolution capabilities.

    Args:
        checkpoint_path: Path to trained model checkpoint
        config_path: Path to model config
        highres_data_dir: Directory containing high-res samples
        holdout_info_path: Path to preprocessing info with holdout indices
        output_dir: Directory to save results
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load holdout info
    with open(holdout_info_path, 'r') as f:
        holdout_info = json.load(f)

    holdout_indices = holdout_info['holdout_indices']
    print(f"Testing on {len(holdout_indices)} holdout high-res samples")

    # Build model
    print("Building model...")
    model = build_model(config)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Check if model uses analytical solution
    add_analytical = config.get('model', {}).get('add_analytical_solution', False)

    # Create high-res dataset
    print(f"Loading high-res data from {highres_data_dir}")
    highres_dataset = HighResDataset(
        data_dir=highres_data_dir,
        holdout_indices=holdout_indices,
        add_analytical_solution=add_analytical,
    )
    print(f"Loaded {len(highres_dataset)} high-res samples")

    # Test on first sample to verify dimensions
    sample = highres_dataset[0]
    print(f"\nHigh-res sample shapes:")
    print(f"  sigma: {sample['sigma'].shape}")
    print(f"  source: {sample['source'].shape}")
    print(f"  target: {sample['target'].shape}")
    print(f"  coords: {sample['coords'].shape}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Run inference on all samples
    all_metrics = []

    print("\nRunning super-resolution inference...")
    with torch.no_grad():
        for idx in tqdm(range(len(highres_dataset))):
            batch = highres_dataset[idx]

            # Add batch dimension and move to device
            sigma = batch['sigma'].unsqueeze(0).to(device)
            source = batch['source'].unsqueeze(0).to(device)
            coords = batch['coords'].unsqueeze(0).to(device)
            spacing = batch['spacing'].unsqueeze(0).to(device)
            target = batch['target'].unsqueeze(0).to(device)

            # Handle analytical solution if model expects it
            analytical = None
            if add_analytical and 'analytical' in batch:
                analytical = batch['analytical'].unsqueeze(0).to(device)

            # Forward pass
            pred = model(sigma, source, coords, spacing, analytical=analytical)

            # Compute metrics
            metrics = compute_all_metrics_extended(pred, target, sigma, source, spacing)
            all_metrics.append(metrics)

            # Save visualization for first 5 samples
            if idx < 5:
                create_comprehensive_visualization(
                    pred[0].cpu(), target[0].cpu(),
                    source=source[0].cpu(),
                    mask=None,
                    title_prefix=f"Super-Res Sample {idx}",
                    save_dir=vis_dir,
                    sample_idx=idx,
                )

    # Aggregate metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        avg_metrics[key] = float(np.mean(values)) if values else float('nan')

    # Print results
    print("\n" + "="*60)
    print("SUPER-RESOLUTION TEST RESULTS")
    print("="*60)
    print(f"Resolution: 96x96x192 (2x each dimension)")
    print(f"Number of samples: {len(highres_dataset)}")
    print()
    print("Key Metrics:")
    print(f"  Relative L2:           {avg_metrics['relative_l2']:.4f}")
    print(f"  L2 Norm Ratio:         {avg_metrics['l2_norm_ratio']:.4f}")
    print(f"  Gradient Energy Ratio: {avg_metrics['gradient_energy_ratio']:.4f}")
    print(f"  Laplacian Ratio:       {avg_metrics['laplacian_energy_ratio']:.4f}")
    print(f"  RMSE:                  {avg_metrics['rmse']:.6f}")
    print()
    if 'rel_l2_muscle' in avg_metrics:
        print("Muscle Region:")
        print(f"  Rel L2 Muscle:         {avg_metrics['rel_l2_muscle']:.4f}")
    print("="*60)

    # Save results
    results = {
        "experiment": "super_resolution_test",
        "model_checkpoint": checkpoint_path,
        "config": config_path,
        "resolution": "96x96x192",
        "num_samples": len(highres_dataset),
        "metrics": avg_metrics,
    }

    results_path = os.path.join(output_dir, "super_resolution_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Test FNO super-resolution capabilities")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config",
    )
    parser.add_argument(
        "--highres-dir",
        type=str,
        default="data/voxel_96_96_192",
        help="Directory containing high-resolution samples",
    )
    parser.add_argument(
        "--holdout-info",
        type=str,
        default="data/downsampled_highres/preprocessing_info.json",
        help="Path to preprocessing info with holdout indices",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/super_resolution",
        help="Directory to save results",
    )
    args = parser.parse_args()

    test_super_resolution(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        highres_data_dir=args.highres_dir,
        holdout_info_path=args.holdout_info,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
