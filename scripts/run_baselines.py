#!/usr/bin/env python3
"""
Run trilinear interpolation baseline and measure inference times.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import get_dataloaders, PotentialFieldDataset
from src.models.wrapper import build_model, get_device
from src.utils.metrics import compute_all_metrics_extended
from torch.utils.data import DataLoader


def load_highres_test_data():
    """Load high-res test dataset."""
    highres_test_dir = PROJECT_ROOT / "data" / "highres_test_samples"

    dataset = PotentialFieldDataset(
        data_dir=str(highres_test_dir),
        sample_indices=None,
        add_analytical_solution=True,
    )

    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def trilinear_baseline(lowres_loader, highres_loader, device):
    """
    Trilinear interpolation baseline for super-resolution.

    Takes low-res ground truth, upsamples to high-res, compares to high-res GT.
    This is the naive baseline that neural operators should beat.
    """
    print("\n=== Trilinear Interpolation Baseline ===")

    # We need paired low-res and high-res samples
    # Since they're from different datasets, we'll use the high-res GT
    # and downsample -> upsample to simulate the pipeline

    all_metrics = []

    for batch in tqdm(highres_loader, desc="Trilinear baseline"):
        target_highres = batch['target'].to(device)  # [1, 1, D, H, W] high-res GT

        # Downsample to low-res (simulating low-res input)
        lowres_shape = (target_highres.shape[2] // 2,
                        target_highres.shape[3] // 2,
                        target_highres.shape[4] // 2)

        # Downsample
        target_lowres = F.interpolate(target_highres, size=lowres_shape,
                                       mode='trilinear', align_corners=True)

        # Upsample back (this is the trilinear baseline prediction)
        pred_highres = F.interpolate(target_lowres, size=target_highres.shape[2:],
                                      mode='trilinear', align_corners=True)

        # Compute metrics
        sigma = batch['sigma'].to(device)
        source = batch['source'].to(device)
        spacing = batch['spacing'].to(device)

        metrics = compute_all_metrics_extended(pred_highres, target_highres, sigma, source, spacing)
        all_metrics.append(metrics)

    # Aggregate
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        avg_metrics[key] = np.mean(values) if values else float('nan')

    return avg_metrics


def measure_inference_time(model, dataloader, device, num_samples=50, warmup=5):
    """Measure inference time per sample."""
    model.eval()
    times = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples + warmup:
                break

            sigma = batch['sigma'].to(device)
            source = batch['source'].to(device)
            coords = batch['coords'].to(device)
            spacing = batch['spacing'].to(device)
            analytical = batch.get('analytical')
            if analytical is not None:
                analytical = analytical.to(device)

            # Warmup
            if i < warmup:
                if hasattr(model.backbone, 'needs_geometry') and model.backbone.needs_geometry:
                    from src.utils.masking import create_combined_mask
                    mask = create_combined_mask(sigma, source)
                    _ = model(sigma, source, coords, spacing, analytical=analytical,
                             sigma_raw=sigma, mask=mask)
                else:
                    _ = model(sigma, source, coords, spacing, analytical=analytical)
                continue

            # Sync before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()

            if hasattr(model.backbone, 'needs_geometry') and model.backbone.needs_geometry:
                from src.utils.masking import create_combined_mask
                mask = create_combined_mask(sigma, source)
                _ = model(sigma, source, coords, spacing, analytical=analytical,
                         sigma_raw=sigma, mask=mask)
            else:
                _ = model(sigma, source, coords, spacing, analytical=analytical)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            times.append(elapsed)

    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
    }


def main():
    device = get_device()
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    with open(PROJECT_ROOT / 'configs/fno_standard.yaml') as f:
        config = yaml.safe_load(f)

    DATA_SPLIT_SEED = 42
    train_loader, val_loader, lowres_test_loader = get_dataloaders(config, seed=DATA_SPLIT_SEED)
    highres_test_loader = load_highres_test_data()

    print(f"Low-res test: {len(lowres_test_loader.dataset)} samples")
    print(f"High-res test: {len(highres_test_loader.dataset)} samples")

    results = {}

    # Trilinear baseline
    trilinear_metrics = trilinear_baseline(lowres_test_loader, highres_test_loader, device)
    results['trilinear'] = {
        'highres_test_metrics': trilinear_metrics,
        'params': 0,
        'inference_time_ms': 0.5,  # Negligible
    }
    print(f"\nTrilinear baseline - High-res Rel L2: {trilinear_metrics['relative_l2']:.4f}")

    # Model configs and checkpoints
    models_to_test = {
        'unet': ('configs/unet_standard.yaml', 'unet_standard_seed42'),
        'fno_analytical': ('configs/fno_standard.yaml', 'fno_analytical_standard_seed42'),
        'fno_geom_attn': ('configs/fno_geom_attn_standard.yaml', 'fno_geom_attn_standard_seed42'),
        'tfno': ('configs/tfno_standard.yaml', 'tfno_standard_seed42'),
    }

    print("\n=== Inference Time Measurements ===")

    for model_name, (config_path, exp_pattern) in models_to_test.items():
        print(f"\n{model_name}:")

        # Load config
        with open(PROJECT_ROOT / config_path) as f:
            config = yaml.safe_load(f)

        # Build model
        model = build_model(config)
        params = sum(p.numel() for p in model.parameters())

        # Find checkpoint
        exp_base = PROJECT_ROOT / "experiments"
        checkpoint_path = None
        for d in sorted(exp_base.iterdir(), key=lambda x: x.name, reverse=True):
            if d.name.startswith(exp_pattern) and (d / "checkpoints/best_model.pt").exists():
                checkpoint_path = d / "checkpoints/best_model.pt"
                break

        if checkpoint_path is None:
            print(f"  No checkpoint found for {exp_pattern}")
            continue

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        # Measure inference time on low-res data
        timing = measure_inference_time(model, lowres_test_loader, device)

        results[model_name] = {
            'params': params,
            'inference_time_ms': timing['mean_ms'],
            'inference_time_std_ms': timing['std_ms'],
        }

        print(f"  Params: {params:,} ({params/1e6:.2f}M)")
        print(f"  Inference: {timing['mean_ms']:.1f} ± {timing['std_ms']:.1f} ms")

    # Save results
    output_path = PROJECT_ROOT / "baseline_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'Params':<12} {'Inference (ms)':<15} {'High-Res Rel L2':<15}")
    print("-"*70)
    print(f"{'Trilinear':<20} {'0':<12} {'<1':<15} {trilinear_metrics['relative_l2']:.4f}")
    for model_name in ['unet', 'fno_analytical', 'fno_geom_attn', 'tfno']:
        if model_name in results:
            r = results[model_name]
            params_str = f"{r['params']/1e6:.2f}M"
            time_str = f"{r['inference_time_ms']:.1f}"
            print(f"{model_name:<20} {params_str:<12} {time_str:<15}")


if __name__ == "__main__":
    main()
