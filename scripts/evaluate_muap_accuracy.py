#!/usr/bin/env python3
"""Evaluate MUAP accuracy metrics (correlation, MSE, relative MSE) for trained models.

Compares MUAPs generated from predicted potential fields vs ground truth.
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import yaml
from scipy.stats import pearsonr
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'Model_Cylindrical' / 'python_implementation'))

from src.data.loader import get_dataloaders
from src.models.wrapper import PotentialFieldModel

# Import MUAP generation
from muap_optimized import (
    MUAPConfig, generate_muap_from_phi,
    sample_fibers_in_annulus, extract_phi_lines
)


@dataclass
class MUAPMetrics:
    """MUAP comparison metrics."""
    correlation: float
    mse: float
    rel_mse: float
    rel_l2: float
    peak_to_peak_ratio: float


def generate_muap_from_field(field: np.ndarray, spacing: np.ndarray, n_fibers: int = 50,
                              seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate MUAP from a 3D potential field.

    Args:
        field: Potential field (Z, Y, X)
        spacing: Voxel spacing (dz, dy, dx) in mm
        n_fibers: Number of fibers to sample
        seed: Random seed for fiber sampling

    Returns:
        t_ms: Time vector
        muap: MUAP waveform
    """
    config = MUAPConfig(seed=seed)

    # Sample fiber positions
    xs, ys = sample_fibers_in_annulus(
        n_fibers, field.shape,
        r_min=config.r_min, r_max=config.r_max, seed=seed
    )

    # Extract phi lines
    phi_mat = extract_phi_lines(field, xs, ys)

    # Get dz spacing (z is first dimension)
    dz_mm = float(spacing[0])

    # Generate MUAP
    result = generate_muap_from_phi(phi_mat, dz_mm, config)

    return result.t_ms, result.muap


def compute_muap_metrics(muap_pred: np.ndarray, muap_target: np.ndarray) -> MUAPMetrics:
    """Compute comparison metrics between predicted and target MUAPs."""
    # Correlation
    corr, _ = pearsonr(muap_pred, muap_target)

    # MSE
    mse = np.mean((muap_pred - muap_target) ** 2)

    # Relative MSE (normalized by target variance)
    target_var = np.var(muap_target)
    rel_mse = mse / target_var if target_var > 0 else float('inf')

    # Relative L2
    target_norm = np.linalg.norm(muap_target)
    rel_l2 = np.linalg.norm(muap_pred - muap_target) / target_norm if target_norm > 0 else float('inf')

    # Peak-to-peak ratio
    p2p_pred = np.max(muap_pred) - np.min(muap_pred)
    p2p_target = np.max(muap_target) - np.min(muap_target)
    p2p_ratio = p2p_pred / p2p_target if p2p_target > 0 else float('inf')

    return MUAPMetrics(
        correlation=float(corr),
        mse=float(mse),
        rel_mse=float(rel_mse),
        rel_l2=float(rel_l2),
        peak_to_peak_ratio=float(p2p_ratio)
    )


def load_model(exp_dir: Path, device: torch.device) -> PotentialFieldModel:
    """Load a trained model from experiment directory."""
    config_path = exp_dir / 'config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})
    fno_config = model_config.get('fno', {})
    geo_config = model_config.get('geometry_encoder', {})

    model = PotentialFieldModel(
        backbone_type=model_config.get('backbone', 'fno'),
        geometry_hidden_dim=geo_config.get('hidden_dim', 64),
        geometry_num_layers=geo_config.get('num_layers', 2),
        fno_config={
            'modes1': fno_config.get('modes1', 8),
            'modes2': fno_config.get('modes2', 8),
            'modes3': fno_config.get('modes3', 8),
            'width': fno_config.get('width', 32),
            'num_layers': fno_config.get('num_layers', 4),
            'fc_dim': fno_config.get('fc_dim', 128),
        },
        add_analytical_solution=model_config.get('add_analytical_solution', False),
        use_spacing_conditioning=config.get('spacing', {}).get('use_spacing_conditioning', True),
    )

    checkpoint_path = exp_dir / 'checkpoints' / 'best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


def evaluate_experiment(exp_dir: Path, device: torch.device,
                       n_fibers: int = 50, max_samples: int = None,
                       test_highres: bool = False) -> Dict[str, List[MUAPMetrics]]:
    """Evaluate MUAP accuracy for a single experiment.

    Returns:
        Dictionary with 'lowres' and optionally 'highres' lists of MUAPMetrics
    """
    print(f"Evaluating {exp_dir.name}...")

    model, config = load_model(exp_dir, device)

    # Get test data
    _, _, test_loader = get_dataloaders(config, seed=42)

    results = {'lowres': []}

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if max_samples and batch_idx >= max_samples:
                break

            sigma = batch['sigma'].to(device)
            source = batch['source'].to(device)
            coords = batch['coords'].to(device)
            spacing = batch['spacing'].to(device)
            target = batch['target'].to(device)

            analytical = batch.get('analytical', None)
            if analytical is not None:
                analytical = analytical.to(device)

            # Run model
            pred = model(sigma, source, coords, spacing, analytical=analytical)

            # Convert to numpy
            pred_np = pred[0, 0].cpu().numpy()  # (Z, Y, X)
            target_np = target[0, 0].cpu().numpy()
            spacing_np = spacing[0].cpu().numpy()  # (3,)

            # Generate MUAPs
            try:
                _, muap_pred = generate_muap_from_field(pred_np, spacing_np, n_fibers)
                _, muap_target = generate_muap_from_field(target_np, spacing_np, n_fibers)

                # Compute metrics
                metrics = compute_muap_metrics(muap_pred, muap_target)
                results['lowres'].append(metrics)
            except Exception as e:
                print(f"  Warning: Failed to generate MUAP for sample {batch_idx}: {e}")

    # Test on high-res if requested
    if test_highres:
        results['highres'] = []
        # Load high-res test data
        highres_config = config.copy()
        highres_config['data']['data_dirs'] = ['data/highres_test_samples']

        try:
            _, _, highres_loader = get_dataloaders(highres_config, seed=42)

            with torch.no_grad():
                for batch_idx, batch in enumerate(highres_loader):
                    if max_samples and batch_idx >= max_samples:
                        break

                    sigma = batch['sigma'].to(device)
                    source = batch['source'].to(device)
                    coords = batch['coords'].to(device)
                    spacing = batch['spacing'].to(device)
                    target = batch['target'].to(device)

                    analytical = batch.get('analytical', None)
                    if analytical is not None:
                        analytical = analytical.to(device)

                    pred = model(sigma, source, coords, spacing, analytical=analytical)

                    pred_np = pred[0, 0].cpu().numpy()
                    target_np = target[0, 0].cpu().numpy()
                    spacing_np = spacing[0].cpu().numpy()

                    try:
                        _, muap_pred = generate_muap_from_field(pred_np, spacing_np, n_fibers)
                        _, muap_target = generate_muap_from_field(target_np, spacing_np, n_fibers)

                        metrics = compute_muap_metrics(muap_pred, muap_target)
                        results['highres'].append(metrics)
                    except Exception as e:
                        print(f"  Warning: Failed to generate high-res MUAP for sample {batch_idx}: {e}")
        except Exception as e:
            print(f"  Warning: Could not load high-res data: {e}")

    return results


def aggregate_metrics(all_metrics: List[MUAPMetrics]) -> Dict[str, Tuple[float, float]]:
    """Aggregate metrics across samples, returning mean and std."""
    if not all_metrics:
        return {}

    correlations = [m.correlation for m in all_metrics]
    mses = [m.mse for m in all_metrics]
    rel_mses = [m.rel_mse for m in all_metrics]
    rel_l2s = [m.rel_l2 for m in all_metrics]
    p2p_ratios = [m.peak_to_peak_ratio for m in all_metrics]

    return {
        'correlation': (np.mean(correlations), np.std(correlations)),
        'mse': (np.mean(mses), np.std(mses)),
        'rel_mse': (np.mean(rel_mses), np.std(rel_mses)),
        'rel_l2': (np.mean(rel_l2s), np.std(rel_l2s)),
        'peak_to_peak_ratio': (np.mean(p2p_ratios), np.std(p2p_ratios)),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate MUAP accuracy metrics')
    parser.add_argument('--experiments-dir', default='experiments',
                       help='Directory containing experiment folders')
    parser.add_argument('--pattern', default='fno_analytical_mixed_seed*',
                       help='Glob pattern for experiment directories')
    parser.add_argument('--n-fibers', type=int, default=50,
                       help='Number of fibers for MUAP generation')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to evaluate per experiment')
    parser.add_argument('--test-highres', action='store_true',
                       help='Also test on high-resolution data')
    parser.add_argument('--output', default=None,
                       help='Output file for results (JSON)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Find experiment directories
    exp_base = Path(args.experiments_dir)
    exp_dirs = sorted(exp_base.glob(args.pattern))

    if not exp_dirs:
        print(f"No experiments found matching pattern: {args.pattern}")
        return

    print(f"Found {len(exp_dirs)} experiments")

    all_results = {}

    for exp_dir in exp_dirs:
        try:
            results = evaluate_experiment(
                exp_dir, device,
                n_fibers=args.n_fibers,
                max_samples=args.max_samples,
                test_highres=args.test_highres
            )
            all_results[exp_dir.name] = results

            # Print per-experiment summary
            lowres_agg = aggregate_metrics(results['lowres'])
            print(f"  Low-res: Corr={lowres_agg['correlation'][0]:.4f}±{lowres_agg['correlation'][1]:.4f}, "
                  f"Rel L2={lowres_agg['rel_l2'][0]:.4f}±{lowres_agg['rel_l2'][1]:.4f}")

            if 'highres' in results and results['highres']:
                highres_agg = aggregate_metrics(results['highres'])
                print(f"  High-res: Corr={highres_agg['correlation'][0]:.4f}±{highres_agg['correlation'][1]:.4f}, "
                      f"Rel L2={highres_agg['rel_l2'][0]:.4f}±{highres_agg['rel_l2'][1]:.4f}")
        except Exception as e:
            print(f"  Error: {e}")

    # Aggregate across all seeds
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS (Mean ± Std over all seeds)")
    print("=" * 60)

    # Collect all low-res metrics
    all_lowres = []
    all_highres = []
    for exp_name, results in all_results.items():
        all_lowres.extend(results['lowres'])
        if 'highres' in results:
            all_highres.extend(results['highres'])

    # Low-res results
    lowres_final = aggregate_metrics(all_lowres)
    print(f"\nLow-Resolution Test Set ({len(all_lowres)} samples):")
    print(f"  Correlation:       {lowres_final['correlation'][0]:.4f} ± {lowres_final['correlation'][1]:.4f}")
    print(f"  MSE:               {lowres_final['mse'][0]:.6e} ± {lowres_final['mse'][1]:.6e}")
    print(f"  Relative MSE:      {lowres_final['rel_mse'][0]:.4f} ± {lowres_final['rel_mse'][1]:.4f}")
    print(f"  Relative L2:       {lowres_final['rel_l2'][0]:.4f} ± {lowres_final['rel_l2'][1]:.4f}")
    print(f"  Peak-to-Peak Ratio: {lowres_final['peak_to_peak_ratio'][0]:.4f} ± {lowres_final['peak_to_peak_ratio'][1]:.4f}")

    if all_highres:
        highres_final = aggregate_metrics(all_highres)
        print(f"\nHigh-Resolution Test Set ({len(all_highres)} samples):")
        print(f"  Correlation:       {highres_final['correlation'][0]:.4f} ± {highres_final['correlation'][1]:.4f}")
        print(f"  MSE:               {highres_final['mse'][0]:.6e} ± {highres_final['mse'][1]:.6e}")
        print(f"  Relative MSE:      {highres_final['rel_mse'][0]:.4f} ± {highres_final['rel_mse'][1]:.4f}")
        print(f"  Relative L2:       {highres_final['rel_l2'][0]:.4f} ± {highres_final['rel_l2'][1]:.4f}")
        print(f"  Peak-to-Peak Ratio: {highres_final['peak_to_peak_ratio'][0]:.4f} ± {highres_final['peak_to_peak_ratio'][1]:.4f}")

    # Save results if output specified
    if args.output:
        import json

        def metrics_to_dict(m: MUAPMetrics) -> dict:
            return {
                'correlation': m.correlation,
                'mse': m.mse,
                'rel_mse': m.rel_mse,
                'rel_l2': m.rel_l2,
                'peak_to_peak_ratio': m.peak_to_peak_ratio
            }

        output_data = {
            'per_experiment': {
                exp_name: {
                    'lowres': [metrics_to_dict(m) for m in results['lowres']],
                    'highres': [metrics_to_dict(m) for m in results.get('highres', [])]
                }
                for exp_name, results in all_results.items()
            },
            'aggregated': {
                'lowres': {k: {'mean': v[0], 'std': v[1]} for k, v in lowres_final.items()},
                'highres': {k: {'mean': v[0], 'std': v[1]} for k, v in highres_final.items()} if all_highres else {}
            }
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Print markdown table
    print("\n" + "=" * 60)
    print("MARKDOWN TABLE")
    print("=" * 60)
    print("\n| Test Set | Correlation | Rel L2 | Rel MSE | P2P Ratio |")
    print("|----------|-------------|--------|---------|-----------|")
    print(f"| Low-res (48×48×96) | {lowres_final['correlation'][0]:.4f} ± {lowres_final['correlation'][1]:.4f} | "
          f"{lowres_final['rel_l2'][0]:.4f} ± {lowres_final['rel_l2'][1]:.4f} | "
          f"{lowres_final['rel_mse'][0]:.4f} ± {lowres_final['rel_mse'][1]:.4f} | "
          f"{lowres_final['peak_to_peak_ratio'][0]:.4f} ± {lowres_final['peak_to_peak_ratio'][1]:.4f} |")
    if all_highres:
        print(f"| High-res (96×96×192) | {highres_final['correlation'][0]:.4f} ± {highres_final['correlation'][1]:.4f} | "
              f"{highres_final['rel_l2'][0]:.4f} ± {highres_final['rel_l2'][1]:.4f} | "
              f"{highres_final['rel_mse'][0]:.4f} ± {highres_final['rel_mse'][1]:.4f} | "
              f"{highres_final['peak_to_peak_ratio'][0]:.4f} ± {highres_final['peak_to_peak_ratio'][1]:.4f} |")


if __name__ == '__main__':
    main()
