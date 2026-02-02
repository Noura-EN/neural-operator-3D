#!/usr/bin/env python3
"""
Aggregate ablation study results and generate markdown tables.

This script:
1. Scans all experiment directories
2. Loads evaluation results from each
3. Generates tables for ABLATIONS.md with per-seed and averaged results
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import argparse

PROJECT_ROOT = Path(__file__).parent.parent

# Expected experiments
MODELS = ['unet', 'fno_analytical', 'fno_no_analytical', 'fno_geom_attn', 'tfno']
TRAINING_TYPES = ['standard', 'mixed']
SEEDS = [42, 142, 242, 342, 442, 542, 642, 742, 842, 942]

# Key metrics to report
KEY_METRICS = [
    'relative_l2',
    'l2_norm_ratio',
    'gradient_energy_ratio',
    'laplacian_ratio',
    'mse_muscle',
]


def find_experiment_dir(model: str, training_type: str, seed: int) -> Path:
    """Find experiment directory for given configuration.

    Prefers directories with final_results.json, then most recent by name.
    """
    exp_name = f"{model}_{training_type}_seed{seed}"
    exp_base = PROJECT_ROOT / "experiments"

    # Find all matching directories
    matches = []
    for d in exp_base.iterdir():
        if d.is_dir() and d.name.startswith(exp_name):
            matches.append(d)
        # Also check symlinks
        elif d.is_symlink() and d.name == exp_name:
            target = d.resolve()
            if target.is_dir():
                matches.append(target)

    if not matches:
        return None

    # Prefer directories with final_results.json
    with_results = [d for d in matches if (d / "final_results.json").exists()]
    if with_results:
        # Return most recent (by directory name timestamp)
        return sorted(with_results, key=lambda d: d.name, reverse=True)[0]

    # Fall back to most recent directory
    return sorted(matches, key=lambda d: d.name, reverse=True)[0]


def load_experiment_results(exp_dir: Path) -> dict:
    """Load evaluation results from experiment directory."""
    # Try evaluation_results.json first (from evaluate_ablations.py)
    eval_path = exp_dir / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            return json.load(f)

    # Try final_results.json (from main.py training)
    final_path = exp_dir / "final_results.json"
    if final_path.exists():
        with open(final_path) as f:
            data = json.load(f)
            # Convert to expected format
            return {
                'experiment_name': exp_dir.name,
                'lowres_test_metrics': data.get('test_metrics', {}),
                'highres_test_metrics': {},  # Not available from main.py
            }

    # Fall back to metrics.json (from main.py)
    metrics_path = exp_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            data = json.load(f)
            # Convert to expected format
            return {
                'experiment_name': exp_dir.name,
                'lowres_test_metrics': data.get('test_metrics', {}),
                'highres_test_metrics': {},  # Not available from main.py
            }

    return None


def format_metric(value, precision=4):
    """Format metric value for display."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.{precision}f}"


def generate_markdown_tables(results: dict, output_file: Path):
    """Generate markdown tables from aggregated results."""
    lines = []

    lines.append("# Ablation Study Results\n")
    lines.append("## Summary\n")
    lines.append("- **Models**: UNet, FNO (with/without analytical), FNO+GeomAttn, TFNO")
    lines.append("- **Training Types**: Standard (901 samples), Mixed (901 + 50 high-res)")
    lines.append("- **Seeds**: 10 seeds per configuration for uncertainty quantification")
    lines.append("- **Test Sets**: Low-res (48x48x96) and High-res (96x96x192)\n")

    # Per-seed tables
    for seed in SEEDS:
        lines.append(f"\n## Seed {seed} Results\n")

        # Low-res test results
        lines.append("### Low-Resolution Test Set (48x48x96)\n")
        lines.append("| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |")
        lines.append("|-------|----------|--------|---------------|-------------|-----------|--------------|")

        for model in MODELS:
            for training_type in TRAINING_TYPES:
                # Skip mixed training for UNet
                if model == 'unet' and training_type == 'mixed':
                    continue

                key = f"{model}_{training_type}_seed{seed}"
                if key in results:
                    r = results[key].get('lowres_test_metrics', {})
                    lines.append(f"| {model} | {training_type} | "
                               f"{format_metric(r.get('relative_l2'))} | "
                               f"{format_metric(r.get('l2_norm_ratio'))} | "
                               f"{format_metric(r.get('gradient_energy_ratio'))} | "
                               f"{format_metric(r.get('laplacian_ratio'))} | "
                               f"{format_metric(r.get('mse_muscle'), 6)} |")
                else:
                    lines.append(f"| {model} | {training_type} | N/A | N/A | N/A | N/A | N/A |")

        # High-res test results
        lines.append("\n### High-Resolution Test Set (96x96x192)\n")
        lines.append("| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |")
        lines.append("|-------|----------|--------|---------------|-------------|-----------|--------------|")

        for model in MODELS:
            for training_type in TRAINING_TYPES:
                if model == 'unet' and training_type == 'mixed':
                    continue

                key = f"{model}_{training_type}_seed{seed}"
                if key in results:
                    r = results[key].get('highres_test_metrics', {})
                    if r:
                        lines.append(f"| {model} | {training_type} | "
                                   f"{format_metric(r.get('relative_l2'))} | "
                                   f"{format_metric(r.get('l2_norm_ratio'))} | "
                                   f"{format_metric(r.get('gradient_energy_ratio'))} | "
                                   f"{format_metric(r.get('laplacian_ratio'))} | "
                                   f"{format_metric(r.get('mse_muscle'), 6)} |")
                    else:
                        lines.append(f"| {model} | {training_type} | N/A | N/A | N/A | N/A | N/A |")
                else:
                    lines.append(f"| {model} | {training_type} | N/A | N/A | N/A | N/A | N/A |")

    # Aggregated results (mean ± std)
    lines.append("\n\n## Aggregated Results (Mean ± Std over 10 seeds)\n")

    # Aggregate by model and training type
    aggregated = defaultdict(lambda: defaultdict(list))

    for model in MODELS:
        for training_type in TRAINING_TYPES:
            if model == 'unet' and training_type == 'mixed':
                continue

            for seed in SEEDS:
                key = f"{model}_{training_type}_seed{seed}"
                if key in results:
                    config_key = f"{model}_{training_type}"
                    for test_type in ['lowres_test_metrics', 'highres_test_metrics']:
                        r = results[key].get(test_type, {})
                        for metric in KEY_METRICS:
                            val = r.get(metric)
                            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                                aggregated[config_key][f"{test_type}_{metric}"].append(val)

    # Low-res aggregated table
    lines.append("### Low-Resolution Test Set (Aggregated)\n")
    lines.append("| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian |")
    lines.append("|-------|----------|--------|---------------|-------------|-----------|")

    for model in MODELS:
        for training_type in TRAINING_TYPES:
            if model == 'unet' and training_type == 'mixed':
                continue

            config_key = f"{model}_{training_type}"
            agg = aggregated[config_key]

            def fmt_agg(metric_key):
                vals = agg.get(f"lowres_test_metrics_{metric_key}", [])
                if not vals:
                    return "N/A"
                mean = np.mean(vals)
                std = np.std(vals)
                return f"{mean:.4f} ± {std:.4f}"

            lines.append(f"| {model} | {training_type} | "
                       f"{fmt_agg('relative_l2')} | "
                       f"{fmt_agg('l2_norm_ratio')} | "
                       f"{fmt_agg('gradient_energy_ratio')} | "
                       f"{fmt_agg('laplacian_ratio')} |")

    # High-res aggregated table
    lines.append("\n### High-Resolution Test Set (Aggregated)\n")
    lines.append("| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian |")
    lines.append("|-------|----------|--------|---------------|-------------|-----------|")

    for model in MODELS:
        for training_type in TRAINING_TYPES:
            if model == 'unet' and training_type == 'mixed':
                continue

            config_key = f"{model}_{training_type}"
            agg = aggregated[config_key]

            def fmt_agg(metric_key):
                vals = agg.get(f"highres_test_metrics_{metric_key}", [])
                if not vals:
                    return "N/A"
                mean = np.mean(vals)
                std = np.std(vals)
                return f"{mean:.4f} ± {std:.4f}"

            lines.append(f"| {model} | {training_type} | "
                       f"{fmt_agg('relative_l2')} | "
                       f"{fmt_agg('l2_norm_ratio')} | "
                       f"{fmt_agg('gradient_energy_ratio')} | "
                       f"{fmt_agg('laplacian_ratio')} |")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Results written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate ablation results")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "ABLATION_RESULTS.md"),
                        help="Output markdown file")
    parser.add_argument("--run-evaluation", action="store_true",
                        help="Run evaluation on experiments missing evaluation_results.json")
    args = parser.parse_args()

    print("Scanning experiment directories...")

    # Collect all results
    results = {}
    missing_eval = []

    for model in MODELS:
        for training_type in TRAINING_TYPES:
            if model == 'unet' and training_type == 'mixed':
                continue

            for seed in SEEDS:
                exp_dir = find_experiment_dir(model, training_type, seed)
                if exp_dir is None:
                    print(f"  Not found: {model}_{training_type}_seed{seed}")
                    continue

                exp_results = load_experiment_results(exp_dir)
                if exp_results is None:
                    print(f"  No results: {exp_dir.name}")
                    missing_eval.append(exp_dir)
                    continue

                key = f"{model}_{training_type}_seed{seed}"
                results[key] = exp_results
                print(f"  Loaded: {key}")

    print(f"\nLoaded {len(results)} experiments")

    if missing_eval and args.run_evaluation:
        print(f"\nRunning evaluation on {len(missing_eval)} experiments...")
        import subprocess
        for exp_dir in missing_eval:
            cmd = ["python3", str(PROJECT_ROOT / "scripts" / "evaluate_ablations.py"),
                   "--experiment-dir", str(exp_dir)]
            subprocess.run(cmd, capture_output=True)

        # Reload results
        for exp_dir in missing_eval:
            exp_results = load_experiment_results(exp_dir)
            if exp_results:
                # Parse key from exp_dir name
                name = exp_dir.name
                for model in MODELS:
                    for training_type in TRAINING_TYPES:
                        for seed in SEEDS:
                            pattern = f"{model}_{training_type}_seed{seed}"
                            if name.startswith(pattern):
                                results[pattern] = exp_results
                                break

    # Generate tables
    generate_markdown_tables(results, Path(args.output))


if __name__ == "__main__":
    main()
