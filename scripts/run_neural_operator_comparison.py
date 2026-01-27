#!/usr/bin/env python3
"""Run comparison experiments across different neural operator architectures.

This script trains and evaluates multiple neural operator architectures:
- FNO (Fourier Neural Operator) - baseline
- TFNO (Factorized Fourier Neural Operator)
- U-NO (U-shaped Neural Operator)
- DeepONet (Deep Operator Network)
- LSM (Latent Spectral Model)
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Neural operator configurations
NEURAL_OPERATORS = {
    "fno": {
        "description": "Fourier Neural Operator (baseline)",
        "backbone": "fno",
        "config_overrides": {},
    },
    "tfno": {
        "description": "Factorized Fourier Neural Operator",
        "backbone": "tfno",
        "config_overrides": {},
    },
    "uno": {
        "description": "U-shaped Neural Operator",
        "backbone": "uno",
        "config_overrides": {
            "model.uno.base_width": 32,
            "model.uno.depth": 3,
            "model.uno.base_modes": 8,
        },
    },
    "deeponet": {
        "description": "Deep Operator Network",
        "backbone": "deeponet",
        "config_overrides": {
            "model.deeponet.hidden_dim": 128,
            "model.deeponet.num_basis": 64,
            "model.deeponet.branch_layers": 4,
            "model.deeponet.trunk_layers": 4,
        },
    },
    "lsm": {
        "description": "Latent Spectral Model",
        "backbone": "lsm",
        "config_overrides": {
            "model.lsm.latent_dim": 32,
            "model.lsm.num_layers": 4,
            "model.lsm.hidden_dim": 64,
        },
    },
}


def create_config_file(base_config_path: str, output_path: str, operator_name: str, operator_config: dict):
    """Create a config file for a specific neural operator."""
    import yaml

    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set backbone type
    config['model']['backbone'] = operator_config['backbone']

    # Apply any config overrides
    for key, value in operator_config.get('config_overrides', {}).items():
        parts = key.split('.')
        d = config
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    # Update experiment name
    config['logging']['experiment_name'] = f"neural_operator_{operator_name}"

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return output_path


def run_experiment(operator_name: str, config_path: str, dry_run: bool = False):
    """Run a single experiment."""
    experiment_name = f"neural_operator_{operator_name}"

    cmd = [
        sys.executable, "scripts/main.py",
        "--config", config_path,
        "--experiment-name", experiment_name,
    ]

    print(f"\n{'='*60}")
    print(f"Running: {operator_name}")
    print(f"Config: {config_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    if dry_run:
        print("[DRY RUN] Would execute command")
        return None

    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode


def collect_results(experiments_dir: str):
    """Collect results from all experiments."""
    results = {}

    for exp_dir in Path(experiments_dir).glob("neural_operator_*"):
        results_file = exp_dir / "final_results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)

            # Extract operator name from experiment name
            exp_name = exp_dir.name
            for op_name in NEURAL_OPERATORS.keys():
                if op_name in exp_name:
                    results[op_name] = {
                        "experiment_dir": str(exp_dir),
                        "test_metrics": data.get("test_metrics", {}),
                        "best_epoch": data.get("best_epoch"),
                        "total_epochs": data.get("total_epochs"),
                    }
                    break

    return results


def print_results_table(results: dict):
    """Print a comparison table of results."""
    print("\n" + "=" * 100)
    print("NEURAL OPERATOR COMPARISON RESULTS")
    print("=" * 100)

    # Header
    print(f"{'Operator':<15} {'Rel L2':<10} {'L2 Norm Ratio':<15} {'Grad Energy':<12} {'Laplacian':<12} {'Epochs':<8}")
    print("-" * 100)

    # Sort by relative L2
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get('test_metrics', {}).get('relative_l2', float('inf'))
    )

    for op_name, data in sorted_results:
        metrics = data.get('test_metrics', {})
        rel_l2 = metrics.get('relative_l2', float('nan'))
        l2_ratio = metrics.get('l2_norm_ratio', float('nan'))
        grad_ratio = metrics.get('gradient_energy_ratio', float('nan'))
        lap_ratio = metrics.get('laplacian_energy_ratio', float('nan'))
        epochs = data.get('best_epoch', '?')

        desc = NEURAL_OPERATORS.get(op_name, {}).get('description', op_name)
        print(f"{op_name:<15} {rel_l2:<10.4f} {l2_ratio:<15.4f} {grad_ratio:<12.4f} {lap_ratio:<12.4f} {epochs:<8}")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Compare neural operator architectures")
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/config.yaml",
        help="Base configuration file",
    )
    parser.add_argument(
        "--operators",
        type=str,
        nargs="+",
        default=list(NEURAL_OPERATORS.keys()),
        choices=list(NEURAL_OPERATORS.keys()),
        help="Which operators to test",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Only collect and display results from existing experiments",
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="experiments",
        help="Directory containing experiments",
    )
    args = parser.parse_args()

    if args.results_only:
        results = collect_results(args.experiments_dir)
        if results:
            print_results_table(results)
        else:
            print("No results found.")
        return

    # Create temp config directory
    config_dir = Path("configs/neural_operators")
    config_dir.mkdir(exist_ok=True)

    # Run experiments
    for op_name in args.operators:
        op_config = NEURAL_OPERATORS[op_name]

        # Create config file
        config_path = config_dir / f"config_{op_name}.yaml"
        create_config_file(args.base_config, str(config_path), op_name, op_config)

        # Run experiment
        returncode = run_experiment(op_name, str(config_path), args.dry_run)

        if returncode != 0 and returncode is not None:
            print(f"WARNING: Experiment {op_name} failed with return code {returncode}")

    # Collect and display results
    if not args.dry_run:
        print("\n\nCollecting results...")
        results = collect_results(args.experiments_dir)
        if results:
            print_results_table(results)


if __name__ == "__main__":
    main()
