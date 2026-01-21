#!/usr/bin/env python3
"""Run ablation experiments with config overrides."""

import argparse
import os
import sys
import yaml
import subprocess
from pathlib import Path
from copy import deepcopy


def deep_update(base_dict, update_dict):
    """Deep update a nested dictionary."""
    result = deepcopy(base_dict)
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def run_experiment(base_config_path: str, experiment_name: str, overrides: dict, seed: int = 42):
    """Run a single experiment with config overrides."""
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Apply overrides
    config = deep_update(config, overrides)

    # Create temporary config file
    exp_config_dir = Path("configs/ablations")
    exp_config_dir.mkdir(parents=True, exist_ok=True)
    exp_config_path = exp_config_dir / f"{experiment_name}.yaml"

    with open(exp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"Config overrides: {overrides}")
    print(f"{'='*60}\n")

    # Run training
    cmd = [
        sys.executable, "scripts/main.py",
        "--config", str(exp_config_path),
        "--seed", str(seed),
        "--experiment-name", experiment_name,
        "--description", f"Ablation: {experiment_name}"
    ]

    result = subprocess.run(cmd, capture_output=False)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--base-config", type=str, default="configs/config.yaml",
                        help="Base configuration file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name or 'all' to run all ablations")
    args = parser.parse_args()

    # Define ablation experiments
    ablations = {
        # Baseline with new metrics
        "baseline_new_metrics": {},

        # Singularity mask ablation
        "no_singularity_mask": {
            "experiment": {"use_singularity_mask": False}
        },

        # Muscle mask ablation (with muscle mask = True)
        "with_muscle_mask": {
            "experiment": {"use_muscle_mask": True}
        },

        # Loss type ablations
        "normalized_loss": {
            "loss": {"type": "normalized"}
        },
        "logcosh_loss": {
            "loss": {"type": "logcosh"}
        },

        # MSE + log-cosh hybrid (noise reduction)
        "mse_logcosh_0.05": {
            "loss": {"logcosh_weight": 0.05}
        },
        "mse_logcosh_0.1": {
            "loss": {"logcosh_weight": 0.1}
        },
        "mse_logcosh_0.2": {
            "loss": {"logcosh_weight": 0.2}
        },

        # Total variation regularizer (smoothness)
        "tv_0.001": {
            "loss": {"tv_weight": 0.001}
        },
        "tv_0.01": {
            "loss": {"tv_weight": 0.01}
        },
        "tv_0.1": {
            "loss": {"tv_weight": 0.1}
        },

        # PDE residual loss sweep
        "pde_residual_0.1": {
            "loss": {"pde_weight": 0.1}
        },
        "pde_residual_0.5": {
            "loss": {"pde_weight": 0.5}
        },
        "pde_residual_1.0": {
            "loss": {"pde_weight": 1.0}
        },

        # Spectral smoothing
        "spectral_threshold_0.01": {
            "loss": {"spectral_weight": 0.01, "spectral_mode": "threshold"}
        },
        "spectral_weighted_0.01": {
            "loss": {"spectral_weight": 0.01, "spectral_mode": "weighted"}
        },

        # Combined experiments
        "normalized_spectral": {
            "loss": {"type": "normalized", "spectral_weight": 0.01}
        },

        # MSE + log-cosh + TV (combined noise reduction)
        "mse_logcosh_tv": {
            "loss": {"logcosh_weight": 0.1, "tv_weight": 0.01}
        },

        # Fourier mode ablations (noise reduction via frequency limiting)
        "fno_modes_6": {
            "model": {"fno": {"modes1": 6, "modes2": 6, "modes3": 6}}
        },
        "fno_modes_4": {
            "model": {"fno": {"modes1": 4, "modes2": 4, "modes3": 4}}
        },
        "fno_modes_2": {
            "model": {"fno": {"modes1": 2, "modes2": 2, "modes3": 2}}
        },
    }

    if args.experiment == "all":
        for exp_name, overrides in ablations.items():
            run_experiment(args.base_config, exp_name, overrides, args.seed)
    elif args.experiment in ablations:
        run_experiment(args.base_config, args.experiment, ablations[args.experiment], args.seed)
    else:
        print(f"Unknown experiment: {args.experiment}")
        print(f"Available experiments: {list(ablations.keys())}")
        sys.exit(1)


if __name__ == "__main__":
    main()
