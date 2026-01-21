#!/usr/bin/env python3
"""
Experiment runner with automatic summary tracking.

This script:
1. Runs an experiment with specified config overrides
2. Updates the global experiment summary
3. Provides utilities for ablation studies
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import yaml


EXPERIMENTS_DIR = Path("experiments")
SUMMARY_FILE = EXPERIMENTS_DIR / "experiment_summary.json"


def load_base_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the base configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, overrides: Dict[str, Any]) -> dict:
    """Apply nested overrides to config.

    Overrides use dot notation: "model.fno.width" -> config['model']['fno']['width']
    """
    for key, value in overrides.items():
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    return config


def save_experiment_config(config: dict, exp_name: str) -> str:
    """Save experiment-specific config and return path."""
    config_dir = EXPERIMENTS_DIR / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / f"{exp_name}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return str(config_path)


def load_summary() -> dict:
    """Load or create experiment summary."""
    if SUMMARY_FILE.exists():
        with open(SUMMARY_FILE, 'r') as f:
            return json.load(f)
    return {
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "experiments": [],
        "best_experiment": None,
    }


def save_summary(summary: dict):
    """Save experiment summary."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    summary["last_updated"] = datetime.now().isoformat()
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2)


def update_summary_with_results(exp_name: str, exp_dir: str, description: str):
    """Update summary with results from completed experiment."""
    summary = load_summary()

    # Load results from experiment
    results_path = Path(exp_dir) / "final_results.json"
    if not results_path.exists():
        print(f"Warning: No results found at {results_path}")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    # Extract key metrics
    test_metrics = results.get('test_metrics', {})
    val_metrics = results.get('val_metrics', {})

    entry = {
        "name": exp_name,
        "description": description,
        "timestamp": datetime.now().isoformat(),
        "path": str(exp_dir),
        "status": "completed",
        "epochs_trained": results.get('total_epochs', 0),
        "best_epoch": results.get('best_epoch', 0),
        "early_stopped": results.get('early_stopped', False),
        "test_metrics": {
            "loss": test_metrics.get('loss', float('nan')),
            "rmse": test_metrics.get('rmse', float('nan')),
            "relative_l2": test_metrics.get('relative_l2', float('nan')),
            "mae": test_metrics.get('mae', float('nan')),
            "mse_muscle": test_metrics.get('mse_muscle', float('nan')),
            "mse_non_muscle": test_metrics.get('mse_non_muscle', float('nan')),
            "mse_near_singularity": test_metrics.get('mse_near_singularity', float('nan')),
            "mse_far_singularity": test_metrics.get('mse_far_singularity', float('nan')),
        },
        "val_metrics": {
            "loss": val_metrics.get('loss', float('nan')),
            "relative_l2": val_metrics.get('relative_l2', float('nan')),
        },
    }

    # Check if this is the best experiment
    if summary["best_experiment"] is None:
        summary["best_experiment"] = exp_name
    else:
        # Compare by test relative_l2 (lower is better)
        best_exp = next((e for e in summary["experiments"]
                        if e["name"] == summary["best_experiment"]), None)
        if best_exp:
            best_rel_l2 = best_exp.get("test_metrics", {}).get("relative_l2", float('inf'))
            current_rel_l2 = entry["test_metrics"].get("relative_l2", float('inf'))
            if current_rel_l2 < best_rel_l2:
                summary["best_experiment"] = exp_name

    summary["experiments"].append(entry)
    save_summary(summary)

    print(f"\nExperiment summary updated: {SUMMARY_FILE}")


def print_summary_table():
    """Print a formatted summary table of all experiments."""
    summary = load_summary()

    if not summary["experiments"]:
        print("No experiments recorded yet.")
        return

    print("\n" + "="*120)
    print("EXPERIMENT SUMMARY")
    print("="*120)

    # Header
    print(f"{'Name':<40} {'Rel L2':<10} {'RMSE':<10} {'Epochs':<8} {'Early':<6} {'Best':<5}")
    print("-"*120)

    for exp in summary["experiments"]:
        is_best = exp["name"] == summary.get("best_experiment")
        marker = "*" if is_best else " "

        test = exp.get("test_metrics", {})
        rel_l2 = test.get("relative_l2", float('nan'))
        rmse = test.get("rmse", float('nan'))
        epochs = exp.get("epochs_trained", 0)
        early = "Yes" if exp.get("early_stopped") else "No"

        rel_l2_str = f"{rel_l2:.6f}" if rel_l2 == rel_l2 else "N/A"
        rmse_str = f"{rmse:.6f}" if rmse == rmse else "N/A"

        print(f"{marker}{exp['name']:<39} {rel_l2_str:<10} {rmse_str:<10} {epochs:<8} {early:<6} {exp.get('best_epoch', 0):<5}")

    print("="*120)
    print(f"Best experiment: {summary.get('best_experiment', 'None')}")
    print(f"Total experiments: {len(summary['experiments'])}")
    print()


def run_experiment(
    name: str,
    description: str,
    overrides: Optional[Dict[str, Any]] = None,
    base_config: str = "configs/config.yaml",
) -> Optional[str]:
    """Run a single experiment.

    Args:
        name: Experiment name
        description: Description for tracking
        overrides: Config overrides (dot notation)
        base_config: Path to base config

    Returns:
        Path to experiment directory if successful
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {name}")
    print(f"Description: {description}")
    print(f"{'='*60}")

    # Load and modify config
    config = load_base_config(base_config)
    if overrides:
        config = apply_overrides(config, overrides)
        print(f"Config overrides: {overrides}")

    # Save experiment config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_config_name = f"{name}_{timestamp}"
    config_path = save_experiment_config(config, exp_config_name)

    # Run training
    cmd = [
        "python3", "scripts/main.py",
        "--config", config_path,
        "--experiment-name", name,
        "--description", description,
    ]

    print(f"\nCommand: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, text=True)

        # Find the experiment directory (most recent matching name)
        exp_dirs = sorted(EXPERIMENTS_DIR.glob(f"{name}_*"), key=lambda p: p.stat().st_mtime)
        if exp_dirs:
            exp_dir = str(exp_dirs[-1])
            update_summary_with_results(exp_config_name, exp_dir, description)
            return exp_dir

    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with return code {e.returncode}")
        return None

    return None


# Predefined experiment configurations
EXPERIMENTS = {
    # Baselines
    "baseline_fno": {
        "description": "Baseline FNO (width=32, layers=4, modes=8)",
        "overrides": {
            "model.backbone": "fno",
        }
    },
    "baseline_unet": {
        "description": "Baseline UNet (channels=32, depth=4)",
        "overrides": {
            "model.backbone": "unet",
        }
    },

    # FNO ablations
    "fno_medium": {
        "description": "FNO with width=64 (increased capacity)",
        "overrides": {
            "model.backbone": "fno",
            "model.fno.width": 64,
        }
    },
    "fno_deep": {
        "description": "FNO with 6 layers (increased depth)",
        "overrides": {
            "model.backbone": "fno",
            "model.fno.num_layers": 6,
        }
    },
    "fno_high_modes": {
        "description": "FNO with modes=12 (more frequency components)",
        "overrides": {
            "model.backbone": "fno",
            "model.fno.modes1": 12,
            "model.fno.modes2": 12,
            "model.fno.modes3": 12,
        }
    },
    "fno_wide": {
        "description": "FNO with width=48 (moderate increase)",
        "overrides": {
            "model.backbone": "fno",
            "model.fno.width": 48,
        }
    },

    # Loss ablations
    "no_muscle_mask": {
        "description": "FNO without muscle mask weighting",
        "overrides": {
            "model.backbone": "fno",
            "experiment.use_muscle_mask": False,
        }
    },
    "high_grad_weight": {
        "description": "FNO with higher gradient loss weight (0.5)",
        "overrides": {
            "model.backbone": "fno",
            "training.loss_weights.pde": 0.5,
        }
    },
    "low_grad_weight": {
        "description": "FNO with lower gradient loss weight (0.01)",
        "overrides": {
            "model.backbone": "fno",
            "training.loss_weights.pde": 0.01,
        }
    },

    # UNet ablations
    "unet_small": {
        "description": "UNet with base_channels=16 (smaller)",
        "overrides": {
            "model.backbone": "unet",
            "model.unet.base_channels": 16,
        }
    },
    "unet_deep": {
        "description": "UNet with depth=5 (deeper)",
        "overrides": {
            "model.backbone": "unet",
            "model.unet.depth": 5,
        }
    },
    "unet_wide": {
        "description": "UNet with base_channels=64 (wider)",
        "overrides": {
            "model.backbone": "unet",
            "model.unet.base_channels": 64,
        }
    },
}


def main():
    parser = argparse.ArgumentParser(description="Run experiments with tracking")
    parser.add_argument("--experiment", "-e", type=str,
                        help=f"Predefined experiment name. Options: {list(EXPERIMENTS.keys())}")
    parser.add_argument("--name", type=str, help="Custom experiment name")
    parser.add_argument("--description", "-d", type=str, default="",
                        help="Experiment description")
    parser.add_argument("--override", "-o", action="append", default=[],
                        help="Config override in format 'key=value' (can repeat)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List predefined experiments")
    parser.add_argument("--summary", "-s", action="store_true",
                        help="Print experiment summary table")
    parser.add_argument("--run-all-baselines", action="store_true",
                        help="Run all baseline experiments")

    args = parser.parse_args()

    if args.list:
        print("\nPredefined experiments:")
        print("-" * 60)
        for name, config in EXPERIMENTS.items():
            print(f"  {name}: {config['description']}")
        return

    if args.summary:
        print_summary_table()
        return

    if args.run_all_baselines:
        for name in ["baseline_fno", "baseline_unet"]:
            exp_config = EXPERIMENTS[name]
            run_experiment(name, exp_config["description"], exp_config.get("overrides"))
        print_summary_table()
        return

    if args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {list(EXPERIMENTS.keys())}")
            return

        exp_config = EXPERIMENTS[args.experiment]
        run_experiment(
            args.experiment,
            exp_config["description"],
            exp_config.get("overrides")
        )
    elif args.name:
        # Parse overrides
        overrides = {}
        for o in args.override:
            if '=' in o:
                key, value = o.split('=', 1)
                # Try to parse value as int/float/bool
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                overrides[key] = value

        run_experiment(args.name, args.description, overrides if overrides else None)
    else:
        parser.print_help()

    # Always print summary at the end
    if args.experiment or args.name:
        print_summary_table()


if __name__ == "__main__":
    main()
