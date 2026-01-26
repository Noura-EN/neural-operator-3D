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

        # Spacing encoding ablations
        "no_spacing_conditioning": {
            "spacing": {"use_spacing_conditioning": False}
        },
        "physical_coords": {
            "spacing": {"use_physical_coords": True}
        },
        "spacing_channels": {
            "spacing": {"add_spacing_channels": True}
        },
        "physical_coords_no_cond": {
            "spacing": {"use_physical_coords": True, "use_spacing_conditioning": False}
        },

        # Spacing conditioning mode ablations
        "spacing_additive": {
            "spacing": {"conditioning_mode": "additive"}
        },
        "spacing_film": {
            "spacing": {"conditioning_mode": "film"}
        },
        "spacing_gate": {
            "spacing": {"conditioning_mode": "gate"}
        },

        # ============================================
        # 352-sample architectural ablations
        # ============================================

        # Fourier modes ablations (more modes with more data)
        "modes_10": {
            "model": {"fno": {"modes1": 10, "modes2": 10, "modes3": 10}}
        },
        "modes_12": {
            "model": {"fno": {"modes1": 12, "modes2": 12, "modes3": 12}}
        },

        # Width ablations (wider models with more data)
        "width_48": {
            "model": {"fno": {"width": 48, "fc_dim": 192}}
        },
        "width_64": {
            "model": {"fno": {"width": 64, "fc_dim": 256}}
        },

        # Depth ablations (deeper models with more data)
        "layers_5": {
            "model": {"fno": {"num_layers": 5}}
        },
        "layers_6": {
            "model": {"fno": {"num_layers": 6}}
        },

        # Combination ablations (best settings)
        "modes10_layers6": {
            "model": {"fno": {"modes1": 10, "modes2": 10, "modes3": 10, "num_layers": 6}}
        },
        "width48_layers6": {
            "model": {"fno": {"width": 48, "fc_dim": 192, "num_layers": 6}}
        },
        "modes10_width48": {
            "model": {"fno": {"modes1": 10, "modes2": 10, "modes3": 10, "width": 48, "fc_dim": 192}}
        },

        # Deeper models (exploring if depth continues to help)
        "layers_7": {
            "model": {"fno": {"num_layers": 7}}
        },
        "layers_8": {
            "model": {"fno": {"num_layers": 8}}
        },

        # layers=6 variations (isolate TV contribution)
        "layers6_no_tv": {
            "model": {"fno": {"num_layers": 6}},
            "loss": {"tv_weight": 0.0}
        },
        "layers6_tv_0.02": {
            "model": {"fno": {"num_layers": 6}},
            "loss": {"tv_weight": 0.02}
        },
        "layers6_tv_0.005": {
            "model": {"fno": {"num_layers": 6}},
            "loss": {"tv_weight": 0.005}
        },

        # Best combination with TV
        "layers6_modes10_tv": {
            "model": {"fno": {"modes1": 10, "modes2": 10, "modes3": 10, "num_layers": 6}},
            "loss": {"tv_weight": 0.01}
        },

        # ============================================
        # 901-sample combined dataset ablations
        # (Use with --base-config configs/config_combined.yaml)
        # ============================================

        # Baseline for combined dataset
        "baseline_901": {},

        # Depth ablations (exploring deeper models with more data)
        "layers_6_901": {
            "model": {"fno": {"num_layers": 6}}
        },
        "layers_7_901": {
            "model": {"fno": {"num_layers": 7}}
        },
        "layers_8_901": {
            "model": {"fno": {"num_layers": 8}}
        },

        # Modes ablation
        "modes_10_901": {
            "model": {"fno": {"modes1": 10, "modes2": 10, "modes3": 10}}
        },

        # TV regularization ablations
        "no_tv_901": {
            "loss": {"tv_weight": 0.0}
        },
        "tv_0.02_901": {
            "loss": {"tv_weight": 0.02}
        },

        # Analytical solution input
        "analytical_solution": {
            "model": {"add_analytical_solution": True}
        },
        "layers6_analytical": {
            "model": {"fno": {"num_layers": 6}, "add_analytical_solution": True}
        },

        # Best combinations with 901 samples
        "layers6_modes10_901": {
            "model": {"fno": {"modes1": 10, "modes2": 10, "modes3": 10, "num_layers": 6}}
        },
        "layers6_tv_0.02_901": {
            "model": {"fno": {"num_layers": 6}},
            "loss": {"tv_weight": 0.02}
        },
        "layers7_tv_901": {
            "model": {"fno": {"num_layers": 7}},
            "loss": {"tv_weight": 0.01}
        },

        # Full combination: layers + modes + analytical
        "layers6_analytical_tv": {
            "model": {"fno": {"num_layers": 6}, "add_analytical_solution": True},
            "loss": {"tv_weight": 0.01}
        },

        # ============================================
        # Super-resolution experiments
        # Test if removing spacing conditioning improves zero-shot super-res
        # ============================================

        # No spacing conditioning + analytical solution
        # Hypothesis: spacing MLP extrapolates poorly to unseen spacing values
        # Analytical solution provides physics-aware scale information instead
        "no_spacing_analytical": {
            "model": {"add_analytical_solution": True},
            "spacing": {"use_spacing_conditioning": False}
        },

        # Best model without spacing conditioning
        "no_spacing_layers6_analytical": {
            "model": {"fno": {"num_layers": 6}, "add_analytical_solution": True},
            "spacing": {"use_spacing_conditioning": False}
        },

        # Baseline without spacing (for comparison)
        "no_spacing_baseline": {
            "spacing": {"use_spacing_conditioning": False}
        },

        # ============================================
        # Mixed-resolution training experiments
        # Include 50 original high-res samples to expose spacing MLP to high-res spacing
        # ============================================

        # Mixed resolution with layers6 + analytical (use config_mixed_resolution.yaml as base)
        "mixed_res_layers6_analytical": {},

        # ============================================
        # Spacing transform experiments
        # Test different spacing transformations for better generalization
        # ============================================

        # Log-transform spacing (compresses range for better extrapolation)
        "log_spacing_analytical": {
            "model": {"add_analytical_solution": True},
            "spacing": {"spacing_transform": "log"}
        },
        "log_spacing_layers6_analytical": {
            "model": {"fno": {"num_layers": 6}, "add_analytical_solution": True},
            "spacing": {"spacing_transform": "log"}
        },

        # Normalized spacing (divide by reference spacing)
        "normalized_spacing_analytical": {
            "model": {"add_analytical_solution": True},
            "spacing": {"spacing_transform": "normalized", "reference_spacing": 2.0}
        },
        "normalized_spacing_layers6_analytical": {
            "model": {"fno": {"num_layers": 6}, "add_analytical_solution": True},
            "spacing": {"spacing_transform": "normalized", "reference_spacing": 2.0}
        },

        # ============================================
        # Spacing conditioning comparison (option 4 vs 5)
        # Option 4: No spacing, no analytical
        # Option 5: No spacing, with analytical (already tested as no_spacing_analytical)
        # ============================================

        # Option 4: No spacing conditioning, no analytical solution
        "no_spacing_no_analytical": {
            "spacing": {"use_spacing_conditioning": False}
        },
        "no_spacing_no_analytical_layers6": {
            "model": {"fno": {"num_layers": 6}},
            "spacing": {"use_spacing_conditioning": False}
        },

        # ============================================
        # Residual learning experiments
        # Predict (u - analytical) instead of u directly
        # ============================================

        # Residual learning with baseline architecture
        "residual_learning": {
            "model": {"add_analytical_solution": True, "residual_learning": True}
        },

        # Residual learning with layers=6
        "residual_learning_layers6": {
            "model": {"fno": {"num_layers": 6}, "add_analytical_solution": True, "residual_learning": True}
        },

        # Residual learning without spacing conditioning
        "residual_learning_no_spacing": {
            "model": {"add_analytical_solution": True, "residual_learning": True},
            "spacing": {"use_spacing_conditioning": False}
        },

        # Residual learning with log spacing
        "residual_learning_log_spacing": {
            "model": {"add_analytical_solution": True, "residual_learning": True},
            "spacing": {"spacing_transform": "log"}
        },

        # Best combo: residual + layers6 + log spacing
        "residual_layers6_log_spacing": {
            "model": {"fno": {"num_layers": 6}, "add_analytical_solution": True, "residual_learning": True},
            "spacing": {"spacing_transform": "log"}
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
