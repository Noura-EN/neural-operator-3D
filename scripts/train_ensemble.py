#!/usr/bin/env python3
"""Train an ensemble of models with different random seeds."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train ensemble of models")
    parser.add_argument("--config", type=str, required=True, help="Base config file")
    parser.add_argument("--num-models", type=int, default=5, help="Number of models in ensemble")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--experiment-name", type=str, default="ensemble", help="Base experiment name")
    args = parser.parse_args()

    # Create ensemble directory
    ensemble_dir = Path(f"experiments/{args.experiment_name}_ensemble")
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {args.num_models} models for ensemble")
    print(f"Base config: {args.config}")
    print(f"Output directory: {ensemble_dir}")
    print("=" * 50)

    # Train each model with different seed
    for i in range(args.num_models):
        seed = args.base_seed + i * 100
        model_name = f"{args.experiment_name}_seed{seed}"

        print(f"\n[{i+1}/{args.num_models}] Training model with seed={seed}")
        print("-" * 40)

        cmd = [
            sys.executable, "scripts/main.py",
            "--config", args.config,
            "--seed", str(seed),
            "--experiment-name", model_name,
        ]

        result = subprocess.run(cmd, cwd=os.getcwd())

        if result.returncode != 0:
            print(f"Warning: Training failed for seed {seed}")
            continue

        # Find the experiment directory and create symlink
        exp_dirs = sorted(Path("experiments").glob(f"{model_name}_fno_*"))
        if exp_dirs:
            latest = exp_dirs[-1]
            link_path = ensemble_dir / f"model_{i}"
            if link_path.exists():
                link_path.unlink()
            link_path.symlink_to(latest.resolve())
            print(f"Linked: {link_path} -> {latest}")

    print("\n" + "=" * 50)
    print(f"Ensemble training complete!")
    print(f"Models saved in: {ensemble_dir}")
    print(f"\nTo evaluate ensemble, run:")
    print(f"  python scripts/eval_ensemble.py --ensemble-dir {ensemble_dir}")


if __name__ == "__main__":
    main()
