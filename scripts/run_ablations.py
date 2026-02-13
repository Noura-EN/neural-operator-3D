#!/usr/bin/env python3
"""
Ablation study runner for neural operator experiments.

Runs 70 experiments:
- 4 model types × 2 training configs × 10 seeds = 70 experiments
  (UNet only runs standard, not mixed)

Models:
- UNet (standard only)
- FNO (standard + mixed)
- FNO with geometry attention (standard + mixed)
- TFNO (standard + mixed)

Seeds: 42, 142, 242, 342, 442, 542, 642, 742, 842, 942
"""

import subprocess
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import argparse

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Experiment configurations
# (config_filename, model_name, training_type)
EXPERIMENTS = [
    ("unet_standard", "unet", "standard"),
    ("fno_standard", "fno_analytical", "standard"),
    ("fno_mixed", "fno_analytical", "mixed"),
    ("fno_geom_attn_standard", "fno_geom_attn", "standard"),
    ("fno_geom_attn_mixed", "fno_geom_attn", "mixed"),
    ("tfno_standard", "tfno", "standard"),
    ("tfno_mixed", "tfno", "mixed"),
]

SEEDS = [42, 142, 242, 342, 442, 542, 642, 742, 842, 942]


def get_experiment_name(model_name: str, training_type: str, seed: int) -> str:
    """Generate experiment name."""
    return f"{model_name}_{training_type}_seed{seed}"


def get_experiment_dir(exp_name: str) -> Path:
    """Get experiment directory, preferring ones with final_results.json."""
    exp_base = PROJECT_ROOT / "experiments"
    matches = []
    for d in exp_base.iterdir():
        if d.is_dir() and d.name.startswith(exp_name):
            matches.append(d)

    if not matches:
        return None

    # Prefer directories with final_results.json
    for d in sorted(matches, key=lambda x: x.name, reverse=True):
        if (d / "final_results.json").exists():
            return d

    # Fall back to most recent
    return sorted(matches, key=lambda x: x.name, reverse=True)[0]


def check_experiment_completed(exp_name: str) -> bool:
    """Check if ANY matching experiment has completed (has final_results.json)."""
    exp_base = PROJECT_ROOT / "experiments"
    for d in exp_base.iterdir():
        if d.is_dir() and d.name.startswith(exp_name):
            if (d / "final_results.json").exists():
                return True
    return False


def run_experiment(config_name: str, exp_name: str, seed: int) -> subprocess.Popen:
    """Start an experiment process."""
    config_path = PROJECT_ROOT / "configs" / f"{config_name}.yaml"

    cmd = [
        "python3", str(PROJECT_ROOT / "scripts" / "main.py"),
        "--config", str(config_path),
        "--experiment-name", exp_name,
        "--seed", str(seed),
    ]

    # Start process
    log_file = PROJECT_ROOT / "logs" / f"{exp_name}.log"
    log_file.parent.mkdir(exist_ok=True)

    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )

    return process


def get_running_experiments() -> list:
    """Get list of currently running experiment processes."""
    result = subprocess.run(
        ["pgrep", "-af", "python3.*main.py"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().split("\n") if result.stdout.strip() else []


def count_running_experiments() -> int:
    """Count number of running experiment processes."""
    running = get_running_experiments()
    return len([r for r in running if r and "main.py" in r])


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--max-parallel", type=int, default=2,
                        help="Maximum number of parallel experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiments without running")
    parser.add_argument("--skip-completed", action="store_true", default=True,
                        help="Skip already completed experiments")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help="Seeds to run")
    parser.add_argument("--models", type=str, nargs="+",
                        help="Models to run (unet, fno_analytical, fno_geom_attn, tfno)")
    args = parser.parse_args()

    # Build experiment queue
    queue = []
    for config_name, model_name, training_type in EXPERIMENTS:
        # Filter by model if specified
        if args.models and model_name not in args.models:
            continue

        for seed in args.seeds:
            exp_name = get_experiment_name(model_name, training_type, seed)

            # Skip completed experiments
            if args.skip_completed and check_experiment_completed(exp_name):
                print(f"Skipping completed: {exp_name}")
                continue

            queue.append((config_name, exp_name, seed))

    print(f"\n{'='*60}")
    print(f"Ablation Study Runner")
    print(f"{'='*60}")
    print(f"Total experiments to run: {len(queue)}")
    print(f"Max parallel: {args.max_parallel}")
    print(f"Seeds: {args.seeds}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("Experiments to run:")
        for config_name, exp_name, seed in queue:
            print(f"  - {exp_name} (config: {config_name})")
        return

    # Run experiments
    running_processes = {}  # exp_name -> (process, start_time)
    completed = []
    failed = []

    while queue or running_processes:
        # Check completed processes
        finished = []
        for exp_name, (proc, start_time) in running_processes.items():
            if proc.poll() is not None:
                elapsed = time.time() - start_time
                if proc.returncode == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed: {exp_name} ({elapsed/60:.1f} min)")
                    completed.append(exp_name)
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] FAILED: {exp_name} (code {proc.returncode})")
                    failed.append(exp_name)
                finished.append(exp_name)

        for exp_name in finished:
            del running_processes[exp_name]

        # Start new experiments if slots available
        while queue and len(running_processes) < args.max_parallel:
            config_name, exp_name, seed = queue.pop(0)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting: {exp_name}")
            proc = run_experiment(config_name, exp_name, seed)
            running_processes[exp_name] = (proc, time.time())

        # Status update
        if running_processes:
            running_names = list(running_processes.keys())
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Running: {running_names}, Queue: {len(queue)}")

        # Wait before checking again
        time.sleep(60)  # Check every minute

    # Summary
    print(f"\n{'='*60}")
    print(f"Ablation Study Complete")
    print(f"{'='*60}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed experiments: {failed}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
