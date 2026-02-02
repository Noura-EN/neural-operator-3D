#!/usr/bin/env python3
"""
Run no singularity mask ablation across all 10 seeds.

This experiment tests whether removing the singularity mask improves results.
The config uses use_singularity_mask: false and singularity_mask_radius: 0.
"""

import subprocess
import os
import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent

SEEDS = [42, 142, 242, 342, 442, 542, 642, 742, 842, 942]
CONFIG_NAME = "fno_analytical_standard_no_mask"


def get_experiment_name(seed: int) -> str:
    """Generate experiment name."""
    return f"fno_analytical_standard_no_mask_seed{seed}"


def check_experiment_completed(exp_name: str) -> bool:
    """Check if experiment has completed."""
    exp_base = PROJECT_ROOT / "experiments"
    for d in exp_base.iterdir():
        if d.is_dir() and d.name.startswith(exp_name):
            if (d / "final_results.json").exists():
                return True
    return False


def run_experiment(exp_name: str, seed: int) -> subprocess.Popen:
    """Start an experiment process."""
    config_path = PROJECT_ROOT / "configs" / "ablations" / f"{CONFIG_NAME}.yaml"

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


def main():
    print(f"\n{'='*60}")
    print(f"No Singularity Mask Ablation Runner")
    print(f"{'='*60}")
    print(f"Config: {CONFIG_NAME}")
    print(f"Seeds: {SEEDS}")
    print(f"{'='*60}\n")

    # Build queue
    queue = []
    for seed in SEEDS:
        exp_name = get_experiment_name(seed)
        if check_experiment_completed(exp_name):
            print(f"Skipping completed: {exp_name}")
            continue
        queue.append((exp_name, seed))

    print(f"Experiments to run: {len(queue)}")

    if not queue:
        print("All experiments completed!")
        return

    # Run experiments sequentially (1 at a time due to GPU)
    completed = []
    failed = []

    for exp_name, seed in queue:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting: {exp_name}")
        start_time = time.time()

        proc = run_experiment(exp_name, seed)

        # Wait for completion
        while proc.poll() is None:
            elapsed = (time.time() - start_time) / 60
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Running: {exp_name} ({elapsed:.1f} min elapsed)")
            time.sleep(60)  # Status update every minute

        elapsed = (time.time() - start_time) / 60
        if proc.returncode == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed: {exp_name} ({elapsed:.1f} min)")
            completed.append(exp_name)
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] FAILED: {exp_name} (code {proc.returncode})")
            failed.append(exp_name)

    # Summary
    print(f"\n{'='*60}")
    print(f"No Singularity Mask Ablation Complete")
    print(f"{'='*60}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed experiments: {failed}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
