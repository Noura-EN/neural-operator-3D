"""Training utilities including early stopping and experiment tracking."""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import torch
import numpy as np


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait after last improvement
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss (lower is better) or 'max' for metrics (higher is better)
        verbose: Whether to print messages
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        mode: str = 'min',
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """Check if training should stop.

        Args:
            score: Current validation score (loss or metric)
            epoch: Current epoch number

        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience} (best was epoch {self.best_epoch})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

        return self.early_stop

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


class ExperimentTracker:
    """Track experiments and maintain a summary across runs.

    Stores:
    - Configuration for each experiment
    - Training history (losses, metrics)
    - Best results
    """

    def __init__(
        self,
        base_dir: str = "experiments",
        summary_file: str = "experiment_summary.json",
    ):
        self.base_dir = Path(base_dir)
        self.summary_file = self.base_dir / summary_file
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Load existing summary or create new
        self.summary = self._load_summary()

    def _load_summary(self) -> Dict[str, Any]:
        """Load existing summary or create new one."""
        if self.summary_file.exists():
            with open(self.summary_file, 'r') as f:
                return json.load(f)
        return {
            "experiments": [],
            "best_experiment": None,
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _save_summary(self):
        """Save summary to file."""
        self.summary["last_updated"] = datetime.now().isoformat()
        with open(self.summary_file, 'w') as f:
            json.dump(self.summary, f, indent=2)

    def create_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        description: str = "",
    ) -> str:
        """Create a new experiment directory and entry.

        Args:
            name: Experiment name
            config: Configuration dictionary
            description: Human-readable description of changes

        Returns:
            Path to experiment directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{name}_{timestamp}"
        exp_dir = self.base_dir / exp_name

        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "visualizations").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)

        # Save config
        with open(exp_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        # Add to summary
        exp_entry = {
            "name": exp_name,
            "description": description,
            "config_changes": description,
            "timestamp": timestamp,
            "status": "running",
            "results": {},
            "path": str(exp_dir),
        }
        self.summary["experiments"].append(exp_entry)
        self._save_summary()

        return str(exp_dir)

    def update_experiment(
        self,
        exp_name: str,
        results: Dict[str, float],
        status: str = "completed",
    ):
        """Update experiment results.

        Args:
            exp_name: Experiment name
            results: Dictionary of results (metrics)
            status: Experiment status
        """
        for exp in self.summary["experiments"]:
            if exp["name"] == exp_name:
                exp["results"] = results
                exp["status"] = status

                # Update best experiment if this is better
                if self.summary["best_experiment"] is None:
                    self.summary["best_experiment"] = exp_name
                else:
                    best_exp = next(
                        (e for e in self.summary["experiments"]
                         if e["name"] == self.summary["best_experiment"]),
                        None
                    )
                    if best_exp and "val_loss" in results and "val_loss" in best_exp.get("results", {}):
                        if results["val_loss"] < best_exp["results"]["val_loss"]:
                            self.summary["best_experiment"] = exp_name

                break

        self._save_summary()

    def get_summary_table(self) -> str:
        """Generate a markdown table of all experiments.

        Returns:
            Markdown-formatted table
        """
        if not self.summary["experiments"]:
            return "No experiments recorded yet."

        # Collect all metric keys
        all_keys = set()
        for exp in self.summary["experiments"]:
            all_keys.update(exp.get("results", {}).keys())

        # Sort keys for consistent ordering
        metric_keys = sorted([k for k in all_keys if k not in ["epoch", "status"]])

        # Build table header
        header = "| Experiment | Description | Status |"
        for key in metric_keys[:6]:  # Limit columns
            header += f" {key} |"
        header += "\n"

        separator = "|" + "|".join(["---"] * (3 + min(len(metric_keys), 6))) + "|\n"

        # Build rows
        rows = []
        for exp in self.summary["experiments"]:
            is_best = exp["name"] == self.summary.get("best_experiment")
            name = f"**{exp['name']}**" if is_best else exp["name"]
            desc = exp.get("description", "")[:30]
            status = exp.get("status", "unknown")

            row = f"| {name} | {desc} | {status} |"
            for key in metric_keys[:6]:
                val = exp.get("results", {}).get(key, "N/A")
                if isinstance(val, float):
                    row += f" {val:.6f} |"
                else:
                    row += f" {val} |"
            rows.append(row)

        return header + separator + "\n".join(rows)

    def save_training_history(
        self,
        exp_dir: str,
        history: Dict[str, List[float]],
    ):
        """Save training history to experiment directory.

        Args:
            exp_dir: Experiment directory
            history: Dictionary with loss/metric histories
        """
        history_path = Path(exp_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: Dict[str, Any],
    save_path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    is_best: bool = False,
    additional_info: Optional[Dict[str, Any]] = None,
):
    """Save a training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        config: Configuration
        save_path: Path to save checkpoint
        scheduler: Optional scheduler state
        is_best: Whether this is the best model so far
        additional_info: Additional information to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
        "is_best": is_best,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if additional_info is not None:
        checkpoint.update(additional_info)

    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load to

    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
