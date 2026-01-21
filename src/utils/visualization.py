"""Visualization utilities for potential field prediction."""

import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def create_slice_comparison(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    slice_type: str = "axial",
    slice_idx: Optional[int] = None,
    title_prefix: str = "",
    cmap: str = "viridis",
    error_cmap: str = "hot",
) -> plt.Figure:
    """Create a 3-panel comparison figure for a single slice.

    Panels:
    1. Masked ground-truth potential
    2. Masked predicted potential
    3. Log-absolute error

    Args:
        pred: Predicted potential (D, H, W) or (1, D, H, W)
        target: Target potential (D, H, W) or (1, D, H, W)
        mask: Optional mask (D, H, W) or (1, D, H, W)
        slice_type: Type of slice ("axial", "sagittal", "coronal")
        slice_idx: Index of slice (defaults to middle)
        title_prefix: Prefix for figure title
        cmap: Colormap for potential
        error_cmap: Colormap for error

    Returns:
        Matplotlib figure
    """
    # Convert to numpy and squeeze
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    pred = np.squeeze(pred)
    target = np.squeeze(target)
    if mask is not None:
        mask = np.squeeze(mask)

    D, H, W = pred.shape

    # Get slice based on type
    if slice_type == "axial":
        # Slice along depth (z-axis)
        idx = slice_idx if slice_idx is not None else D // 2
        pred_slice = pred[idx, :, :]
        target_slice = target[idx, :, :]
        mask_slice = mask[idx, :, :] if mask is not None else None
        axis_label = f"Axial (z={idx})"
    elif slice_type == "sagittal":
        # Slice along width (x-axis)
        idx = slice_idx if slice_idx is not None else W // 2
        pred_slice = pred[:, :, idx]
        target_slice = target[:, :, idx]
        mask_slice = mask[:, :, idx] if mask is not None else None
        axis_label = f"Sagittal (x={idx})"
    elif slice_type == "coronal":
        # Slice along height (y-axis)
        idx = slice_idx if slice_idx is not None else H // 2
        pred_slice = pred[:, idx, :]
        target_slice = target[:, idx, :]
        mask_slice = mask[:, idx, :] if mask is not None else None
        axis_label = f"Coronal (y={idx})"
    else:
        raise ValueError(f"Unknown slice type: {slice_type}")

    # Apply mask if provided
    if mask_slice is not None:
        pred_masked = np.where(mask_slice > 0.5, pred_slice, np.nan)
        target_masked = np.where(mask_slice > 0.5, target_slice, np.nan)
    else:
        pred_masked = pred_slice
        target_masked = target_slice

    # Compute error
    error = np.abs(pred_slice - target_slice)
    log_error = np.log10(error + 1e-10)  # Log-absolute error

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Determine common color scale for potential
    vmin = np.nanmin([np.nanmin(target_masked), np.nanmin(pred_masked)])
    vmax = np.nanmax([np.nanmax(target_masked), np.nanmax(pred_masked)])

    # Panel 1: Ground truth
    im1 = axes[0].imshow(target_masked.T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title("Ground Truth (FEM)")
    axes[0].set_xlabel("Dim 1")
    axes[0].set_ylabel("Dim 2")
    plt.colorbar(im1, ax=axes[0], label="Potential")

    # Panel 2: Prediction
    im2 = axes[1].imshow(pred_masked.T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title("Prediction")
    axes[1].set_xlabel("Dim 1")
    axes[1].set_ylabel("Dim 2")
    plt.colorbar(im2, ax=axes[1], label="Potential")

    # Panel 3: Log-absolute error
    im3 = axes[2].imshow(log_error.T, cmap=error_cmap, origin='lower')
    axes[2].set_title("Log₁₀(|Error|)")
    axes[2].set_xlabel("Dim 1")
    axes[2].set_ylabel("Dim 2")
    plt.colorbar(im3, ax=axes[2], label="Log₁₀(|Pred - Target|)")

    # Overall title
    fig.suptitle(f"{title_prefix} {axis_label}", fontsize=14)
    plt.tight_layout()

    return fig


def create_multi_slice_figure(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    slice_types: List[str] = ["axial", "sagittal"],
    title_prefix: str = "",
) -> plt.Figure:
    """Create a multi-row figure with different slice types.

    Args:
        pred: Predicted potential (D, H, W) or (1, D, H, W)
        target: Target potential (D, H, W) or (1, D, H, W)
        mask: Optional mask
        slice_types: List of slice types to include
        title_prefix: Prefix for figure title

    Returns:
        Matplotlib figure
    """
    n_rows = len(slice_types)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Convert to numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    pred = np.squeeze(pred)
    target = np.squeeze(target)
    if mask is not None:
        mask = np.squeeze(mask)

    D, H, W = pred.shape

    for row, slice_type in enumerate(slice_types):
        # Get slice
        if slice_type == "axial":
            idx = D // 2
            pred_slice = pred[idx, :, :]
            target_slice = target[idx, :, :]
            mask_slice = mask[idx, :, :] if mask is not None else None
            axis_label = f"Axial (z={idx})"
        elif slice_type == "sagittal":
            idx = W // 2
            pred_slice = pred[:, :, idx]
            target_slice = target[:, :, idx]
            mask_slice = mask[:, :, idx] if mask is not None else None
            axis_label = f"Sagittal (x={idx})"
        elif slice_type == "coronal":
            idx = H // 2
            pred_slice = pred[:, idx, :]
            target_slice = target[:, idx, :]
            mask_slice = mask[:, idx, :] if mask is not None else None
            axis_label = f"Coronal (y={idx})"

        # Apply mask
        if mask_slice is not None:
            pred_masked = np.where(mask_slice > 0.5, pred_slice, np.nan)
            target_masked = np.where(mask_slice > 0.5, target_slice, np.nan)
        else:
            pred_masked = pred_slice
            target_masked = target_slice

        # Compute error
        error = np.abs(pred_slice - target_slice)
        log_error = np.log10(error + 1e-10)

        # Color scale
        vmin = np.nanmin([np.nanmin(target_masked), np.nanmin(pred_masked)])
        vmax = np.nanmax([np.nanmax(target_masked), np.nanmax(pred_masked)])

        # Plot
        im1 = axes[row, 0].imshow(target_masked.T, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        axes[row, 0].set_title(f"Ground Truth - {axis_label}")
        plt.colorbar(im1, ax=axes[row, 0])

        im2 = axes[row, 1].imshow(pred_masked.T, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        axes[row, 1].set_title(f"Prediction - {axis_label}")
        plt.colorbar(im2, ax=axes[row, 1])

        im3 = axes[row, 2].imshow(log_error.T, cmap='hot', origin='lower')
        axes[row, 2].set_title(f"Log₁₀(|Error|) - {axis_label}")
        plt.colorbar(im3, ax=axes[row, 2])

    fig.suptitle(f"{title_prefix}", fontsize=14)
    plt.tight_layout()

    return fig


def save_validation_figure(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    epoch: int = 0,
    save_dir: str = "visualizations",
    slice_types: List[str] = ["axial", "sagittal"],
) -> str:
    """Save a validation visualization figure.

    Args:
        pred: Predicted potential
        target: Target potential
        mask: Optional mask
        epoch: Current epoch number
        save_dir: Directory to save figures
        slice_types: Slice types to include

    Returns:
        Path to saved figure
    """
    os.makedirs(save_dir, exist_ok=True)

    fig = create_multi_slice_figure(
        pred, target, mask,
        slice_types=slice_types,
        title_prefix=f"Epoch {epoch}",
    )

    save_path = os.path.join(save_dir, f"validation_epoch_{epoch:04d}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return save_path


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Progress', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Use log scale if range is large
    loss_range = max(train_losses) / (min(train_losses) + 1e-10)
    if loss_range > 100:
        ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
