"""Enhanced visualization utilities for potential field prediction.

Includes:
- 3D surface plots (height encodes potential value)
- Multiple slices at 25%, 50%, 75% depth + source peak slice
- Loss curve plotting with component breakdown
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def _to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def find_source_peak_indices(source: np.ndarray) -> Tuple[int, int, int]:
    """Find the voxel indices of the source peak.

    Args:
        source: Source field (D, H, W) or (1, D, H, W)

    Returns:
        Tuple of (z_idx, y_idx, x_idx) for the source peak
    """
    source = np.squeeze(source)
    peak_idx = np.unravel_index(np.argmax(np.abs(source)), source.shape)
    return peak_idx  # (z, y, x)


def get_slice_indices(
    shape: Tuple[int, int, int],
    source_peak: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, List[int]]:
    """Get slice indices at 25%, 50%, 75% and source peak.

    Args:
        shape: Volume shape (D, H, W)
        source_peak: Optional source peak location (z, y, x)

    Returns:
        Dictionary with slice indices for each axis
    """
    D, H, W = shape

    indices = {
        'axial': [D // 4, D // 2, 3 * D // 4],      # z-slices
        'coronal': [H // 4, H // 2, 3 * H // 4],    # y-slices
        'sagittal': [W // 4, W // 2, 3 * W // 4],   # x-slices
    }

    # Add source peak slices if provided
    if source_peak is not None:
        z_peak, y_peak, x_peak = source_peak
        if z_peak not in indices['axial']:
            indices['axial'].append(z_peak)
        if y_peak not in indices['coronal']:
            indices['coronal'].append(y_peak)
        if x_peak not in indices['sagittal']:
            indices['sagittal'].append(x_peak)

        # Sort indices
        for key in indices:
            indices[key] = sorted(indices[key])

    return indices


def create_3d_surface(
    data: np.ndarray,
    slice_type: str,
    slice_idx: int,
    mask: Optional[np.ndarray] = None,
    title: str = "",
    cmap: str = "viridis",
    ax: Optional[Axes3D] = None,
    downsample: int = 2,
) -> Axes3D:
    """Create a 3D surface plot where height encodes potential value.

    Args:
        data: 3D volume (D, H, W)
        slice_type: "axial", "coronal", or "sagittal"
        slice_idx: Index of slice
        mask: Optional mask to apply
        title: Plot title
        cmap: Colormap
        ax: Optional 3D axes to plot on
        downsample: Downsampling factor for performance

    Returns:
        Matplotlib 3D axes
    """
    data = np.squeeze(data)
    if mask is not None:
        mask = np.squeeze(mask)

    # Extract slice
    if slice_type == "axial":
        slice_data = data[slice_idx, :, :]
        mask_slice = mask[slice_idx, :, :] if mask is not None else None
        xlabel, ylabel = "X", "Y"
    elif slice_type == "coronal":
        slice_data = data[:, slice_idx, :]
        mask_slice = mask[:, slice_idx, :] if mask is not None else None
        xlabel, ylabel = "X", "Z"
    elif slice_type == "sagittal":
        slice_data = data[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx] if mask is not None else None
        xlabel, ylabel = "Y", "Z"
    else:
        raise ValueError(f"Unknown slice type: {slice_type}")

    # Apply mask
    if mask_slice is not None:
        slice_data = np.where(mask_slice > 0.5, slice_data, np.nan)

    # Downsample for performance
    slice_data = slice_data[::downsample, ::downsample]

    # Create meshgrid
    x = np.arange(slice_data.shape[0])
    y = np.arange(slice_data.shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(X, Y, slice_data, cmap=cmap,
                           linewidth=0, antialiased=True, alpha=0.9)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Potential")
    ax.set_title(title)

    return ax


def create_comprehensive_visualization(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    source: Optional[Union[torch.Tensor, np.ndarray]] = None,
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title_prefix: str = "",
    save_dir: Optional[str] = None,
    sample_idx: int = 0,
) -> Dict[str, str]:
    """Create comprehensive 3D surface visualizations.

    Creates:
    - 3D surface plots: masked and unmasked for target, pred, error
    - Multiple slices at 25%, 50%, 75% + source peak

    Args:
        pred: Predicted potential (D, H, W) or (1, D, H, W)
        target: Target potential (D, H, W) or (1, D, H, W)
        source: Optional source field for finding peak
        mask: Optional mask
        title_prefix: Prefix for titles
        save_dir: Directory to save figures
        sample_idx: Sample index for filename

    Returns:
        Dictionary of saved file paths
    """
    pred = np.squeeze(_to_numpy(pred))
    target = np.squeeze(_to_numpy(target))
    if source is not None:
        source = np.squeeze(_to_numpy(source))
    if mask is not None:
        mask = np.squeeze(_to_numpy(mask))

    D, H, W = pred.shape
    error = np.abs(pred - target)

    # Find source peak
    source_peak = None
    if source is not None:
        source_peak = find_source_peak_indices(source)

    # Get slice indices
    slice_indices = get_slice_indices((D, H, W), source_peak)

    saved_paths = {}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # --- 3D Surface Plots ---
    # Create 3D surfaces for multiple slices of each type
    for slice_type in ['axial', 'coronal', 'sagittal']:
        indices = slice_indices[slice_type]

        for idx in indices:
            is_peak = source_peak is not None and (
                (slice_type == 'axial' and idx == source_peak[0]) or
                (slice_type == 'coronal' and idx == source_peak[1]) or
                (slice_type == 'sagittal' and idx == source_peak[2])
            )
            suffix = "_peak" if is_peak else ""

            # Unmasked 3D
            fig = plt.figure(figsize=(18, 6))

            ax1 = fig.add_subplot(131, projection='3d')
            create_3d_surface(target, slice_type, idx, mask=None,
                             title=f"Target - {slice_type} idx={idx}", ax=ax1)

            ax2 = fig.add_subplot(132, projection='3d')
            create_3d_surface(pred, slice_type, idx, mask=None,
                             title=f"Prediction - {slice_type} idx={idx}", ax=ax2)

            ax3 = fig.add_subplot(133, projection='3d')
            create_3d_surface(error, slice_type, idx, mask=None, cmap='hot',
                             title=f"|Error| - {slice_type} idx={idx}", ax=ax3)

            fig.suptitle(f"{title_prefix} - 3D Surface (Unmasked)", fontsize=14)
            plt.tight_layout()

            if save_dir:
                path = os.path.join(save_dir, f"sample_{sample_idx:04d}_3d_{slice_type}_idx{idx}{suffix}.png")
                fig.savefig(path, dpi=150, bbox_inches='tight')
                saved_paths[f'3d_{slice_type}_idx{idx}'] = path
            plt.close(fig)

            # Masked 3D
            if mask is not None:
                fig = plt.figure(figsize=(18, 6))

                ax1 = fig.add_subplot(131, projection='3d')
                create_3d_surface(target, slice_type, idx, mask=mask,
                                 title=f"Target (masked) - {slice_type} idx={idx}", ax=ax1)

                ax2 = fig.add_subplot(132, projection='3d')
                create_3d_surface(pred, slice_type, idx, mask=mask,
                                 title=f"Prediction (masked) - {slice_type} idx={idx}", ax=ax2)

                ax3 = fig.add_subplot(133, projection='3d')
                create_3d_surface(error, slice_type, idx, mask=mask, cmap='hot',
                                 title=f"|Error| (masked) - {slice_type} idx={idx}", ax=ax3)

                fig.suptitle(f"{title_prefix} - 3D Surface (Masked)", fontsize=14)
                plt.tight_layout()

                if save_dir:
                    path = os.path.join(save_dir, f"sample_{sample_idx:04d}_3d_{slice_type}_idx{idx}{suffix}_masked.png")
                    fig.savefig(path, dpi=150, bbox_inches='tight')
                    saved_paths[f'3d_{slice_type}_idx{idx}_masked'] = path
                plt.close(fig)

    return saved_paths


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_mse: Optional[List[float]] = None,
    train_grad: Optional[List[float]] = None,
    val_mse: Optional[List[float]] = None,
    val_grad: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    title: str = "Training Progress",
) -> plt.Figure:
    """Plot training and validation loss curves with component breakdown.

    Args:
        train_losses: Total training losses per epoch
        val_losses: Total validation losses per epoch
        train_mse: Optional MSE component of training loss
        train_grad: Optional gradient component of training loss
        val_mse: Optional MSE component of validation loss
        val_grad: Optional gradient component of validation loss
        save_path: Optional path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    has_components = train_mse is not None and train_grad is not None

    if has_components:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Total loss
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Total Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Auto log scale
    loss_range = max(train_losses) / (min(train_losses) + 1e-10)
    if loss_range > 100:
        axes[0].set_yscale('log')

    if has_components:
        # MSE component
        axes[1].plot(epochs, train_mse, 'b-', label='Train MSE', linewidth=2)
        axes[1].plot(epochs, val_mse, 'r-', label='Val MSE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MSE Loss', fontsize=12)
        axes[1].set_title('MSE Component', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        if max(train_mse) / (min(train_mse) + 1e-10) > 100:
            axes[1].set_yscale('log')

        # Gradient component
        axes[2].plot(epochs, train_grad, 'b-', label='Train Grad', linewidth=2)
        axes[2].plot(epochs, val_grad, 'r-', label='Val Grad', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Gradient Loss', fontsize=12)
        axes[2].set_title('Gradient Component', fontsize=14)
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)
        if max(train_grad) / (min(train_grad) + 1e-10) > 100:
            axes[2].set_yscale('log')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def save_validation_figure(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    source: Optional[torch.Tensor] = None,
    epoch: int = 0,
    save_dir: str = "visualizations",
    sample_idx: int = 0,
    comprehensive: bool = True,
) -> Dict[str, str]:
    """Save validation visualization figures.

    Args:
        pred: Predicted potential
        target: Target potential
        mask: Optional mask
        source: Optional source field for finding peak
        epoch: Current epoch number
        save_dir: Directory to save figures
        sample_idx: Sample index
        comprehensive: Whether to create comprehensive visualizations (always True)

    Returns:
        Dictionary of saved file paths
    """
    os.makedirs(save_dir, exist_ok=True)

    return create_comprehensive_visualization(
        pred, target, source, mask,
        title_prefix=f"Epoch {epoch}",
        save_dir=save_dir,
        sample_idx=sample_idx,
    )
