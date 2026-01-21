"""Data transforms for 3D potential field prediction."""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def resample_volume(
    volume: torch.Tensor,
    target_shape: Tuple[int, int, int],
    mode: str = 'trilinear',
    align_corners: bool = True,
) -> torch.Tensor:
    """Resample a 3D volume to a target resolution.

    Args:
        volume: Input tensor of shape (C, D, H, W) or (B, C, D, H, W)
        target_shape: Target shape (D, H, W)
        mode: Interpolation mode ('trilinear' or 'nearest')
        align_corners: Whether to align corners in interpolation

    Returns:
        Resampled tensor
    """
    # Add batch dimension if needed
    squeeze_batch = False
    if volume.dim() == 4:
        volume = volume.unsqueeze(0)
        squeeze_batch = True

    # Resample
    resampled = F.interpolate(
        volume,
        size=target_shape,
        mode=mode,
        align_corners=align_corners if mode != 'nearest' else None,
    )

    if squeeze_batch:
        resampled = resampled.squeeze(0)

    return resampled


def resample_batch(
    batch: Dict[str, torch.Tensor],
    target_shape: Tuple[int, int, int],
    coord_range: Tuple[float, float] = (-1.0, 1.0),
) -> Dict[str, torch.Tensor]:
    """Resample all volumetric tensors in a batch to target resolution.

    Args:
        batch: Dictionary containing sigma, source, coords, mask, target, etc.
        target_shape: Target shape (D, H, W)
        coord_range: Range for coordinate generation

    Returns:
        Resampled batch
    """
    resampled = {}

    # Resample volumetric fields
    for key in ['sigma', 'source', 'mask', 'target']:
        if key in batch:
            resampled[key] = resample_volume(
                batch[key],
                target_shape,
                mode='trilinear' if key != 'mask' else 'nearest',
            )

    # Regenerate coordinates for new resolution
    D, H, W = target_shape
    low, high = coord_range
    z = torch.linspace(low, high, D, device=batch['coords'].device)
    y = torch.linspace(low, high, H, device=batch['coords'].device)
    x = torch.linspace(low, high, W, device=batch['coords'].device)
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
    coords = torch.stack([X, Y, Z], dim=0)

    # Add batch dimension if needed
    if batch['coords'].dim() == 5:
        coords = coords.unsqueeze(0).expand(batch['coords'].shape[0], -1, -1, -1, -1)

    resampled['coords'] = coords

    # Scale spacing inversely with resolution
    if 'spacing' in batch:
        original_shape = batch['sigma'].shape[-3:]
        scale = torch.tensor([
            original_shape[0] / target_shape[0],
            original_shape[1] / target_shape[1],
            original_shape[2] / target_shape[2],
        ], device=batch['spacing'].device)
        resampled['spacing'] = batch['spacing'] * scale

    # Copy non-volumetric fields
    for key in ['source_point']:
        if key in batch:
            resampled[key] = batch[key]

    return resampled


def normalize_field(
    field: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize a field to zero mean and unit variance.

    Args:
        field: Input tensor
        eps: Small value for numerical stability

    Returns:
        Tuple of (normalized_field, mean, std)
    """
    mean = field.mean()
    std = field.std() + eps
    normalized = (field - mean) / std
    return normalized, mean, std


def denormalize_field(
    field: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Denormalize a field.

    Args:
        field: Normalized tensor
        mean: Mean used for normalization
        std: Std used for normalization

    Returns:
        Denormalized tensor
    """
    return field * std + mean
