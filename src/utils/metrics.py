"""Evaluation metrics for potential field prediction.

Includes:
- Standard metrics: MSE, RMSE, MAE, relative L2, max error
- Region-wise metrics: inside/outside muscle, near/far from singularity
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


def mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Mean Squared Error.

    Args:
        pred: Predicted values
        target: Target values
        mask: Optional mask (1 = include, 0 = exclude)

    Returns:
        MSE value
    """
    diff_sq = (pred - target) ** 2

    if mask is not None:
        return (diff_sq * mask).sum() / (mask.sum() + 1e-8)
    else:
        return diff_sq.mean()


def rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Root Mean Squared Error.

    Args:
        pred: Predicted values
        target: Target values
        mask: Optional mask

    Returns:
        RMSE value
    """
    return torch.sqrt(mse(pred, target, mask))


def mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Mean Absolute Error.

    Args:
        pred: Predicted values
        target: Target values
        mask: Optional mask

    Returns:
        MAE value
    """
    diff_abs = torch.abs(pred - target)

    if mask is not None:
        return (diff_abs * mask).sum() / (mask.sum() + 1e-8)
    else:
        return diff_abs.mean()


def relative_l2_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute relative L2 error.

    Relative error = ||pred - target||_2 / ||target||_2

    Args:
        pred: Predicted values
        target: Target values
        mask: Optional mask
        eps: Small value for numerical stability

    Returns:
        Relative L2 error
    """
    if mask is not None:
        diff_sq = ((pred - target) ** 2 * mask).sum()
        target_sq = (target ** 2 * mask).sum()
    else:
        diff_sq = ((pred - target) ** 2).sum()
        target_sq = (target ** 2).sum()

    return torch.sqrt(diff_sq / (target_sq + eps))


def max_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute maximum absolute error.

    Args:
        pred: Predicted values
        target: Target values
        mask: Optional mask

    Returns:
        Max error value
    """
    diff_abs = torch.abs(pred - target)

    if mask is not None:
        diff_abs = diff_abs * mask
        # Set masked-out regions to -inf so they don't affect max
        diff_abs = diff_abs - (1 - mask) * 1e10

    return diff_abs.max()


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        pred: Predicted potential (B, 1, D, H, W)
        target: Target potential (B, 1, D, H, W)
        mask: Optional mask (B, 1, D, H, W)

    Returns:
        Dictionary with all metrics
    """
    with torch.no_grad():
        metrics = {
            "mse": mse(pred, target, mask).item(),
            "rmse": rmse(pred, target, mask).item(),
            "mae": mae(pred, target, mask).item(),
            "relative_l2": relative_l2_error(pred, target, mask).item(),
            "max_error": max_error(pred, target, mask).item(),
        }

    return metrics


def create_muscle_mask(
    sigma: torch.Tensor,
    muscle_sigma_values: Tuple[float, float, float] = (0.2455, 0.2455, 1.2275),
    tolerance: float = 1e-4,
) -> torch.Tensor:
    """Create a binary mask for muscle tissue regions.

    Args:
        sigma: Conductivity tensor (B, 6, D, H, W)
        muscle_sigma_values: Diagonal values for muscle
        tolerance: Matching tolerance

    Returns:
        Binary mask (B, 1, D, H, W)
    """
    sigma_xx = sigma[:, 0:1, ...]
    sigma_yy = sigma[:, 1:2, ...]
    sigma_zz = sigma[:, 2:3, ...]

    is_muscle = (
        (torch.abs(sigma_xx - muscle_sigma_values[0]) < tolerance) &
        (torch.abs(sigma_yy - muscle_sigma_values[1]) < tolerance) &
        (torch.abs(sigma_zz - muscle_sigma_values[2]) < tolerance)
    )

    return is_muscle.float()


def create_singularity_distance_mask(
    source: torch.Tensor,
    near_radius: int = 5,
    far_radius: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create masks for near and far from singularity regions.

    Args:
        source: Source field (B, 1, D, H, W)
        near_radius: Radius defining "near" singularity (voxels)
        far_radius: Distance beyond which is "far" from singularity

    Returns:
        Tuple of (near_mask, far_mask) each of shape (B, 1, D, H, W)
    """
    B, _, D, H, W = source.shape
    device = source.device

    near_masks = []
    far_masks = []

    for b in range(B):
        # Find source peak
        source_flat = source[b, 0].view(-1)
        max_idx = torch.argmax(torch.abs(source_flat))
        z_idx = max_idx // (H * W)
        y_idx = (max_idx % (H * W)) // W
        x_idx = max_idx % W

        # Create coordinate grids
        z_coords = torch.arange(D, device=device).float()
        y_coords = torch.arange(H, device=device).float()
        x_coords = torch.arange(W, device=device).float()
        ZZ, YY, XX = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

        # Compute distance from source
        dist = torch.sqrt((ZZ - z_idx.float())**2 + (YY - y_idx.float())**2 + (XX - x_idx.float())**2)

        # Create masks
        near_mask = (dist <= near_radius).float()
        far_mask = (dist >= far_radius).float()

        near_masks.append(near_mask)
        far_masks.append(far_mask)

    near_mask = torch.stack(near_masks, dim=0).unsqueeze(1)
    far_mask = torch.stack(far_masks, dim=0).unsqueeze(1)

    return near_mask, far_mask


def compute_region_wise_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigma: torch.Tensor,
    source: torch.Tensor,
    singularity_exclusion_radius: int = 3,
    near_singularity_radius: int = 8,
    far_singularity_radius: int = 15,
) -> Dict[str, float]:
    """Compute region-wise error metrics.

    Regions:
    - Inside muscle vs outside muscle
    - Near singularity vs far from singularity

    Args:
        pred: Predicted potential (B, 1, D, H, W)
        target: Target potential (B, 1, D, H, W)
        sigma: Conductivity tensor (B, 6, D, H, W)
        source: Source field (B, 1, D, H, W)
        singularity_exclusion_radius: Radius to exclude around singularity
        near_singularity_radius: Radius defining "near" region
        far_singularity_radius: Distance defining "far" region

    Returns:
        Dictionary with region-wise metrics
    """
    with torch.no_grad():
        # Create masks
        muscle_mask = create_muscle_mask(sigma)
        non_muscle_mask = 1.0 - muscle_mask

        # Create singularity exclusion mask
        near_sing, far_sing = create_singularity_distance_mask(
            source, near_singularity_radius, far_singularity_radius
        )

        # Exclude immediate singularity from all metrics
        _, exclusion_mask = create_singularity_distance_mask(
            source, singularity_exclusion_radius, singularity_exclusion_radius + 1
        )
        singularity_exclusion = 1.0 - create_singularity_distance_mask(
            source, singularity_exclusion_radius, singularity_exclusion_radius + 1
        )[0]

        metrics = {}

        # Muscle region metrics (excluding singularity)
        muscle_valid = muscle_mask * singularity_exclusion
        if muscle_valid.sum() > 0:
            metrics["mse_muscle"] = mse(pred, target, muscle_valid).item()
            metrics["rel_l2_muscle"] = relative_l2_error(pred, target, muscle_valid).item()
        else:
            metrics["mse_muscle"] = float('nan')
            metrics["rel_l2_muscle"] = float('nan')

        # Non-muscle region metrics (excluding singularity)
        non_muscle_valid = non_muscle_mask * singularity_exclusion
        if non_muscle_valid.sum() > 0:
            metrics["mse_non_muscle"] = mse(pred, target, non_muscle_valid).item()
            metrics["rel_l2_non_muscle"] = relative_l2_error(pred, target, non_muscle_valid).item()
        else:
            metrics["mse_non_muscle"] = float('nan')
            metrics["rel_l2_non_muscle"] = float('nan')

        # Near singularity metrics (but not inside exclusion zone)
        near_valid = near_sing * singularity_exclusion
        if near_valid.sum() > 0:
            metrics["mse_near_singularity"] = mse(pred, target, near_valid).item()
            metrics["rel_l2_near_singularity"] = relative_l2_error(pred, target, near_valid).item()
        else:
            metrics["mse_near_singularity"] = float('nan')
            metrics["rel_l2_near_singularity"] = float('nan')

        # Far from singularity metrics
        if far_sing.sum() > 0:
            metrics["mse_far_singularity"] = mse(pred, target, far_sing).item()
            metrics["rel_l2_far_singularity"] = relative_l2_error(pred, target, far_sing).item()
        else:
            metrics["mse_far_singularity"] = float('nan')
            metrics["rel_l2_far_singularity"] = float('nan')

        # Combined region: muscle AND far from singularity
        muscle_far = muscle_mask * far_sing
        if muscle_far.sum() > 0:
            metrics["mse_muscle_far"] = mse(pred, target, muscle_far).item()
            metrics["rel_l2_muscle_far"] = relative_l2_error(pred, target, muscle_far).item()
        else:
            metrics["mse_muscle_far"] = float('nan')
            metrics["rel_l2_muscle_far"] = float('nan')

    return metrics


def compute_all_metrics_extended(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigma: torch.Tensor,
    source: torch.Tensor,
    spacing: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute all metrics including region-wise breakdown and diagnostics.

    Args:
        pred: Predicted potential (B, 1, D, H, W)
        target: Target potential (B, 1, D, H, W)
        sigma: Conductivity tensor (B, 6, D, H, W)
        source: Source field (B, 1, D, H, W)
        spacing: Optional voxel spacing (B, 3)
        mask: Optional global mask

    Returns:
        Dictionary with all metrics
    """
    # Standard metrics
    metrics = compute_all_metrics(pred, target, mask)

    # Region-wise metrics
    region_metrics = compute_region_wise_metrics(pred, target, sigma, source)
    metrics.update(region_metrics)

    # Diagnostic metrics (scale and smoothness)
    diagnostic_metrics = compute_diagnostic_metrics(pred, target, sigma, source, spacing, mask)
    metrics.update(diagnostic_metrics)

    return metrics


def compute_scale_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute scale-related metrics to diagnose magnitude errors.

    Args:
        pred: Predicted potential (B, 1, D, H, W)
        target: Target potential (B, 1, D, H, W)
        mask: Optional mask (1 = include, 0 = exclude)
        eps: Small value for numerical stability

    Returns:
        Dictionary with scale metrics:
        - l2_norm_ratio: ||pred||_2 / ||target||_2
        - mean_abs_ratio: mean(|pred|) / mean(|target|)
    """
    with torch.no_grad():
        if mask is not None:
            pred_masked = pred * mask
            target_masked = target * mask
            n_valid = mask.sum() + eps

            pred_l2 = torch.sqrt((pred_masked ** 2).sum())
            target_l2 = torch.sqrt((target_masked ** 2).sum())

            pred_mean_abs = (torch.abs(pred_masked)).sum() / n_valid
            target_mean_abs = (torch.abs(target_masked)).sum() / n_valid
        else:
            pred_l2 = torch.sqrt((pred ** 2).sum())
            target_l2 = torch.sqrt((target ** 2).sum())

            pred_mean_abs = torch.abs(pred).mean()
            target_mean_abs = torch.abs(target).mean()

        l2_norm_ratio = (pred_l2 / (target_l2 + eps)).item()
        mean_abs_ratio = (pred_mean_abs / (target_mean_abs + eps)).item()

    return {
        "l2_norm_ratio": l2_norm_ratio,
        "mean_abs_ratio": mean_abs_ratio,
    }


def compute_gradient_field(
    field: torch.Tensor,
    spacing: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute gradient magnitude of a 3D field.

    Args:
        field: Input field (B, 1, D, H, W)
        spacing: Optional voxel spacing (B, 3) for [dz, dy, dx]

    Returns:
        Gradient magnitude field (B, 1, D, H, W)
    """
    B, _, D, H, W = field.shape
    device = field.device

    # Default spacing
    if spacing is None:
        spacing = torch.ones(B, 3, device=device)

    grad_mags = []
    for b in range(B):
        f = field[b, 0]  # (D, H, W)
        dz, dy, dx = spacing[b, 0].item(), spacing[b, 1].item(), spacing[b, 2].item()

        # Compute gradients using central differences
        grad_z = torch.zeros_like(f)
        grad_y = torch.zeros_like(f)
        grad_x = torch.zeros_like(f)

        # Central differences (interior points)
        grad_z[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2 * dz)
        grad_y[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2 * dy)
        grad_x[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * dx)

        # Forward/backward differences at boundaries
        grad_z[0, :, :] = (f[1, :, :] - f[0, :, :]) / dz
        grad_z[-1, :, :] = (f[-1, :, :] - f[-2, :, :]) / dz
        grad_y[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / dy
        grad_y[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / dy
        grad_x[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dx
        grad_x[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / dx

        grad_mag = torch.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
        grad_mags.append(grad_mag)

    return torch.stack(grad_mags, dim=0).unsqueeze(1)


def compute_laplacian_field(
    field: torch.Tensor,
    spacing: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Laplacian magnitude of a 3D field.

    Args:
        field: Input field (B, 1, D, H, W)
        spacing: Optional voxel spacing (B, 3) for [dz, dy, dx]

    Returns:
        Laplacian field (B, 1, D, H, W)
    """
    B, _, D, H, W = field.shape
    device = field.device

    # Default spacing
    if spacing is None:
        spacing = torch.ones(B, 3, device=device)

    laplacians = []
    for b in range(B):
        f = field[b, 0]  # (D, H, W)
        dz, dy, dx = spacing[b, 0].item(), spacing[b, 1].item(), spacing[b, 2].item()

        # Compute second derivatives using central differences
        lap = torch.zeros_like(f)

        # d²f/dz² (interior)
        lap[1:-1, :, :] += (f[2:, :, :] - 2*f[1:-1, :, :] + f[:-2, :, :]) / (dz**2)
        # d²f/dy² (interior)
        lap[:, 1:-1, :] += (f[:, 2:, :] - 2*f[:, 1:-1, :] + f[:, :-2, :]) / (dy**2)
        # d²f/dx² (interior)
        lap[:, :, 1:-1] += (f[:, :, 2:] - 2*f[:, :, 1:-1] + f[:, :, :-2]) / (dx**2)

        laplacians.append(lap)

    return torch.stack(laplacians, dim=0).unsqueeze(1)


def compute_smoothness_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    spacing: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute smoothness metrics to diagnose noise in predictions.

    Args:
        pred: Predicted potential (B, 1, D, H, W)
        target: Target potential (B, 1, D, H, W)
        spacing: Optional voxel spacing (B, 3)
        mask: Optional mask (1 = include, 0 = exclude)
        eps: Small value for numerical stability

    Returns:
        Dictionary with smoothness metrics:
        - gradient_energy_ratio: mean(||∇pred||) / mean(||∇target||)
        - laplacian_energy_ratio: mean(||Δpred||) / mean(||Δtarget||)
    """
    with torch.no_grad():
        # Compute gradient magnitudes
        grad_pred = compute_gradient_field(pred, spacing)
        grad_target = compute_gradient_field(target, spacing)

        # Compute Laplacian magnitudes
        lap_pred = compute_laplacian_field(pred, spacing)
        lap_target = compute_laplacian_field(target, spacing)

        if mask is not None:
            # Shrink mask to avoid boundary issues
            mask_shrunk = mask.clone()
            mask_shrunk[:, :, 0, :, :] = 0
            mask_shrunk[:, :, -1, :, :] = 0
            mask_shrunk[:, :, :, 0, :] = 0
            mask_shrunk[:, :, :, -1, :] = 0
            mask_shrunk[:, :, :, :, 0] = 0
            mask_shrunk[:, :, :, :, -1] = 0

            n_valid = mask_shrunk.sum() + eps

            mean_grad_pred = (torch.abs(grad_pred) * mask_shrunk).sum() / n_valid
            mean_grad_target = (torch.abs(grad_target) * mask_shrunk).sum() / n_valid

            mean_lap_pred = (torch.abs(lap_pred) * mask_shrunk).sum() / n_valid
            mean_lap_target = (torch.abs(lap_target) * mask_shrunk).sum() / n_valid
        else:
            # Use interior points only to avoid boundary artifacts
            interior = (slice(None), slice(None), slice(1, -1), slice(1, -1), slice(1, -1))

            mean_grad_pred = torch.abs(grad_pred[interior]).mean()
            mean_grad_target = torch.abs(grad_target[interior]).mean()

            mean_lap_pred = torch.abs(lap_pred[interior]).mean()
            mean_lap_target = torch.abs(lap_target[interior]).mean()

        gradient_energy_ratio = (mean_grad_pred / (mean_grad_target + eps)).item()
        laplacian_energy_ratio = (mean_lap_pred / (mean_lap_target + eps)).item()

    return {
        "gradient_energy_ratio": gradient_energy_ratio,
        "laplacian_energy_ratio": laplacian_energy_ratio,
    }


def compute_diagnostic_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigma: torch.Tensor,
    source: torch.Tensor,
    spacing: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute all diagnostic metrics for scale and smoothness.

    Args:
        pred: Predicted potential (B, 1, D, H, W)
        target: Target potential (B, 1, D, H, W)
        sigma: Conductivity tensor (B, 6, D, H, W)
        source: Source field (B, 1, D, H, W)
        spacing: Optional voxel spacing (B, 3)
        mask: Optional global mask

    Returns:
        Dictionary with all diagnostic metrics for entire domain and muscle region
    """
    metrics = {}

    # Create muscle mask for region-specific metrics
    muscle_mask = create_muscle_mask(sigma)

    # Entire domain metrics
    scale_entire = compute_scale_metrics(pred, target, mask)
    smooth_entire = compute_smoothness_metrics(pred, target, spacing, mask)

    metrics["l2_norm_ratio"] = scale_entire["l2_norm_ratio"]
    metrics["mean_abs_ratio"] = scale_entire["mean_abs_ratio"]
    metrics["gradient_energy_ratio"] = smooth_entire["gradient_energy_ratio"]
    metrics["laplacian_energy_ratio"] = smooth_entire["laplacian_energy_ratio"]

    # Muscle region metrics
    scale_muscle = compute_scale_metrics(pred, target, muscle_mask)
    smooth_muscle = compute_smoothness_metrics(pred, target, spacing, muscle_mask)

    metrics["l2_norm_ratio_muscle"] = scale_muscle["l2_norm_ratio"]
    metrics["mean_abs_ratio_muscle"] = scale_muscle["mean_abs_ratio"]
    metrics["gradient_energy_ratio_muscle"] = smooth_muscle["gradient_energy_ratio"]
    metrics["laplacian_energy_ratio_muscle"] = smooth_muscle["laplacian_energy_ratio"]

    return metrics


def gradient_norm(model: torch.nn.Module) -> float:
    """Compute the total gradient norm of a model.

    Args:
        model: PyTorch model

    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def parameter_norm(model: torch.nn.Module) -> float:
    """Compute the total parameter norm of a model.

    Args:
        model: PyTorch model

    Returns:
        Total parameter norm
    """
    total_norm = 0.0
    for p in model.parameters():
        total_norm += p.data.norm(2).item() ** 2
    return total_norm ** 0.5
