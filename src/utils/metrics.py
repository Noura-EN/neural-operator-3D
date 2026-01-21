"""Evaluation metrics for potential field prediction."""

import torch
from typing import Dict, Optional


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
