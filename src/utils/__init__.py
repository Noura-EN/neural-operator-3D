"""Utilities module for 3D potential field prediction."""

from .masking import (
    create_muscle_mask,
    create_singularity_mask,
    create_combined_mask,
    WeightedMaskedMSELoss,
    GradientLoss,
    CombinedLoss,
)
from .metrics import (
    mse,
    rmse,
    mae,
    relative_l2_error,
    max_error,
    compute_all_metrics,
    gradient_norm,
    parameter_norm,
)
from .visualization import (
    create_slice_comparison,
    create_multi_slice_figure,
    save_validation_figure,
    plot_loss_curves,
)

__all__ = [
    "create_muscle_mask",
    "create_singularity_mask",
    "create_combined_mask",
    "WeightedMaskedMSELoss",
    "GradientLoss",
    "CombinedLoss",
    "mse",
    "rmse",
    "mae",
    "relative_l2_error",
    "max_error",
    "compute_all_metrics",
    "gradient_norm",
    "parameter_norm",
    "create_slice_comparison",
    "create_multi_slice_figure",
    "save_validation_figure",
    "plot_loss_curves",
]
