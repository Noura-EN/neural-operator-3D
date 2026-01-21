"""Masking utilities for physics-informed loss computation."""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def create_muscle_mask(
    sigma: torch.Tensor,
    muscle_sigma_values: Tuple[float, float, float] = (0.2455, 0.2455, 1.2275),
    tolerance: float = 1e-4,
) -> torch.Tensor:
    """Create a binary mask for muscle tissue regions.

    Muscle tissue is identified by its anisotropic conductivity tensor
    with diagonal values (0.2455, 0.2455, 1.2275).

    Args:
        sigma: Conductivity tensor of shape (B, 6, D, H, W)
        muscle_sigma_values: Expected diagonal values for muscle
        tolerance: Tolerance for matching conductivity values

    Returns:
        Binary mask of shape (B, 1, D, H, W) where 1 indicates muscle
    """
    # Extract diagonal components (first 3 channels)
    sigma_xx = sigma[:, 0:1, ...]
    sigma_yy = sigma[:, 1:2, ...]
    sigma_zz = sigma[:, 2:3, ...]

    # Check if conductivity matches muscle values
    is_muscle = (
        (torch.abs(sigma_xx - muscle_sigma_values[0]) < tolerance) &
        (torch.abs(sigma_yy - muscle_sigma_values[1]) < tolerance) &
        (torch.abs(sigma_zz - muscle_sigma_values[2]) < tolerance)
    )

    return is_muscle.float()


def create_singularity_mask(
    source: torch.Tensor,
    radius: int = 3,
    source_point: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Create a spherical mask around the source singularity.

    Args:
        source: Source field of shape (B, 1, D, H, W)
        radius: Radius of the exclusion sphere in voxels
        source_point: Optional pre-computed source point (B, 3)

    Returns:
        Binary mask of shape (B, 1, D, H, W) where 1 indicates the singularity region
    """
    B, _, D, H, W = source.shape
    device = source.device

    # Find source peak location if not provided
    if source_point is None:
        # Flatten spatial dimensions and find argmax
        source_flat = source.view(B, -1)
        max_idx = torch.argmax(torch.abs(source_flat), dim=1)

        # Convert flat index to 3D coordinates
        z_idx = max_idx // (H * W)
        y_idx = (max_idx % (H * W)) // W
        x_idx = max_idx % W

        source_point = torch.stack([z_idx, y_idx, x_idx], dim=1).float()

    # Create coordinate grids
    z_coords = torch.arange(D, device=device).float()
    y_coords = torch.arange(H, device=device).float()
    x_coords = torch.arange(W, device=device).float()
    ZZ, YY, XX = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

    # Compute distance from source point for each sample in batch
    masks = []
    for b in range(B):
        sp = source_point[b] if source_point.dim() > 1 else source_point
        dist_sq = (ZZ - sp[0])**2 + (YY - sp[1])**2 + (XX - sp[2])**2
        mask = (dist_sq <= radius**2).float()
        masks.append(mask)

    singularity_mask = torch.stack(masks, dim=0).unsqueeze(1)  # (B, 1, D, H, W)

    return singularity_mask


def create_combined_mask(
    sigma: torch.Tensor,
    source: torch.Tensor,
    singularity_radius: int = 3,
    muscle_sigma_values: Tuple[float, float, float] = (0.2455, 0.2455, 1.2275),
    source_point: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Create combined mask: muscle region excluding singularity.

    M_combined = M_muscle AND NOT(M_singularity)

    Args:
        sigma: Conductivity tensor (B, 6, D, H, W)
        source: Source field (B, 1, D, H, W)
        singularity_radius: Radius around source to exclude
        muscle_sigma_values: Diagonal values identifying muscle tissue
        source_point: Optional pre-computed source location

    Returns:
        Combined mask (B, 1, D, H, W)
    """
    muscle_mask = create_muscle_mask(sigma, muscle_sigma_values)
    singularity_mask = create_singularity_mask(source, singularity_radius, source_point)

    # M_combined = M_muscle AND NOT(M_singularity)
    combined_mask = muscle_mask * (1 - singularity_mask)

    return combined_mask


class WeightedMaskedMSELoss(nn.Module):
    """Weighted MSE loss applied only to masked regions."""

    def __init__(
        self,
        weight: float = 1.0,
        singularity_radius: int = 3,
        use_muscle_mask: bool = True,
        muscle_sigma_values: Tuple[float, float, float] = (0.2455, 0.2455, 1.2275),
    ):
        """Initialize loss.

        Args:
            weight: Loss weight multiplier
            singularity_radius: Radius around source to exclude
            use_muscle_mask: Whether to restrict loss to muscle regions
            muscle_sigma_values: Diagonal conductivity values for muscle
        """
        super().__init__()
        self.weight = weight
        self.singularity_radius = singularity_radius
        self.use_muscle_mask = use_muscle_mask
        self.muscle_sigma_values = muscle_sigma_values

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sigma: torch.Tensor,
        source: torch.Tensor,
        source_point: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute weighted masked MSE loss.

        Args:
            pred: Predicted potential (B, 1, D, H, W)
            target: Target potential (B, 1, D, H, W)
            sigma: Conductivity tensor (B, 6, D, H, W)
            source: Source field (B, 1, D, H, W)
            source_point: Optional source location

        Returns:
            Scalar loss value
        """
        if self.use_muscle_mask:
            mask = create_combined_mask(
                sigma, source, self.singularity_radius,
                self.muscle_sigma_values, source_point
            )
        else:
            # Just exclude singularity
            singularity_mask = create_singularity_mask(
                source, self.singularity_radius, source_point
            )
            mask = 1 - singularity_mask

        # Apply mask
        masked_pred = pred * mask
        masked_target = target * mask

        # Compute MSE over masked region
        # Normalize by number of masked voxels to avoid scale issues
        num_masked = mask.sum() + 1e-8
        loss = ((masked_pred - masked_target) ** 2).sum() / num_masked

        return self.weight * loss


class GradientLoss(nn.Module):
    """Loss on the gradient of the potential field (electric field)."""

    def __init__(
        self,
        weight: float = 0.1,
    ):
        """Initialize gradient loss.

        Args:
            weight: Loss weight (lambda_grad)
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        spacing: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute gradient consistency loss.

        E = -∇Φ, so we compare gradients of predicted and target potentials.

        Args:
            pred: Predicted potential (B, 1, D, H, W)
            target: Target potential (B, 1, D, H, W)
            spacing: Voxel spacing (B, 3) for [dz, dy, dx]
            mask: Optional mask to restrict gradient comparison

        Returns:
            Scalar loss value
        """
        # Squeeze channel dimension for gradient computation
        pred_squeezed = pred.squeeze(1)  # (B, D, H, W)
        target_squeezed = target.squeeze(1)  # (B, D, H, W)

        # Compute gradients using torch.gradient
        # Note: torch.gradient expects spacing as float, we'll use mean spacing
        losses = []

        for b in range(pred.shape[0]):
            sp = spacing[b]  # (3,) - [dz, dy, dx] or [dx, dy, dz]

            # Compute gradients along each dimension
            grad_pred = torch.gradient(pred_squeezed[b], spacing=(sp[0].item(), sp[1].item(), sp[2].item()))
            grad_target = torch.gradient(target_squeezed[b], spacing=(sp[0].item(), sp[1].item(), sp[2].item()))

            # MSE between gradients
            grad_loss = 0.0
            for gp, gt in zip(grad_pred, grad_target):
                if mask is not None:
                    m = mask[b, 0]  # (D, H, W)
                    grad_loss += ((gp - gt) ** 2 * m).sum() / (m.sum() + 1e-8)
                else:
                    grad_loss += ((gp - gt) ** 2).mean()

            losses.append(grad_loss / 3.0)  # Average over 3 dimensions

        loss = torch.stack(losses).mean()

        return self.weight * loss


class CombinedLoss(nn.Module):
    """Combined loss function for potential field prediction."""

    def __init__(
        self,
        mse_weight: float = 1.0,
        grad_weight: float = 0.1,
        singularity_radius: int = 3,
        use_muscle_mask: bool = True,
    ):
        """Initialize combined loss.

        Args:
            mse_weight: Weight for MSE loss
            grad_weight: Weight for gradient loss
            singularity_radius: Radius around source to exclude
            use_muscle_mask: Whether to use muscle masking
        """
        super().__init__()

        self.mse_loss = WeightedMaskedMSELoss(
            weight=mse_weight,
            singularity_radius=singularity_radius,
            use_muscle_mask=use_muscle_mask,
        )

        self.grad_loss = GradientLoss(weight=grad_weight)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sigma: torch.Tensor,
        source: torch.Tensor,
        spacing: torch.Tensor,
        source_point: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined loss.

        Args:
            pred: Predicted potential (B, 1, D, H, W)
            target: Target potential (B, 1, D, H, W)
            sigma: Conductivity tensor (B, 6, D, H, W)
            source: Source field (B, 1, D, H, W)
            spacing: Voxel spacing (B, 3)
            source_point: Optional source location

        Returns:
            Tuple of (total_loss, loss_dict with individual components)
        """
        # Compute individual losses
        mse = self.mse_loss(pred, target, sigma, source, source_point)

        # Create mask for gradient loss
        mask = create_combined_mask(sigma, source, self.mse_loss.singularity_radius)
        grad = self.grad_loss(pred, target, spacing, mask)

        total_loss = mse + grad

        loss_dict = {
            "loss": total_loss.item(),
            "mse_loss": mse.item(),
            "grad_loss": grad.item(),
        }

        return total_loss, loss_dict
