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


def create_percentile_singularity_mask(
    target: torch.Tensor,
    sigma: torch.Tensor,
    percentile: float = 99.0,
    muscle_sigma_values: Tuple[float, float, float] = (0.2455, 0.2455, 1.2275),
) -> torch.Tensor:
    """Create singularity mask based on percentile of target values.

    Masks voxels where |target| > percentile threshold within muscle region.

    Args:
        target: Target potential field (B, 1, D, H, W)
        sigma: Conductivity tensor (B, 6, D, H, W)
        percentile: Percentile threshold (e.g., 99 means mask top 1%)
        muscle_sigma_values: Diagonal values identifying muscle tissue

    Returns:
        Binary mask of shape (B, 1, D, H, W) where 1 indicates singularity region
    """
    B = target.shape[0]
    muscle_mask = create_muscle_mask(sigma, muscle_sigma_values)

    masks = []
    for b in range(B):
        muscle_bool = muscle_mask[b, 0] > 0.5
        muscle_values = torch.abs(target[b, 0][muscle_bool])

        if muscle_values.numel() > 0:
            threshold = torch.quantile(muscle_values, percentile / 100.0)
            singularity = (torch.abs(target[b, 0]) > threshold).float()
        else:
            singularity = torch.zeros_like(target[b, 0])

        masks.append(singularity)

    return torch.stack(masks, dim=0).unsqueeze(1)


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


def compute_distance_weights(
    source: torch.Tensor,
    shape: tuple,
    source_point: Optional[torch.Tensor] = None,
    alpha: float = 0.1,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute distance-based weights that upweight far-from-source voxels.

    Weight = 1 + alpha * distance (or normalized version)

    Args:
        source: Source field (B, 1, D, H, W)
        shape: Output shape (D, H, W)
        source_point: Pre-computed source location (B, 3)
        alpha: Weight scaling factor for distance
        normalize: If True, normalize distances to [0, 1] range

    Returns:
        Weights tensor (B, 1, D, H, W)
    """
    B = source.shape[0]
    D, H, W = shape
    device = source.device

    # Find source point if not provided
    if source_point is None:
        source_flat = source.view(B, -1)
        max_idx = torch.argmax(torch.abs(source_flat), dim=1)
        z_idx = max_idx // (H * W)
        y_idx = (max_idx % (H * W)) // W
        x_idx = max_idx % W
        source_point = torch.stack([z_idx, y_idx, x_idx], dim=1).float()

    # Create coordinate grids
    z_coords = torch.arange(D, device=device).float()
    y_coords = torch.arange(H, device=device).float()
    x_coords = torch.arange(W, device=device).float()
    ZZ, YY, XX = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

    weights = []
    for b in range(B):
        sp = source_point[b] if source_point.dim() > 1 else source_point
        dist = torch.sqrt((ZZ - sp[0])**2 + (YY - sp[1])**2 + (XX - sp[2])**2)

        if normalize:
            max_dist = dist.max()
            if max_dist > 0:
                dist = dist / max_dist  # Normalize to [0, 1]

        weight = 1.0 + alpha * dist
        weights.append(weight)

    return torch.stack(weights, dim=0).unsqueeze(1)


class WeightedMaskedMSELoss(nn.Module):
    """Weighted MSE loss with singularity exclusion.

    Excludes the singularity region around the source to avoid fitting
    the numerical artifacts near the point source.
    """

    def __init__(
        self,
        weight: float = 1.0,
        singularity_radius: int = 3,
        distance_weight_alpha: float = 0.0,  # 0 = no distance weighting
    ):
        super().__init__()
        self.weight = weight
        self.singularity_radius = singularity_radius
        self.distance_weight_alpha = distance_weight_alpha

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sigma: torch.Tensor,
        source: torch.Tensor,
        source_point: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute weighted masked MSE loss."""
        # Exclude singularity region
        singularity_mask = create_singularity_mask(
            source, self.singularity_radius, source_point
        )
        mask = 1 - singularity_mask

        # Compute distance weights if enabled
        if self.distance_weight_alpha > 0:
            dist_weights = compute_distance_weights(
                source, pred.shape[2:], source_point,
                alpha=self.distance_weight_alpha, normalize=True
            )
            # Combine mask and distance weights
            combined_weights = mask * dist_weights
        else:
            combined_weights = mask

        # Compute weighted MSE
        sq_diff = (pred - target) ** 2
        weighted_sq_diff = sq_diff * combined_weights

        # Normalize by sum of weights
        total_weight = combined_weights.sum() + 1e-8
        loss = weighted_sq_diff.sum() / total_weight

        return self.weight * loss


class GradientLoss(nn.Module):
    """Loss on the gradient of the potential field (electric field)."""

    def __init__(self, weight: float = 0.1):
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

        E = -grad(phi), so we compare gradients of predicted and target potentials.
        """
        pred_squeezed = pred.squeeze(1)  # (B, D, H, W)
        target_squeezed = target.squeeze(1)

        losses = []
        for b in range(pred.shape[0]):
            sp = spacing[b]  # (3,)

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

            losses.append(grad_loss / 3.0)

        loss = torch.stack(losses).mean()
        return self.weight * loss


class PDEResidualLoss(nn.Module):
    """True PDE residual loss: penalize -div(sigma * grad(phi)) - f.

    This enforces the governing equation directly.
    """

    def __init__(self, weight: float = 0.1, singularity_radius: int = 5):
        super().__init__()
        self.weight = weight
        self.singularity_radius = singularity_radius

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sigma: torch.Tensor,
        source: torch.Tensor,
        spacing: torch.Tensor,
        source_point: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute PDE residual loss.

        PDE: -div(sigma * grad(phi)) = f
        """
        B, _, D, H, W = pred.shape

        # Exclude singularity region
        sing_mask = create_singularity_mask(source, self.singularity_radius, source_point)
        valid_mask = 1 - sing_mask

        # Also exclude boundaries (for stable finite differences)
        valid_mask[:, :, 0, :, :] = 0
        valid_mask[:, :, -1, :, :] = 0
        valid_mask[:, :, :, 0, :] = 0
        valid_mask[:, :, :, -1, :] = 0
        valid_mask[:, :, :, :, 0] = 0
        valid_mask[:, :, :, :, -1] = 0

        residuals = []

        for b in range(B):
            phi = pred[b, 0]  # (D, H, W)
            f = source[b, 0]

            # Get diagonal conductivity components
            sigma_zz = sigma[b, 2]
            sigma_yy = sigma[b, 1]
            sigma_xx = sigma[b, 0]

            dz = spacing[b, 0].item()
            dy = spacing[b, 1].item()
            dx = spacing[b, 2].item()

            # Compute gradients using central differences
            dphi_dz = torch.zeros_like(phi)
            dphi_dy = torch.zeros_like(phi)
            dphi_dx = torch.zeros_like(phi)

            dphi_dz[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2 * dz)
            dphi_dy[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2 * dy)
            dphi_dx[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2 * dx)

            # Compute flux
            Jz = sigma_zz * dphi_dz
            Jy = sigma_yy * dphi_dy
            Jx = sigma_xx * dphi_dx

            # Compute divergence
            div_J = torch.zeros_like(phi)
            div_J[1:-1, :, :] += (Jz[2:, :, :] - Jz[:-2, :, :]) / (2 * dz)
            div_J[:, 1:-1, :] += (Jy[:, 2:, :] - Jy[:, :-2, :]) / (2 * dy)
            div_J[:, :, 1:-1] += (Jx[:, :, 2:] - Jx[:, :, :-2]) / (2 * dx)

            # PDE residual
            residual = -div_J - f
            residuals.append(residual)

        residual_tensor = torch.stack(residuals, dim=0).unsqueeze(1)

        # Compute masked MSE of residual
        masked_residual = (residual_tensor ** 2) * valid_mask
        num_valid = valid_mask.sum() + 1e-8
        loss = masked_residual.sum() / num_valid

        return self.weight * loss


class TotalVariationLoss(nn.Module):
    """Total Variation regularizer for smoothness.

    Penalizes the L1 norm of gradients, encouraging piecewise smooth predictions.
    """

    def __init__(self, weight: float = 0.01, singularity_radius: int = 3):
        super().__init__()
        self.weight = weight
        self.singularity_radius = singularity_radius

    def forward(
        self,
        pred: torch.Tensor,
        source: torch.Tensor,
        source_point: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute total variation loss."""
        # Exclude singularity region
        sing_mask = create_singularity_mask(source, self.singularity_radius, source_point)
        valid_mask = 1 - sing_mask

        # Compute differences along each axis
        diff_z = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
        diff_y = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
        diff_x = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])

        # Apply mask
        mask_z = valid_mask[:, :, 1:, :, :] * valid_mask[:, :, :-1, :, :]
        mask_y = valid_mask[:, :, :, 1:, :] * valid_mask[:, :, :, :-1, :]
        mask_x = valid_mask[:, :, :, :, 1:] * valid_mask[:, :, :, :, :-1]

        tv_z = (diff_z * mask_z).sum() / (mask_z.sum() + 1e-8)
        tv_y = (diff_y * mask_y).sum() / (mask_y.sum() + 1e-8)
        tv_x = (diff_x * mask_x).sum() / (mask_x.sum() + 1e-8)

        tv_loss = (tv_z + tv_y + tv_x) / 3.0

        return self.weight * tv_loss


class GradientMatchingLoss(nn.Module):
    """Gradient matching loss: MSE between predicted and target gradients.

    Loss = weight * MSE(grad(pred), grad(target))

    This directly encourages the model to match the gradient structure of the
    ground truth, which helps with smoothness and fine details.
    """

    def __init__(self, weight: float = 0.1, singularity_radius: int = 3):
        super().__init__()
        self.weight = weight
        self.singularity_radius = singularity_radius

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        spacing: torch.Tensor,
        source_point: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute gradient matching loss using finite differences."""
        # Exclude singularity region
        sing_mask = create_singularity_mask(source, self.singularity_radius, source_point)
        valid_mask = 1 - sing_mask

        B = pred.shape[0]
        total_loss = 0.0

        for b in range(B):
            p = pred[b, 0]  # (D, H, W)
            t = target[b, 0]
            m = valid_mask[b, 0]

            dz = spacing[b, 0].item()
            dy = spacing[b, 1].item()
            dx = spacing[b, 2].item()

            # Compute gradients using central differences
            grad_p_z = torch.zeros_like(p)
            grad_t_z = torch.zeros_like(t)
            grad_p_z[1:-1] = (p[2:] - p[:-2]) / (2 * dz)
            grad_t_z[1:-1] = (t[2:] - t[:-2]) / (2 * dz)

            grad_p_y = torch.zeros_like(p)
            grad_t_y = torch.zeros_like(t)
            grad_p_y[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * dy)
            grad_t_y[:, 1:-1] = (t[:, 2:] - t[:, :-2]) / (2 * dy)

            grad_p_x = torch.zeros_like(p)
            grad_t_x = torch.zeros_like(t)
            grad_p_x[:, :, 1:-1] = (p[:, :, 2:] - p[:, :, :-2]) / (2 * dx)
            grad_t_x[:, :, 1:-1] = (t[:, :, 2:] - t[:, :, :-2]) / (2 * dx)

            # Create interior mask (exclude boundaries)
            interior_mask = m.clone()
            interior_mask[0] = 0
            interior_mask[-1] = 0
            interior_mask[:, 0] = 0
            interior_mask[:, -1] = 0
            interior_mask[:, :, 0] = 0
            interior_mask[:, :, -1] = 0

            # MSE between gradients
            num_valid = interior_mask.sum() + 1e-8
            loss_z = ((grad_p_z - grad_t_z) ** 2 * interior_mask).sum() / num_valid
            loss_y = ((grad_p_y - grad_t_y) ** 2 * interior_mask).sum() / num_valid
            loss_x = ((grad_p_x - grad_t_x) ** 2 * interior_mask).sum() / num_valid

            total_loss += (loss_z + loss_y + loss_x) / 3.0

        return self.weight * (total_loss / B)


class CombinedLoss(nn.Module):
    """Combined loss function for potential field prediction.

    Default configuration uses MSE loss with TV regularization and singularity
    exclusion, which was found to be optimal in ablation studies.

    Available loss components:
    - MSE: Main reconstruction loss (default weight=1.0)
    - Gradient: Gradient consistency loss (default weight=0.5)
    - TV: Total variation regularizer (default weight=0.01)
    - PDE: PDE residual loss (default weight=0.0)
    - Gradient Matching: MSE between gradients (default weight=0.0)
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        grad_weight: float = 0.5,
        singularity_radius: int = 3,
        pde_weight: float = 0.0,
        tv_weight: float = 0.01,
        gradient_matching_weight: float = 0.0,
        use_singularity_mask: bool = True,
        singularity_mode: str = "radius",  # "radius" or "percentile"
        singularity_percentile: float = 99.0,  # Used if mode="percentile"
        distance_weight_alpha: float = 0.0,  # Distance weighting: 0=none, >0=upweight far
    ):
        """Initialize combined loss.

        Args:
            mse_weight: Weight for MSE loss
            grad_weight: Weight for gradient consistency loss
            singularity_radius: Radius around source to exclude (if mode="radius")
            pde_weight: Weight for PDE residual loss
            tv_weight: Weight for total variation regularizer
            gradient_matching_weight: Weight for gradient matching loss
            use_singularity_mask: Whether to exclude singularity region
            singularity_mode: "radius" for fixed radius, "percentile" for value-based
            singularity_percentile: Percentile threshold if mode="percentile"
            distance_weight_alpha: Distance weighting factor (weight = 1 + alpha * normalized_dist)
        """
        super().__init__()

        self.singularity_radius = singularity_radius
        self.use_singularity_mask = use_singularity_mask
        self.singularity_mode = singularity_mode
        self.singularity_percentile = singularity_percentile
        self.distance_weight_alpha = distance_weight_alpha

        # Primary MSE loss - we'll compute mask dynamically for percentile mode
        effective_radius = singularity_radius if (use_singularity_mask and singularity_mode == "radius") else 0
        self.primary_loss = WeightedMaskedMSELoss(
            weight=mse_weight,
            singularity_radius=effective_radius,
            distance_weight_alpha=distance_weight_alpha,
        )
        self.mse_weight = mse_weight

        # Gradient consistency loss
        self.grad_loss = GradientLoss(weight=grad_weight)
        self.grad_weight = grad_weight

        # PDE residual loss
        self.pde_loss = None
        self.pde_weight = pde_weight
        if pde_weight > 0:
            self.pde_loss = PDEResidualLoss(
                weight=pde_weight,
                singularity_radius=singularity_radius + 2,
            )

        # Total variation regularizer
        self.tv_loss = None
        self.tv_weight = tv_weight
        if tv_weight > 0:
            self.tv_loss = TotalVariationLoss(
                weight=tv_weight,
                singularity_radius=singularity_radius,
            )

        # Gradient matching loss
        self.gradient_matching_loss = None
        self.gradient_matching_weight = gradient_matching_weight
        if gradient_matching_weight > 0:
            self.gradient_matching_loss = GradientMatchingLoss(
                weight=gradient_matching_weight,
                singularity_radius=singularity_radius,
            )

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

        Returns:
            Tuple of (total_loss, loss_dict with individual components)
        """
        # Compute singularity mask based on mode
        if self.use_singularity_mask:
            if self.singularity_mode == "percentile":
                # Percentile-based mask: exclude top X% of absolute values
                singularity_mask = create_percentile_singularity_mask(
                    target, sigma, self.singularity_percentile
                )
                mask = 1 - singularity_mask
                # For percentile mode, compute MSE with custom mask
                muscle_mask = create_muscle_mask(sigma)
                valid_mask = muscle_mask * mask  # muscle AND NOT singularity
                sq_diff = (pred - target) ** 2
                if valid_mask.sum() > 0:
                    primary = self.mse_weight * (sq_diff * valid_mask).sum() / valid_mask.sum()
                else:
                    primary = self.mse_weight * sq_diff.mean()
            else:
                # Radius-based mask (original behavior)
                singularity_mask = create_singularity_mask(source, self.singularity_radius, source_point)
                mask = 1 - singularity_mask
                primary = self.primary_loss(pred, target, sigma, source, source_point)
        else:
            mask = None
            primary = self.primary_loss(pred, target, sigma, source, source_point)

        # Gradient consistency loss
        grad = self.grad_loss(pred, target, spacing, mask)

        total_loss = primary + grad

        loss_dict = {
            "mse_loss": primary.item(),
            "grad_loss": grad.item(),
        }

        # PDE residual loss
        if self.pde_loss is not None:
            pde = self.pde_loss(pred, target, sigma, source, spacing, source_point)
            total_loss = total_loss + pde
            loss_dict["pde_loss"] = pde.item()

        # Total variation regularizer
        if self.tv_loss is not None:
            tv = self.tv_loss(pred, source, source_point)
            total_loss = total_loss + tv
            loss_dict["tv_loss"] = tv.item()

        # Gradient matching loss
        if self.gradient_matching_loss is not None:
            grad_match = self.gradient_matching_loss(pred, target, source, spacing, source_point)
            total_loss = total_loss + grad_match
            loss_dict["gradient_matching_loss"] = grad_match.item()

        loss_dict["loss"] = total_loss.item()

        return total_loss, loss_dict
