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


class NormalizedMSELoss(nn.Module):
    """Pointwise normalized MSE loss to address scale issues.

    Loss = mean((pred - target)² / (target² + eps))

    This loss gives equal weight to relative errors regardless of target magnitude.
    """

    def __init__(
        self,
        weight: float = 1.0,
        eps: float = 1e-6,
        singularity_radius: int = 3,
        use_muscle_mask: bool = False,
    ):
        """Initialize normalized MSE loss.

        Args:
            weight: Loss weight multiplier
            eps: Small value to avoid division by zero
            singularity_radius: Radius around source to exclude
            use_muscle_mask: Whether to restrict to muscle regions
        """
        super().__init__()
        self.weight = weight
        self.eps = eps
        self.singularity_radius = singularity_radius
        self.use_muscle_mask = use_muscle_mask

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sigma: torch.Tensor,
        source: torch.Tensor,
        source_point: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute pointwise normalized MSE loss.

        Args:
            pred: Predicted potential (B, 1, D, H, W)
            target: Target potential (B, 1, D, H, W)
            sigma: Conductivity tensor (B, 6, D, H, W)
            source: Source field (B, 1, D, H, W)
            source_point: Optional source location

        Returns:
            Scalar loss value
        """
        # Create mask
        if self.use_muscle_mask:
            mask = create_combined_mask(
                sigma, source, self.singularity_radius, source_point=source_point
            )
        else:
            singularity_mask = create_singularity_mask(
                source, self.singularity_radius, source_point
            )
            mask = 1 - singularity_mask

        # Compute pointwise normalized error
        squared_error = (pred - target) ** 2
        normalization = target ** 2 + self.eps
        normalized_error = squared_error / normalization

        # Apply mask
        masked_error = normalized_error * mask
        num_masked = mask.sum() + 1e-8
        loss = masked_error.sum() / num_masked

        return self.weight * loss


class LogCoshLoss(nn.Module):
    """Log-cosh loss for robust regression.

    Loss = mean(log(cosh(pred - target)))

    This is approximately quadratic for small errors and linear for large errors,
    making it more robust to outliers.
    """

    def __init__(
        self,
        weight: float = 1.0,
        singularity_radius: int = 3,
        use_muscle_mask: bool = False,
    ):
        super().__init__()
        self.weight = weight
        self.singularity_radius = singularity_radius
        self.use_muscle_mask = use_muscle_mask

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sigma: torch.Tensor,
        source: torch.Tensor,
        source_point: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Create mask
        if self.use_muscle_mask:
            mask = create_combined_mask(
                sigma, source, self.singularity_radius, source_point=source_point
            )
        else:
            singularity_mask = create_singularity_mask(
                source, self.singularity_radius, source_point
            )
            mask = 1 - singularity_mask

        diff = pred - target
        # log(cosh(x)) = x + softplus(-2x) - log(2)
        # Use numerically stable version
        loss_pointwise = diff + torch.nn.functional.softplus(-2.0 * diff) - 0.693147

        masked_loss = loss_pointwise * mask
        num_masked = mask.sum() + 1e-8
        loss = masked_loss.sum() / num_masked

        return self.weight * loss


class PDEResidualLoss(nn.Module):
    """True PDE residual loss: penalize -∇·(σ∇Φ) - f.

    This enforces the governing equation directly.
    """

    def __init__(
        self,
        weight: float = 0.1,
        singularity_radius: int = 5,
    ):
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

        PDE: -∇·(σ∇Φ) = f

        For anisotropic conductivity with diagonal σ:
        -∂/∂x(σ_xx ∂Φ/∂x) - ∂/∂y(σ_yy ∂Φ/∂y) - ∂/∂z(σ_zz ∂Φ/∂z) = f
        """
        B, _, D, H, W = pred.shape
        device = pred.device

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
            f = source[b, 0]  # (D, H, W)

            # Get diagonal conductivity components
            sigma_zz = sigma[b, 2]  # (D, H, W)
            sigma_yy = sigma[b, 1]  # (D, H, W)
            sigma_xx = sigma[b, 0]  # (D, H, W)

            dz = spacing[b, 0].item()
            dy = spacing[b, 1].item()
            dx = spacing[b, 2].item()

            # Compute ∂Φ/∂x, ∂Φ/∂y, ∂Φ/∂z using central differences
            dphi_dz = torch.zeros_like(phi)
            dphi_dy = torch.zeros_like(phi)
            dphi_dx = torch.zeros_like(phi)

            dphi_dz[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2 * dz)
            dphi_dy[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2 * dy)
            dphi_dx[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2 * dx)

            # Compute flux: J_i = σ_ii * ∂Φ/∂x_i
            Jz = sigma_zz * dphi_dz
            Jy = sigma_yy * dphi_dy
            Jx = sigma_xx * dphi_dx

            # Compute divergence: ∇·J = ∂J_z/∂z + ∂J_y/∂y + ∂J_x/∂x
            div_J = torch.zeros_like(phi)

            div_J[1:-1, :, :] += (Jz[2:, :, :] - Jz[:-2, :, :]) / (2 * dz)
            div_J[:, 1:-1, :] += (Jy[:, 2:, :] - Jy[:, :-2, :]) / (2 * dy)
            div_J[:, :, 1:-1] += (Jx[:, :, 2:] - Jx[:, :, :-2]) / (2 * dx)

            # PDE residual: -∇·(σ∇Φ) - f should be zero
            residual = -div_J - f

            residuals.append(residual)

        residual_tensor = torch.stack(residuals, dim=0).unsqueeze(1)

        # Compute masked MSE of residual
        masked_residual = (residual_tensor ** 2) * valid_mask
        num_valid = valid_mask.sum() + 1e-8
        loss = masked_residual.sum() / num_valid

        return self.weight * loss


class SpectralSmoothingLoss(nn.Module):
    """Spectral smoothing regularizer to penalize high-frequency content.

    Two variants:
    - threshold: Penalize energy above a frequency threshold
    - weighted: Penalize all frequencies with higher weights for higher frequencies
    """

    def __init__(
        self,
        weight: float = 0.01,
        mode: str = "threshold",  # "threshold" or "weighted"
        threshold_ratio: float = 0.5,  # For threshold mode: ratio of max frequency
        power: float = 2.0,  # For weighted mode: weight = freq^power
    ):
        super().__init__()
        self.weight = weight
        self.mode = mode
        self.threshold_ratio = threshold_ratio
        self.power = power

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spectral smoothing loss.

        Args:
            pred: Predicted potential (B, 1, D, H, W)
            target: Target potential (B, 1, D, H, W)

        Returns:
            Scalar loss value
        """
        B, _, D, H, W = pred.shape
        device = pred.device

        # Compute 3D FFT of prediction
        pred_fft = torch.fft.rfftn(pred, dim=(-3, -2, -1))

        # Create frequency grid
        freq_z = torch.fft.fftfreq(D, device=device)
        freq_y = torch.fft.fftfreq(H, device=device)
        freq_x = torch.fft.rfftfreq(W, device=device)

        FZ, FY, FX = torch.meshgrid(freq_z, freq_y, freq_x, indexing='ij')
        freq_magnitude = torch.sqrt(FZ**2 + FY**2 + FX**2)

        # Normalize frequency magnitude
        max_freq = freq_magnitude.max()
        freq_normalized = freq_magnitude / (max_freq + 1e-8)

        if self.mode == "threshold":
            # Penalize energy above threshold frequency
            high_freq_mask = (freq_normalized > self.threshold_ratio).float()
            pred_energy = torch.abs(pred_fft) ** 2
            high_freq_energy = (pred_energy * high_freq_mask.unsqueeze(0).unsqueeze(0)).mean()
            loss = high_freq_energy

        elif self.mode == "weighted":
            # Penalize all frequencies with weight proportional to frequency^power
            weights = freq_normalized ** self.power
            pred_energy = torch.abs(pred_fft) ** 2
            weighted_energy = (pred_energy * weights.unsqueeze(0).unsqueeze(0)).mean()
            loss = weighted_energy

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return self.weight * loss


class TotalVariationLoss(nn.Module):
    """Total Variation regularizer for smoothness.

    Penalizes the L1 norm of gradients, encouraging piecewise smooth predictions.
    TV(u) = sum(|∇u|) ≈ sum(|u[i+1] - u[i]|)
    """

    def __init__(
        self,
        weight: float = 0.01,
        singularity_radius: int = 3,
    ):
        super().__init__()
        self.weight = weight
        self.singularity_radius = singularity_radius

    def forward(
        self,
        pred: torch.Tensor,
        source: torch.Tensor,
        source_point: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute total variation loss.

        Args:
            pred: Predicted potential (B, 1, D, H, W)
            source: Source field for singularity masking
            source_point: Optional source location

        Returns:
            Scalar loss value
        """
        # Exclude singularity region
        sing_mask = create_singularity_mask(source, self.singularity_radius, source_point)
        valid_mask = 1 - sing_mask

        # Compute differences along each axis
        diff_z = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
        diff_y = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
        diff_x = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])

        # Apply mask (need to adjust mask size for each direction)
        mask_z = valid_mask[:, :, 1:, :, :] * valid_mask[:, :, :-1, :, :]
        mask_y = valid_mask[:, :, :, 1:, :] * valid_mask[:, :, :, :-1, :]
        mask_x = valid_mask[:, :, :, :, 1:] * valid_mask[:, :, :, :, :-1]

        tv_z = (diff_z * mask_z).sum() / (mask_z.sum() + 1e-8)
        tv_y = (diff_y * mask_y).sum() / (mask_y.sum() + 1e-8)
        tv_x = (diff_x * mask_x).sum() / (mask_x.sum() + 1e-8)

        tv_loss = (tv_z + tv_y + tv_x) / 3.0

        return self.weight * tv_loss


class CombinedLoss(nn.Module):
    """Combined loss function for potential field prediction."""

    def __init__(
        self,
        mse_weight: float = 1.0,
        grad_weight: float = 0.1,
        singularity_radius: int = 3,
        use_muscle_mask: bool = True,
        # New options
        loss_type: str = "mse",  # "mse", "normalized", "logcosh", "mse_logcosh"
        pde_weight: float = 0.0,  # True PDE residual weight
        spectral_weight: float = 0.0,  # Spectral smoothing weight
        spectral_mode: str = "threshold",  # "threshold" or "weighted"
        use_singularity_mask: bool = True,  # Whether to exclude singularity
        logcosh_weight: float = 0.0,  # For hybrid MSE + log-cosh
        tv_weight: float = 0.0,  # Total variation regularizer
    ):
        """Initialize combined loss.

        Args:
            mse_weight: Weight for MSE loss
            grad_weight: Weight for gradient consistency loss
            singularity_radius: Radius around source to exclude
            use_muscle_mask: Whether to use muscle masking
            loss_type: Type of primary loss ("mse", "normalized", "logcosh", "mse_logcosh")
            pde_weight: Weight for true PDE residual loss
            spectral_weight: Weight for spectral smoothing regularizer
            spectral_mode: Spectral smoothing mode ("threshold" or "weighted")
            use_singularity_mask: Whether to exclude singularity region
            logcosh_weight: Weight for log-cosh term in hybrid loss
            tv_weight: Weight for total variation regularizer
        """
        super().__init__()

        self.loss_type = loss_type
        self.singularity_radius = singularity_radius
        self.use_muscle_mask = use_muscle_mask
        self.use_singularity_mask = use_singularity_mask

        # Primary loss
        if loss_type == "mse" or loss_type == "mse_logcosh":
            self.primary_loss = WeightedMaskedMSELoss(
                weight=mse_weight,
                singularity_radius=singularity_radius if use_singularity_mask else 0,
                use_muscle_mask=use_muscle_mask,
            )
        elif loss_type == "normalized":
            self.primary_loss = NormalizedMSELoss(
                weight=mse_weight,
                singularity_radius=singularity_radius if use_singularity_mask else 0,
                use_muscle_mask=use_muscle_mask,
            )
        elif loss_type == "logcosh":
            self.primary_loss = LogCoshLoss(
                weight=mse_weight,
                singularity_radius=singularity_radius if use_singularity_mask else 0,
                use_muscle_mask=use_muscle_mask,
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Log-cosh component for hybrid loss
        self.logcosh_loss = None
        self.logcosh_weight = logcosh_weight
        if loss_type == "mse_logcosh" or logcosh_weight > 0:
            self.logcosh_loss = LogCoshLoss(
                weight=logcosh_weight if logcosh_weight > 0 else 0.1,
                singularity_radius=singularity_radius if use_singularity_mask else 0,
                use_muscle_mask=use_muscle_mask,
            )

        # Gradient consistency loss
        self.grad_loss = GradientLoss(weight=grad_weight)
        self.grad_weight = grad_weight

        # PDE residual loss
        self.pde_loss = None
        self.pde_weight = pde_weight
        if pde_weight > 0:
            self.pde_loss = PDEResidualLoss(
                weight=pde_weight,
                singularity_radius=singularity_radius + 2,  # Larger exclusion for PDE
            )

        # Spectral smoothing loss
        self.spectral_loss = None
        self.spectral_weight = spectral_weight
        if spectral_weight > 0:
            self.spectral_loss = SpectralSmoothingLoss(
                weight=spectral_weight,
                mode=spectral_mode,
            )

        # Total variation regularizer
        self.tv_loss = None
        self.tv_weight = tv_weight
        if tv_weight > 0:
            self.tv_loss = TotalVariationLoss(
                weight=tv_weight,
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
        # Compute primary loss (MSE, normalized, or log-cosh)
        primary = self.primary_loss(pred, target, sigma, source, source_point)

        # Create mask for gradient loss
        if self.use_singularity_mask:
            mask = create_combined_mask(sigma, source, self.singularity_radius)
        else:
            mask = None

        # Gradient consistency loss
        grad = self.grad_loss(pred, target, spacing, mask)

        total_loss = primary + grad

        loss_dict = {
            "mse_loss": primary.item(),  # Keep 'mse_loss' key for backward compatibility
            "grad_loss": grad.item(),
        }

        # Log-cosh component (for hybrid loss)
        if self.logcosh_loss is not None:
            logcosh = self.logcosh_loss(pred, target, sigma, source, source_point)
            total_loss = total_loss + logcosh
            loss_dict["logcosh_loss"] = logcosh.item()

        # PDE residual loss
        if self.pde_loss is not None:
            pde = self.pde_loss(pred, target, sigma, source, spacing, source_point)
            total_loss = total_loss + pde
            loss_dict["pde_loss"] = pde.item()

        # Spectral smoothing loss
        if self.spectral_loss is not None:
            spectral = self.spectral_loss(pred, target)
            total_loss = total_loss + spectral
            loss_dict["spectral_loss"] = spectral.item()

        # Total variation regularizer
        if self.tv_loss is not None:
            tv = self.tv_loss(pred, source, source_point)
            total_loss = total_loss + tv
            loss_dict["tv_loss"] = tv.item()

        loss_dict["loss"] = total_loss.item()

        return total_loss, loss_dict
