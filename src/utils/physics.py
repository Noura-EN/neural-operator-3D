"""
Physics-informed loss functions for potential field prediction.
Implements MSE, PDE residual, and Charge conservation losses.
"""

import torch
import torch.nn as nn
import numpy as np


def finite_difference_gradient(
    phi: torch.Tensor,
    h: float,
    dim: int
) -> torch.Tensor:
    """
    Compute gradient using central difference in specified dimension.
    
    Args:
        phi: Potential field (B, 1, D, H, W)
        h: Grid spacing
        dim: Dimension along which to compute gradient (0=D, 1=H, 2=W)
    
    Returns:
        Gradient component (B, 1, D, H, W)
    """
    # Central difference: (f(x+h) - f(x-h)) / (2h)
    if dim == 0:  # D dimension
        grad = torch.zeros_like(phi)
        grad[:, :, 1:-1, :, :] = (phi[:, :, 2:, :, :] - phi[:, :, :-2, :, :]) / (2.0 * h)
        # Forward/backward differences at boundaries
        grad[:, :, 0, :, :] = (phi[:, :, 1, :, :] - phi[:, :, 0, :, :]) / h
        grad[:, :, -1, :, :] = (phi[:, :, -1, :, :] - phi[:, :, -2, :, :]) / h
    elif dim == 1:  # H dimension
        grad = torch.zeros_like(phi)
        grad[:, :, :, 1:-1, :] = (phi[:, :, :, 2:, :] - phi[:, :, :, :-2, :]) / (2.0 * h)
        grad[:, :, :, 0, :] = (phi[:, :, :, 1, :] - phi[:, :, :, 0, :]) / h
        grad[:, :, :, -1, :] = (phi[:, :, :, -1, :] - phi[:, :, :, -2, :]) / h
    else:  # W dimension
        grad = torch.zeros_like(phi)
        grad[:, :, :, :, 1:-1] = (phi[:, :, :, :, 2:] - phi[:, :, :, :, :-2]) / (2.0 * h)
        grad[:, :, :, :, 0] = (phi[:, :, :, :, 1] - phi[:, :, :, :, 0]) / h
        grad[:, :, :, :, -1] = (phi[:, :, :, :, -1] - phi[:, :, :, :, -2]) / h
    
    return grad


def compute_divergence(
    vector_field: torch.Tensor,
    h: float
) -> torch.Tensor:
    """
    Compute divergence of a 3D vector field using central differences.
    
    Args:
        vector_field: Vector field (B, 3, D, H, W) where channels are [Fx, Fy, Fz]
        h: Grid spacing
    
    Returns:
        Divergence (B, 1, D, H, W)
    """
    B, _, D, H, W = vector_field.shape
    
    # Compute gradients in each direction
    grad_x = finite_difference_gradient(vector_field[:, 0:1, :, :, :], h, dim=0)
    grad_y = finite_difference_gradient(vector_field[:, 1:2, :, :, :], h, dim=1)
    grad_z = finite_difference_gradient(vector_field[:, 2:3, :, :, :], h, dim=2)
    
    divergence = grad_x + grad_y + grad_z
    return divergence


def pde_residual(
    phi: torch.Tensor,
    conductivity: torch.Tensor,
    source: torch.Tensor,
    grid_resolution: tuple,
    conductivity_threshold: float = 1e-6
) -> torch.Tensor:
    """
    Compute PDE residual: ∇·(σ∇Φ) - f = 0
    
    Uses 7-point central difference stencil for divergence of gradient.
    
    Args:
        phi: Predicted potential (B, 1, D, H, W)
        conductivity: Conductivity field (B, 1, D, H, W)
        source: Source term (B, 1, D, H, W)
        grid_resolution: (D, H, W) tuple
        conductivity_threshold: Only compute residual where sigma > threshold
    
    Returns:
        PDE residual (B, 1, D, H, W)
    """
    D, H, W = grid_resolution
    h = 1.0 / max(D - 1, H - 1, W - 1)  # Grid spacing in normalized coordinates
    
    # Compute gradient of potential
    grad_phi_x = finite_difference_gradient(phi, h, dim=0)
    grad_phi_y = finite_difference_gradient(phi, h, dim=1)
    grad_phi_z = finite_difference_gradient(phi, h, dim=2)
    grad_phi = torch.cat([grad_phi_x, grad_phi_y, grad_phi_z], dim=1)  # (B, 3, D, H, W)
    
    # Compute current density: J = -σ∇Φ
    current_density = -conductivity * grad_phi  # (B, 3, D, H, W)
    
    # Compute divergence of current density
    div_J = compute_divergence(current_density, h)  # (B, 1, D, H, W)
    
    # PDE residual: ∇·(σ∇Φ) - f = -∇·J - f = 0
    residual = -div_J - source
    
    # Mask out regions with low conductivity
    mask = conductivity > conductivity_threshold
    residual = residual * mask
    
    return residual


def mse_loss_with_singularity_mask(
    pred: torch.Tensor,
    target: torch.Tensor,
    source: torch.Tensor,
    mask_radius: int = 3,
    coord_range: tuple = (-1.0, 1.0)
) -> torch.Tensor:
    """
    Compute MSE loss with masking around source singularity.
    
    Args:
        pred: Predicted potential (B, 1, D, H, W)
        target: Ground truth potential (B, 1, D, H, W)
        source: Source tensor (B, 1, D, H, W)
        mask_radius: Voxel radius to mask around source
        coord_range: Range of normalized coordinates
    
    Returns:
        Masked MSE loss (scalar)
    """
    from .analytical import compute_source_centroid
    
    B, _, D, H, W = pred.shape
    device = pred.device
    
    # Compute source centroids in voxel coordinates
    source_centroids_norm = compute_source_centroid(source)  # (B, 3) in [-1, 1]
    
    # Convert to voxel indices
    source_centroids_vox = torch.stack([
        (source_centroids_norm[:, 0] + 1.0) * (D - 1) / 2.0,
        (source_centroids_norm[:, 1] + 1.0) * (H - 1) / 2.0,
        (source_centroids_norm[:, 2] + 1.0) * (W - 1) / 2.0
    ], dim=1)  # (B, 3)
    
    # Create mask (1 where we compute loss, 0 where masked)
    mask = torch.ones(B, 1, D, H, W, device=device, dtype=torch.bool)
    
    for b in range(B):
        center = source_centroids_vox[b].long()  # (3,)
        center_d, center_h, center_w = center[0].item(), center[1].item(), center[2].item()
        
        # Create spherical mask
        d_coords = torch.arange(D, device=device).float()
        h_coords = torch.arange(H, device=device).float()
        w_coords = torch.arange(W, device=device).float()
        
        dd, hh, ww = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
        dist_sq = (dd - center_d) ** 2 + (hh - center_h) ** 2 + (ww - center_w) ** 2
        mask[b, 0] = dist_sq > (mask_radius ** 2)
    
    # Compute masked MSE
    mse = (pred - target) ** 2
    masked_mse = (mse * mask.float()).sum() / mask.float().sum()
    
    return masked_mse


def charge_conservation_loss(
    phi: torch.Tensor,
    conductivity: torch.Tensor,
    source: torch.Tensor,
    grid_resolution: tuple
) -> torch.Tensor:
    """
    Compute global charge conservation loss.
    
    Loss = |D_total - S_total|²
    where D_total = Σ[∇·(σ∇Φ)] and S_total = Σ[f]
    
    Args:
        phi: Predicted potential (B, 1, D, H, W)
        conductivity: Conductivity field (B, 1, D, H, W)
        source: Source term (B, 1, D, H, W)
        grid_resolution: (D, H, W) tuple
    
    Returns:
        Charge conservation loss (scalar)
    """
    # Compute PDE residual
    residual = pde_residual(phi, conductivity, source, grid_resolution)
    
    # Total divergence (sum over all voxels)
    D_total = residual.sum(dim=(2, 3, 4))  # (B, 1)
    
    # Total source (sum over all voxels)
    S_total = source.sum(dim=(2, 3, 4))  # (B, 1)
    
    # Conservation loss
    loss = ((D_total - S_total) ** 2).mean()
    
    return loss


def compute_total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    conductivity: torch.Tensor,
    source: torch.Tensor,
    grid_resolution: tuple,
    loss_weights: dict,
    config: dict
) -> dict:
    """
    Compute total physics-informed loss.
    
    Args:
        pred: Predicted potential (B, 1, D, H, W)
        target: Ground truth potential (B, 1, D, H, W)
        conductivity: Conductivity field (B, 1, D, H, W)
        source: Source term (B, 1, D, H, W)
        grid_resolution: (D, H, W) tuple
        loss_weights: Dictionary with keys 'mse', 'pde', 'charge'
        config: Configuration dictionary
    
    Returns:
        Dictionary with individual losses and total loss
    """
    losses = {}
    
    # MSE Loss with singularity mask
    if loss_weights.get('mse', 0) > 0:
        losses['mse'] = mse_loss_with_singularity_mask(
            pred, target, source,
            mask_radius=config['physics']['mse']['singularity_mask_radius'],
            coord_range=tuple(config['grid']['coord_range'])
        )
    
    # PDE Loss
    if loss_weights.get('pde', 0) > 0:
        residual = pde_residual(
            pred, conductivity, source, grid_resolution,
            conductivity_threshold=config['physics']['pde']['conductivity_threshold']
        )
        losses['pde'] = (residual ** 2).mean()
    
    # Charge Conservation Loss
    if loss_weights.get('charge', 0) > 0 and config['physics']['charge']['enabled']:
        losses['charge'] = charge_conservation_loss(
            pred, conductivity, source, grid_resolution
        )
    
    # Total weighted loss
    total_loss = sum(loss_weights.get(k, 0) * v for k, v in losses.items())
    losses['total'] = total_loss
    
    return losses
