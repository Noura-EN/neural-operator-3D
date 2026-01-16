"""
Analytical solver for infinite-domain potential field.
Implements the point-source solution in a homogeneous domain.
"""

import torch
import torch.nn as nn
import numpy as np


def compute_source_centroid(source: torch.Tensor) -> torch.Tensor:
    """
    Calculate the center of mass of non-zero voxels in the source input.
    
    Args:
        source: Source tensor of shape (B, 1, D, H, W) or (1, D, H, W)
    
    Returns:
        Centroid coordinates (B, 3) or (3,) in normalized coordinates [-1, 1]
    """
    if source.dim() == 5:
        batch_size = source.shape[0]
        centroids = []
        for b in range(batch_size):
            src = source[b, 0]  # (D, H, W)
            non_zero = src > 1e-6
            if non_zero.sum() == 0:
                # Fallback to center if no source found
                centroids.append(torch.tensor([0.0, 0.0, 0.0], device=source.device))
            else:
                indices = torch.nonzero(non_zero, as_tuple=False).float()
                weights = src[non_zero]
                centroid = (indices * weights.unsqueeze(1)).sum(0) / weights.sum()
                # Normalize to [-1, 1]
                D, H, W = src.shape
                centroid_normalized = torch.stack([
                    2.0 * centroid[0] / (D - 1) - 1.0,
                    2.0 * centroid[1] / (H - 1) - 1.0,
                    2.0 * centroid[2] / (W - 1) - 1.0
                ])
                centroids.append(centroid_normalized)
        return torch.stack(centroids)
    else:
        # Single sample
        non_zero = source[0] > 1e-6
        if non_zero.sum() == 0:
            return torch.tensor([0.0, 0.0, 0.0], device=source.device)
        indices = torch.nonzero(non_zero, as_tuple=False).float()
        weights = source[0][non_zero]
        centroid = (indices * weights.unsqueeze(1)).sum(0) / weights.sum()
        D, H, W = source.shape[1:]
        centroid_normalized = torch.stack([
            2.0 * centroid[0] / (D - 1) - 1.0,
            2.0 * centroid[1] / (H - 1) - 1.0,
            2.0 * centroid[2] / (W - 1) - 1.0
        ])
        return centroid_normalized


def compute_reference_conductivity(
    conductivity: torch.Tensor, 
    threshold: float = 1e-6
) -> torch.Tensor:
    """
    Calculate mean conductivity of conductive tissues only (ignore air/zero voxels).
    
    Args:
        conductivity: Conductivity tensor of shape (B, 1, D, H, W) or (1, D, H, W)
        threshold: Minimum conductivity to consider as conductive tissue
    
    Returns:
        Reference conductivity (B,) or scalar
    """
    if conductivity.dim() == 5:
        batch_size = conductivity.shape[0]
        ref_conductivities = []
        for b in range(batch_size):
            sigma = conductivity[b, 0]  # (D, H, W)
            conductive_mask = sigma > threshold
            if conductive_mask.sum() > 0:
                ref_sigma = sigma[conductive_mask].mean()
            else:
                ref_sigma = torch.tensor(1.0, device=conductivity.device)
            ref_conductivities.append(ref_sigma)
        return torch.stack(ref_conductivities)
    else:
        sigma = conductivity[0]
        conductive_mask = sigma > threshold
        if conductive_mask.sum() > 0:
            return sigma[conductive_mask].mean()
        else:
            return torch.tensor(1.0, device=conductivity.device)


def analytical_potential(
    coords: torch.Tensor,
    source: torch.Tensor,
    conductivity: torch.Tensor,
    current_I: float = 1.0,
    epsilon_factor: float = 0.1,
    coord_range: tuple = (-1.0, 1.0)
) -> torch.Tensor:
    """
    Compute analytical potential field for point source in infinite homogeneous domain.
    
    Formula: Φ_analytical(r) = I / (4π σ_ref √(|r - r_s|² + ε²))
    
    Args:
        coords: Coordinate grids (B, 3, D, H, W) where channels are [X, Y, Z]
        source: Source tensor (B, 1, D, H, W)
        conductivity: Conductivity tensor (B, 1, D, H, W)
        current_I: Source current magnitude
        epsilon_factor: Epsilon = epsilon_factor * voxel_size
        coord_range: Range of normalized coordinates (min, max)
    
    Returns:
        Analytical potential field (B, 1, D, H, W)
    """
    device = coords.device
    B, _, D, H, W = coords.shape
    
    # Compute source centroid
    source_centroids = compute_source_centroid(source)  # (B, 3)
    
    # Compute reference conductivity
    ref_conductivities = compute_reference_conductivity(conductivity)  # (B,)
    
    # Compute voxel size in normalized coordinates
    voxel_size = (coord_range[1] - coord_range[0]) / max(D, H, W)
    epsilon = epsilon_factor * voxel_size
    
    # Expand centroids to match grid dimensions
    source_centroids = source_centroids.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, 3, 1, 1, 1)
    ref_conductivities = ref_conductivities.view(B, 1, 1, 1, 1)
    
    # Compute distance from each point to source
    # coords: (B, 3, D, H, W), source_centroids: (B, 3, 1, 1, 1)
    r_diff = coords - source_centroids  # (B, 3, D, H, W)
    r_squared = (r_diff ** 2).sum(dim=1, keepdim=True)  # (B, 1, D, H, W)
    
    # Compute analytical potential
    denominator = 4.0 * np.pi * ref_conductivities * torch.sqrt(r_squared + epsilon ** 2)
    phi_analytical = current_I / denominator
    
    return phi_analytical
