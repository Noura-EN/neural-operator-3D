"""Unit tests for masking and loss utilities."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.masking import (
    create_muscle_mask,
    create_singularity_mask,
    create_combined_mask,
    WeightedMaskedMSELoss,
    GradientLoss,
    CombinedLoss,
)


class TestMuscleMask:
    """Tests for muscle mask creation."""

    def test_mask_shape(self):
        """Test output mask shape."""
        sigma = torch.randn(2, 6, 16, 16, 16)
        mask = create_muscle_mask(sigma)

        assert mask.shape == (2, 1, 16, 16, 16)

    def test_identifies_muscle(self):
        """Test that muscle regions are correctly identified."""
        # Create sigma with known muscle values
        sigma = torch.zeros(1, 6, 4, 4, 4)
        muscle_values = (0.2455, 0.2455, 1.2275)

        # Set one voxel to muscle values
        sigma[0, 0, 1, 1, 1] = muscle_values[0]
        sigma[0, 1, 1, 1, 1] = muscle_values[1]
        sigma[0, 2, 1, 1, 1] = muscle_values[2]

        mask = create_muscle_mask(sigma, muscle_values)

        # Check that muscle voxel is identified
        assert mask[0, 0, 1, 1, 1] == 1.0

        # Check that other voxels are not identified as muscle
        assert mask[0, 0, 0, 0, 0] == 0.0

    def test_binary_output(self):
        """Test that mask is binary."""
        sigma = torch.randn(2, 6, 8, 8, 8)
        mask = create_muscle_mask(sigma)

        unique_values = torch.unique(mask)
        assert all(v in [0.0, 1.0] for v in unique_values.tolist())


class TestSingularityMask:
    """Tests for singularity mask creation."""

    def test_mask_shape(self):
        """Test output mask shape."""
        source = torch.randn(2, 1, 16, 16, 16)
        mask = create_singularity_mask(source, radius=3)

        assert mask.shape == (2, 1, 16, 16, 16)

    def test_spherical_shape(self):
        """Test that mask is approximately spherical."""
        # Create source with peak at center
        source = torch.zeros(1, 1, 16, 16, 16)
        source[0, 0, 8, 8, 8] = 1.0

        mask = create_singularity_mask(source, radius=2)

        # Center should be masked
        assert mask[0, 0, 8, 8, 8] == 1.0

        # Check approximate radius (points at distance 2 should be masked)
        assert mask[0, 0, 8, 8, 10] == 1.0  # Within radius
        assert mask[0, 0, 8, 8, 11] == 0.0  # Outside radius

    def test_provided_source_point(self):
        """Test with explicitly provided source point."""
        source = torch.zeros(1, 1, 16, 16, 16)
        source_point = torch.tensor([[4.0, 4.0, 4.0]])

        mask = create_singularity_mask(source, radius=2, source_point=source_point)

        assert mask[0, 0, 4, 4, 4] == 1.0


class TestCombinedMask:
    """Tests for combined mask creation."""

    def test_mask_shape(self):
        """Test output mask shape."""
        sigma = torch.randn(2, 6, 16, 16, 16)
        source = torch.randn(2, 1, 16, 16, 16)

        mask = create_combined_mask(sigma, source)

        assert mask.shape == (2, 1, 16, 16, 16)

    def test_excludes_singularity(self):
        """Test that singularity region is excluded."""
        # Create sigma with muscle everywhere
        sigma = torch.zeros(1, 6, 16, 16, 16)
        muscle_values = (0.2455, 0.2455, 1.2275)
        sigma[:, 0, :, :, :] = muscle_values[0]
        sigma[:, 1, :, :, :] = muscle_values[1]
        sigma[:, 2, :, :, :] = muscle_values[2]

        # Create source with peak at center
        source = torch.zeros(1, 1, 16, 16, 16)
        source[0, 0, 8, 8, 8] = 1.0

        mask = create_combined_mask(sigma, source, singularity_radius=2)

        # Center should be excluded (part of singularity)
        assert mask[0, 0, 8, 8, 8] == 0.0

        # Muscle regions away from singularity should be included
        assert mask[0, 0, 0, 0, 0] == 1.0


class TestWeightedMaskedMSELoss:
    """Tests for weighted masked MSE loss."""

    def test_loss_computation(self):
        """Test basic loss computation."""
        loss_fn = WeightedMaskedMSELoss(weight=1.0, singularity_radius=2)

        pred = torch.randn(1, 1, 8, 8, 8)
        target = torch.randn(1, 1, 8, 8, 8)
        sigma = torch.randn(1, 6, 8, 8, 8)
        source = torch.randn(1, 1, 8, 8, 8)

        loss = loss_fn(pred, target, sigma, source)

        assert loss.dim() == 0  # Scalar
        assert loss >= 0  # MSE is non-negative

    def test_zero_loss_for_identical(self):
        """Test that loss is zero when pred equals target (in masked region)."""
        loss_fn = WeightedMaskedMSELoss(weight=1.0, use_muscle_mask=False, singularity_radius=0)

        pred = torch.ones(1, 1, 8, 8, 8)
        target = torch.ones(1, 1, 8, 8, 8)
        sigma = torch.randn(1, 6, 8, 8, 8)
        source = torch.zeros(1, 1, 8, 8, 8)

        loss = loss_fn(pred, target, sigma, source)

        assert loss.item() < 1e-6

    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""
        loss_fn = WeightedMaskedMSELoss()

        pred = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
        target = torch.randn(1, 1, 8, 8, 8)
        sigma = torch.randn(1, 6, 8, 8, 8)
        source = torch.randn(1, 1, 8, 8, 8)

        loss = loss_fn(pred, target, sigma, source)
        loss.backward()

        assert pred.grad is not None


class TestGradientLoss:
    """Tests for gradient consistency loss."""

    def test_loss_computation(self):
        """Test basic gradient loss computation."""
        loss_fn = GradientLoss(weight=0.1)

        pred = torch.randn(1, 1, 8, 8, 8)
        target = torch.randn(1, 1, 8, 8, 8)
        spacing = torch.ones(1, 3)

        loss = loss_fn(pred, target, spacing)

        assert loss.dim() == 0
        assert loss >= 0

    def test_gradient_flow(self):
        """Test gradient flow through loss."""
        loss_fn = GradientLoss(weight=0.1)

        pred = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
        target = torch.randn(1, 1, 8, 8, 8)
        spacing = torch.ones(1, 3)

        loss = loss_fn(pred, target, spacing)
        loss.backward()

        assert pred.grad is not None


class TestCombinedLoss:
    """Tests for combined loss function."""

    def test_loss_computation(self):
        """Test combined loss computation."""
        loss_fn = CombinedLoss(mse_weight=1.0, grad_weight=0.1)

        pred = torch.randn(1, 1, 8, 8, 8)
        target = torch.randn(1, 1, 8, 8, 8)
        sigma = torch.randn(1, 6, 8, 8, 8)
        source = torch.randn(1, 1, 8, 8, 8)
        spacing = torch.ones(1, 3)

        loss, loss_dict = loss_fn(pred, target, sigma, source, spacing)

        assert loss.dim() == 0
        assert 'loss' in loss_dict
        assert 'mse_loss' in loss_dict
        assert 'grad_loss' in loss_dict

    def test_loss_dict_values(self):
        """Test that loss dict contains correct values."""
        loss_fn = CombinedLoss(mse_weight=1.0, grad_weight=0.1)

        pred = torch.randn(1, 1, 8, 8, 8)
        target = torch.randn(1, 1, 8, 8, 8)
        sigma = torch.randn(1, 6, 8, 8, 8)
        source = torch.randn(1, 1, 8, 8, 8)
        spacing = torch.ones(1, 3)

        loss, loss_dict = loss_fn(pred, target, sigma, source, spacing)

        # Total loss should be sum of components
        assert abs(loss_dict['loss'] - (loss_dict['mse_loss'] + loss_dict['grad_loss'])) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
