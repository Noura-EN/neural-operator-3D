"""Unit tests for data loading utilities."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import PotentialFieldDataset, create_data_splits
from src.data.transforms import resample_volume, resample_batch


class TestPotentialFieldDataset:
    """Tests for PotentialFieldDataset."""

    @pytest.fixture
    def sample_data_dir(self, tmp_path):
        """Create temporary directory with sample data files."""
        # Create sample npz files
        for i in range(5):
            D, H, W = 16, 8, 8
            data = {
                'sigma': np.random.randn(D, H, W, 6).astype(np.float32),
                'source': np.random.randn(D, H, W).astype(np.float32),
                'mask': np.ones((D, H, W), dtype=np.uint8),
                'u': np.random.randn(D, H, W).astype(np.float32),
                'spacing': np.array([1.0, 1.0, 1.0], dtype=np.float32),
                'source_point': np.array([D // 2, H // 2, W // 2], dtype=np.float32),
            }
            np.savez(tmp_path / f"sample_{i:06d}.npz", **data)

        return str(tmp_path)

    def test_dataset_length(self, sample_data_dir):
        """Test dataset returns correct length."""
        dataset = PotentialFieldDataset(sample_data_dir)
        assert len(dataset) == 5

    def test_dataset_output_shapes(self, sample_data_dir):
        """Test dataset returns correct tensor shapes."""
        dataset = PotentialFieldDataset(sample_data_dir)
        sample = dataset[0]

        assert sample['sigma'].shape[0] == 6  # 6 channels
        assert sample['sigma'].dim() == 4  # (C, D, H, W)
        assert sample['source'].shape[0] == 1  # 1 channel
        assert sample['coords'].shape[0] == 3  # 3 channels (X, Y, Z)
        assert sample['spacing'].shape[0] == 3  # (dx, dy, dz)
        assert sample['mask'].shape[0] == 1
        assert sample['target'].shape[0] == 1

    def test_coordinate_generation(self, sample_data_dir):
        """Test that coordinates are properly generated in [-1, 1]."""
        dataset = PotentialFieldDataset(sample_data_dir, coord_range=(-1.0, 1.0))
        sample = dataset[0]

        coords = sample['coords']
        assert coords.min() >= -1.0
        assert coords.max() <= 1.0

    def test_sample_indices(self, sample_data_dir):
        """Test filtering by sample indices."""
        dataset = PotentialFieldDataset(sample_data_dir, sample_indices=[0, 2, 4])
        assert len(dataset) == 3

    def test_tensor_types(self, sample_data_dir):
        """Test that all outputs are proper torch tensors."""
        dataset = PotentialFieldDataset(sample_data_dir)
        sample = dataset[0]

        for key, value in sample.items():
            assert isinstance(value, torch.Tensor), f"{key} is not a tensor"


class TestDataSplits:
    """Tests for data split creation."""

    @pytest.fixture
    def sample_data_dir(self, tmp_path):
        """Create temporary directory with sample data files."""
        for i in range(100):
            np.savez(tmp_path / f"sample_{i:06d}.npz",
                     sigma=np.zeros((4, 4, 4, 6)),
                     source=np.zeros((4, 4, 4)),
                     mask=np.ones((4, 4, 4)),
                     u=np.zeros((4, 4, 4)),
                     spacing=np.ones(3),
                     source_point=np.ones(3))
        return str(tmp_path)

    def test_split_sizes(self, sample_data_dir):
        """Test that splits have correct sizes."""
        train, val, test = create_data_splits(
            sample_data_dir,
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
        )

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_no_overlap(self, sample_data_dir):
        """Test that splits don't overlap."""
        train, val, test = create_data_splits(sample_data_dir)

        all_indices = set(train) | set(val) | set(test)
        assert len(all_indices) == len(train) + len(val) + len(test)

    def test_reproducibility(self, sample_data_dir):
        """Test that same seed produces same splits."""
        train1, val1, test1 = create_data_splits(sample_data_dir, seed=42)
        train2, val2, test2 = create_data_splits(sample_data_dir, seed=42)

        assert train1 == train2
        assert val1 == val2
        assert test1 == test2


class TestResampleVolume:
    """Tests for volume resampling."""

    def test_upsample(self):
        """Test upsampling a volume."""
        volume = torch.randn(4, 8, 8, 8)
        resampled = resample_volume(volume, (16, 16, 16))

        assert resampled.shape == (4, 16, 16, 16)

    def test_downsample(self):
        """Test downsampling a volume."""
        volume = torch.randn(4, 16, 16, 16)
        resampled = resample_volume(volume, (8, 8, 8))

        assert resampled.shape == (4, 8, 8, 8)

    def test_batch_dimension(self):
        """Test resampling with batch dimension."""
        volume = torch.randn(2, 4, 8, 8, 8)
        resampled = resample_volume(volume, (16, 16, 16))

        assert resampled.shape == (2, 4, 16, 16, 16)


class TestResampleBatch:
    """Tests for batch resampling."""

    def test_resample_all_fields(self):
        """Test that all fields are resampled correctly."""
        batch = {
            'sigma': torch.randn(2, 6, 8, 8, 8),
            'source': torch.randn(2, 1, 8, 8, 8),
            'coords': torch.randn(2, 3, 8, 8, 8),
            'mask': torch.ones(2, 1, 8, 8, 8),
            'target': torch.randn(2, 1, 8, 8, 8),
            'spacing': torch.ones(2, 3),
        }

        target_shape = (16, 16, 16)
        resampled = resample_batch(batch, target_shape)

        assert resampled['sigma'].shape[-3:] == target_shape
        assert resampled['source'].shape[-3:] == target_shape
        assert resampled['coords'].shape[-3:] == target_shape
        assert resampled['mask'].shape[-3:] == target_shape
        assert resampled['target'].shape[-3:] == target_shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
