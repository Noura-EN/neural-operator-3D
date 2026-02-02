"""Data loading utilities for 3D potential field prediction."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from scipy import ndimage


def log_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply signed log transform to compress dynamic range.

    Transform: sign(x) * log(1 + |x|)

    This compresses the range from [~10^-5, ~10^1] to [~0, ~2.4]
    while preserving sign information.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def inverse_log_transform(x: torch.Tensor) -> torch.Tensor:
    """Inverse of log_transform.

    Transform: sign(x) * (exp(|x|) - 1)
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def load_exclusion_list(exclusion_file: Union[str, Path]) -> Set[str]:
    """Load list of sample filenames to exclude.

    Args:
        exclusion_file: Path to exclusion list file (one filename per line, # for comments)

    Returns:
        Set of filenames to exclude
    """
    exclusion_file = Path(exclusion_file)
    if not exclusion_file.exists():
        return set()

    excluded = set()
    with open(exclusion_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle CSV format (filename,ratio,umax)
                filename = line.split(',')[0].strip()
                excluded.add(filename)

    return excluded


class PotentialFieldDataset(Dataset):
    """Dataset for loading 3D potential field data from .npz files.

    Each sample contains:
        - sigma: Conductivity tensor (6 channels: 3 diagonal + 3 off-diagonal)
        - source: Source field f
        - mask: Binary mask for valid regions
        - u: FEM ground truth potential
        - spacing: Physical voxel spacing
        - source_point: Location of the source
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        sample_indices: Optional[List[int]] = None,
        device: str = "cpu",
        add_spacing_channels: bool = False,
        add_analytical_solution: bool = False,
        add_distance_field: bool = False,
        normalize_target: bool = False,
        singularity_percentile: float = 99.0,
        log_transform_target: bool = False,
    ):
        """Initialize dataset.

        Args:
            data_dir: Directory containing .npz sample files
            sample_indices: List of sample indices to use (for train/val/test splits)
            device: Device to load tensors to
            add_spacing_channels: If True, add spacing as explicit input channels
            add_analytical_solution: If True, compute and add monopole analytical solution
            add_distance_field: If True, compute and add signed distance to muscle boundary
            normalize_target: If True, normalize target by muscle-region mean/std (excluding singularity)
            singularity_percentile: Percentile threshold for singularity mask (exclude top X%)
            log_transform_target: If True, apply signed log transform to compress dynamic range
        """
        self.data_dir = Path(data_dir)
        self.device = device
        self.add_spacing_channels = add_spacing_channels
        self.add_analytical_solution = add_analytical_solution
        self.add_distance_field = add_distance_field
        self.normalize_target = normalize_target
        self.singularity_percentile = singularity_percentile
        self.log_transform_target = log_transform_target

        # Find all sample files
        all_files = sorted(self.data_dir.glob("sample_*.npz"))

        if sample_indices is not None:
            self.sample_files = [all_files[i] for i in sample_indices if i < len(all_files)]
        else:
            self.sample_files = all_files

        if len(self.sample_files) == 0:
            raise ValueError(f"No sample files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.sample_files)

    def _generate_coords(self, shape: Tuple[int, int, int]) -> torch.Tensor:
        """Generate normalized coordinate grids [-1, 1].

        Resolution independence is achieved through:
        - Normalized coords: encode relative position in domain
        - Spacing conditioning (separate): encode physical voxel size

        Args:
            shape: Grid shape (D, H, W)

        Returns:
            Coordinate tensor of shape (3, D, H, W) with X, Y, Z in [-1, 1]
        """
        D, H, W = shape

        z = torch.linspace(-1, 1, D)
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)

        Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
        coords = torch.stack([X, Y, Z], dim=0)

        return coords

    def _compute_distance_to_boundary(
        self,
        mask: np.ndarray,
        spacing: np.ndarray,
    ) -> torch.Tensor:
        """Compute signed distance field from muscle boundary.

        Positive values inside muscle, negative outside.
        Normalized by max distance for scale invariance.

        Args:
            mask: Binary muscle mask (D, H, W)
            spacing: Physical voxel spacing (3,)

        Returns:
            Signed distance field (1, D, H, W), normalized to [-1, 1]
        """
        # Compute distance transform for inside and outside
        # Inside muscle (mask=1): positive distance to boundary
        dist_inside = ndimage.distance_transform_edt(mask, sampling=spacing)
        # Outside muscle (mask=0): negative distance to boundary
        dist_outside = ndimage.distance_transform_edt(1 - mask, sampling=spacing)

        # Signed distance: positive inside, negative outside
        signed_dist = dist_inside - dist_outside

        # Normalize to [-1, 1] for network input stability
        max_dist = max(np.abs(signed_dist).max(), 1e-6)
        signed_dist_normalized = signed_dist / max_dist

        return torch.from_numpy(signed_dist_normalized).unsqueeze(0).float()

    def _compute_analytical_solution(
        self,
        sigma: np.ndarray,
        mask: np.ndarray,
        source_point: np.ndarray,
        spacing: np.ndarray,
        grid_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Compute analytical monopole solution Φ = I/(4πσ_avg·r).

        Args:
            sigma: Conductivity tensor (D, H, W, 6) - diagonal components are [0,1,2]
            mask: Binary mask for muscle region (D, H, W)
            source_point: Source location in voxel coordinates (3,)
            spacing: Physical voxel spacing (3,)
            grid_shape: Grid dimensions (D, H, W)

        Returns:
            Analytical solution tensor (1, D, H, W)
        """
        D, H, W = grid_shape

        # Compute σ_avg from muscle region diagonal conductivity
        # Using muscle average because the field propagates through muscle tissue
        # (source point often has artificially low σ due to electrode modeling)
        # sigma has shape (D, H, W, 6) with diagonal at indices 0, 1, 2
        sigma_diag = sigma[..., :3]  # (D, H, W, 3)
        muscle_mask = mask > 0.5

        if muscle_mask.sum() > 0:
            # Average of diagonal components in muscle region
            sigma_avg = sigma_diag[muscle_mask].mean()
        else:
            # Fallback to overall mean
            sigma_avg = sigma_diag.mean()

        # Ensure sigma_avg is positive and reasonable
        sigma_avg = max(sigma_avg, 1e-6)

        # Create coordinate grids in physical space
        z_coords = np.arange(D) * spacing[0]
        y_coords = np.arange(H) * spacing[1]
        x_coords = np.arange(W) * spacing[2]

        Z, Y, X = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

        # Source position in physical coordinates
        src_z = source_point[0] * spacing[0]
        src_y = source_point[1] * spacing[1]
        src_x = source_point[2] * spacing[2]

        # Compute distance from source
        r = np.sqrt((Z - src_z)**2 + (Y - src_y)**2 + (X - src_x)**2)

        # Avoid division by zero at source point
        r = np.maximum(r, spacing.min())

        # Monopole solution: Φ = I / (4πσr)
        # Assuming unit current I = 1
        analytical = 1.0 / (4.0 * np.pi * sigma_avg * r)

        # Convert to tensor (1, D, H, W)
        return torch.from_numpy(analytical).unsqueeze(0).float()

    def _create_valid_mask(
        self,
        target: torch.Tensor,
        muscle_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Create combined mask excluding singularity region.

        Args:
            target: Potential field (1, D, H, W)
            muscle_mask: Muscle region mask (1, D, H, W)

        Returns:
            Valid mask (1, D, H, W) - muscle region excluding singularity
        """
        # Get values within muscle region
        muscle_bool = muscle_mask > 0.5
        muscle_values = target[muscle_bool]

        if muscle_values.numel() == 0:
            return muscle_mask

        # Compute threshold for singularity (top X% of absolute values)
        threshold = torch.quantile(torch.abs(muscle_values), self.singularity_percentile / 100.0)

        # Singularity mask: where |target| > threshold
        singularity_mask = torch.abs(target) > threshold

        # Combined mask: muscle AND NOT singularity
        valid_mask = muscle_bool & ~singularity_mask

        return valid_mask.float()

    def _normalize_target(
        self,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize target by valid-region statistics.

        Args:
            target: Potential field (1, D, H, W)
            valid_mask: Valid region mask (1, D, H, W)

        Returns:
            Tuple of (normalized_target, mean, std)
        """
        valid_bool = valid_mask > 0.5
        valid_values = target[valid_bool]

        if valid_values.numel() == 0:
            # Fallback to full volume stats
            mean = target.mean()
            std = target.std()
        else:
            mean = valid_values.mean()
            std = valid_values.std()

        # Avoid division by zero
        std = torch.clamp(std, min=1e-8)

        # Normalize
        target_normalized = (target - mean) / std

        return target_normalized, mean, std

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample.

        Returns:
            Dictionary containing:
                - sigma: (6, D, H, W) conductivity tensor
                - source: (1, D, H, W) source field
                - coords: (3, D, H, W) normalized coordinates (X, Y, Z)
                - spacing: (3,) physical voxel spacing
                - mask: (1, D, H, W) binary mask (muscle region)
                - valid_mask: (1, D, H, W) valid mask (muscle excluding singularity)
                - target: (1, D, H, W) FEM ground truth (optionally normalized)
                - source_point: (3,) source location
                - target_mean: scalar, mean used for normalization (if normalize_target=True)
                - target_std: scalar, std used for normalization (if normalize_target=True)
        """
        data = np.load(self.sample_files[idx])

        # Load arrays - data is stored as (D, H, W, C) or (D, H, W)
        sigma = data['sigma']  # (D, H, W, 6)
        source = data['source']  # (D, H, W)
        mask = data['mask']  # (D, H, W)
        u = data['u']  # (D, H, W) - FEM ground truth
        spacing = data['spacing']  # (3,)
        source_point = data['source_point']  # (3,)

        # Convert to torch tensors
        # Transpose sigma from (D, H, W, 6) to (6, D, H, W)
        sigma = torch.from_numpy(sigma).permute(3, 0, 1, 2).float()
        source = torch.from_numpy(source).unsqueeze(0).float()  # (1, D, H, W)
        muscle_mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)  # (1, D, H, W)
        target = torch.from_numpy(u).unsqueeze(0).float()  # (1, D, H, W)
        spacing = torch.from_numpy(spacing).float()  # (3,)
        source_point = torch.from_numpy(source_point).float()  # (3,)

        # Create valid mask (muscle excluding singularity)
        valid_mask = self._create_valid_mask(target, muscle_mask)

        # Optionally normalize target
        target_mean = torch.tensor(0.0)
        target_std = torch.tensor(1.0)
        if self.normalize_target:
            target, target_mean, target_std = self._normalize_target(target, valid_mask)

        # Optionally apply log transform to compress dynamic range
        if self.log_transform_target:
            target = log_transform(target)

        # Generate normalized coordinates [-1, 1]
        grid_shape = sigma.shape[1:]  # (D, H, W)
        coords = self._generate_coords(grid_shape)

        result = {
            'sigma': sigma,
            'source': source,
            'coords': coords,
            'spacing': spacing,
            'mask': muscle_mask,
            'valid_mask': valid_mask,
            'target': target,
            'source_point': source_point,
            'target_mean': target_mean,
            'target_std': target_std,
            'log_transformed': torch.tensor(self.log_transform_target),
        }

        # Optionally add spacing as explicit channels
        if self.add_spacing_channels:
            # Broadcast spacing to (3, D, H, W)
            D, H, W = grid_shape
            spacing_channels = spacing.view(3, 1, 1, 1).expand(3, D, H, W)
            result['spacing_channels'] = spacing_channels

        # Optionally add analytical solution
        if self.add_analytical_solution:
            analytical = self._compute_analytical_solution(
                sigma=data['sigma'],  # Use original numpy array
                mask=data['mask'],
                source_point=data['source_point'],
                spacing=data['spacing'],
                grid_shape=grid_shape,
            )
            result['analytical'] = analytical

        # Optionally add distance-to-boundary field
        if self.add_distance_field:
            distance_field = self._compute_distance_to_boundary(
                mask=data['mask'],
                spacing=data['spacing'],
            )
            result['distance_field'] = distance_field

        return result


def create_data_splits(
    data_dir: Union[str, Path],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    exclusion_file: Optional[Union[str, Path]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """Create train/val/test splits from data directory.

    Args:
        data_dir: Directory containing sample files
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        seed: Random seed for reproducibility
        exclusion_file: Optional path to file listing samples to exclude

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    data_dir = Path(data_dir)
    all_files = sorted(data_dir.glob("sample_*.npz"))

    # Load exclusion list if provided
    excluded = set()
    if exclusion_file is not None:
        excluded = load_exclusion_list(exclusion_file)
        if excluded:
            print(f"Loaded {len(excluded)} samples to exclude")

    # Filter out excluded samples, keeping track of valid indices
    valid_indices = []
    for i, f in enumerate(all_files):
        if f.name not in excluded:
            valid_indices.append(i)

    n_samples = len(valid_indices)
    if excluded:
        print(f"Using {n_samples} samples after exclusion (removed {len(all_files) - n_samples})")

    # Set seed for reproducibility
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_samples)

    # Calculate split sizes
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)

    # Map back to original indices
    train_indices = [valid_indices[i] for i in shuffled[:n_train]]
    val_indices = [valid_indices[i] for i in shuffled[n_train:n_train + n_val]]
    test_indices = [valid_indices[i] for i in shuffled[n_train + n_val:]]

    return train_indices, val_indices, test_indices


def create_combined_data_splits(
    data_dirs: List[Union[str, Path]],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    exclusion_file: Optional[Union[str, Path]] = None,
) -> Dict[str, List[Tuple[Path, int]]]:
    """Create train/val/test splits from multiple data directories.

    This function creates splits where each split contains samples from all directories,
    maintaining the same proportions across directories.

    Args:
        data_dirs: List of directories containing sample files
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        seed: Random seed for reproducibility
        exclusion_file: Optional path to file listing samples to exclude

    Returns:
        Dict with keys 'train', 'val', 'test', each containing list of (dir, index) tuples
    """
    rng = np.random.default_rng(seed)

    # Load exclusion list if provided
    excluded = set()
    if exclusion_file is not None:
        excluded = load_exclusion_list(exclusion_file)
        if excluded:
            print(f"Loaded {len(excluded)} samples to exclude")

    splits = {'train': [], 'val': [], 'test': []}
    total_excluded = 0

    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        all_files = sorted(data_dir.glob("sample_*.npz"))

        if len(all_files) == 0:
            continue

        # Filter out excluded samples, keeping track of valid indices
        valid_indices = []
        for i, f in enumerate(all_files):
            if f.name not in excluded:
                valid_indices.append(i)

        n_excluded_here = len(all_files) - len(valid_indices)
        total_excluded += n_excluded_here
        n_samples = len(valid_indices)

        if n_samples == 0:
            continue

        # Shuffle valid indices for this directory
        shuffled = rng.permutation(n_samples)

        # Calculate split sizes
        n_train = int(n_samples * train_split)
        n_val = int(n_samples * val_split)

        train_indices = [valid_indices[i] for i in shuffled[:n_train]]
        val_indices = [valid_indices[i] for i in shuffled[n_train:n_train + n_val]]
        test_indices = [valid_indices[i] for i in shuffled[n_train + n_val:]]

        # Add (directory, index) tuples to splits
        for idx in train_indices:
            splits['train'].append((data_dir, idx))
        for idx in val_indices:
            splits['val'].append((data_dir, idx))
        for idx in test_indices:
            splits['test'].append((data_dir, idx))

    if excluded and total_excluded > 0:
        total_samples = sum(len(s) for s in splits.values())
        print(f"Using {total_samples} samples after exclusion (removed {total_excluded})")

    # Shuffle each split to mix samples from different directories
    for split_name in splits:
        rng.shuffle(splits[split_name])

    return splits


class CombinedPotentialFieldDataset(Dataset):
    """Dataset that combines samples from multiple directories."""

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        add_spacing_channels: bool = False,
        add_analytical_solution: bool = False,
        add_distance_field: bool = False,
        normalize_target: bool = False,
        singularity_percentile: float = 99.0,
        log_transform_target: bool = False,
    ):
        """Initialize combined dataset.

        Args:
            samples: List of (directory, sample_index) tuples
            add_spacing_channels: If True, add spacing as explicit input channels
            add_analytical_solution: If True, add monopole analytical solution
            add_distance_field: If True, compute and add signed distance to muscle boundary
            normalize_target: If True, normalize target by muscle-region mean/std (excluding singularity)
            singularity_percentile: Percentile threshold for singularity mask (exclude top X%)
            log_transform_target: If True, apply signed log transform to compress dynamic range
        """
        self.samples = samples
        self.add_spacing_channels = add_spacing_channels
        self.add_analytical_solution = add_analytical_solution
        self.add_distance_field = add_distance_field
        self.normalize_target = normalize_target
        self.singularity_percentile = singularity_percentile
        self.log_transform_target = log_transform_target

        # Build file list
        self.sample_files = []
        for data_dir, idx in samples:
            all_files = sorted(Path(data_dir).glob("sample_*.npz"))
            if idx < len(all_files):
                self.sample_files.append(all_files[idx])

    def __len__(self) -> int:
        return len(self.sample_files)

    def _generate_coords(self, shape: Tuple[int, int, int]) -> torch.Tensor:
        """Generate normalized coordinate grids [-1, 1]."""
        D, H, W = shape

        z = torch.linspace(-1, 1, D)
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)

        Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
        coords = torch.stack([X, Y, Z], dim=0)

        return coords

    def _compute_analytical_solution(
        self,
        sigma: np.ndarray,
        mask: np.ndarray,
        source_point: np.ndarray,
        spacing: np.ndarray,
        grid_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Compute analytical monopole solution using muscle average conductivity."""
        D, H, W = grid_shape
        sigma_diag = sigma[..., :3]
        muscle_mask = mask > 0.5

        if muscle_mask.sum() > 0:
            sigma_avg = sigma_diag[muscle_mask].mean()
        else:
            sigma_avg = sigma_diag.mean()

        sigma_avg = max(sigma_avg, 1e-6)

        z_coords = np.arange(D) * spacing[0]
        y_coords = np.arange(H) * spacing[1]
        x_coords = np.arange(W) * spacing[2]

        Z, Y, X = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

        src_z = source_point[0] * spacing[0]
        src_y = source_point[1] * spacing[1]
        src_x = source_point[2] * spacing[2]

        r = np.sqrt((Z - src_z)**2 + (Y - src_y)**2 + (X - src_x)**2)
        r = np.maximum(r, spacing.min())

        analytical = 1.0 / (4.0 * np.pi * sigma_avg * r)
        return torch.from_numpy(analytical).unsqueeze(0).float()

    def _compute_distance_to_boundary(
        self,
        mask: np.ndarray,
        spacing: np.ndarray,
    ) -> torch.Tensor:
        """Compute signed distance field from muscle boundary."""
        dist_inside = ndimage.distance_transform_edt(mask, sampling=spacing)
        dist_outside = ndimage.distance_transform_edt(1 - mask, sampling=spacing)
        signed_dist = dist_inside - dist_outside
        max_dist = max(np.abs(signed_dist).max(), 1e-6)
        signed_dist_normalized = signed_dist / max_dist
        return torch.from_numpy(signed_dist_normalized).unsqueeze(0).float()

    def _create_valid_mask(
        self,
        target: torch.Tensor,
        muscle_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Create combined mask excluding singularity region."""
        muscle_bool = muscle_mask > 0.5
        muscle_values = target[muscle_bool]

        if muscle_values.numel() == 0:
            return muscle_mask

        threshold = torch.quantile(torch.abs(muscle_values), self.singularity_percentile / 100.0)
        singularity_mask = torch.abs(target) > threshold
        valid_mask = muscle_bool & ~singularity_mask

        return valid_mask.float()

    def _normalize_target(
        self,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize target by valid-region statistics."""
        valid_bool = valid_mask > 0.5
        valid_values = target[valid_bool]

        if valid_values.numel() == 0:
            mean = target.mean()
            std = target.std()
        else:
            mean = valid_values.mean()
            std = valid_values.std()

        std = torch.clamp(std, min=1e-8)
        target_normalized = (target - mean) / std

        return target_normalized, mean, std

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample."""
        data = np.load(self.sample_files[idx])

        sigma = data['sigma']
        source = data['source']
        mask = data['mask']
        u = data['u']
        spacing = data['spacing']
        source_point = data['source_point']

        sigma = torch.from_numpy(sigma).permute(3, 0, 1, 2).float()
        source = torch.from_numpy(source).unsqueeze(0).float()
        muscle_mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        target = torch.from_numpy(u).unsqueeze(0).float()
        spacing_tensor = torch.from_numpy(spacing).float()
        source_point_tensor = torch.from_numpy(source_point).float()

        # Create valid mask (muscle excluding singularity)
        valid_mask = self._create_valid_mask(target, muscle_mask)

        # Optionally normalize target
        target_mean = torch.tensor(0.0)
        target_std = torch.tensor(1.0)
        if self.normalize_target:
            target, target_mean, target_std = self._normalize_target(target, valid_mask)

        # Optionally apply log transform to compress dynamic range
        if self.log_transform_target:
            target = log_transform(target)

        grid_shape = sigma.shape[1:]
        coords = self._generate_coords(grid_shape)

        result = {
            'sigma': sigma,
            'source': source,
            'coords': coords,
            'spacing': spacing_tensor,
            'mask': muscle_mask,
            'valid_mask': valid_mask,
            'target': target,
            'source_point': source_point_tensor,
            'target_mean': target_mean,
            'target_std': target_std,
            'log_transformed': torch.tensor(self.log_transform_target),
        }

        if self.add_spacing_channels:
            D, H, W = grid_shape
            spacing_channels = spacing_tensor.view(3, 1, 1, 1).expand(3, D, H, W)
            result['spacing_channels'] = spacing_channels

        if self.add_analytical_solution:
            analytical = self._compute_analytical_solution(
                sigma=data['sigma'],
                mask=data['mask'],
                source_point=data['source_point'],
                spacing=data['spacing'],
                grid_shape=grid_shape,
            )
            result['analytical'] = analytical

        if self.add_distance_field:
            distance_field = self._compute_distance_to_boundary(
                mask=data['mask'],
                spacing=data['spacing'],
            )
            result['distance_field'] = distance_field

        return result


def get_dataloaders(
    config: dict,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders from config.

    Args:
        config: Configuration dictionary
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_config = config['data']

    # Spacing encoding options
    spacing_config = config.get('spacing', {})
    add_spacing_channels = spacing_config.get('add_spacing_channels', False)

    # Analytical solution option
    add_analytical_solution = config.get('model', {}).get('add_analytical_solution', False)

    # Distance field option
    add_distance_field = config.get('model', {}).get('add_distance_field', False)

    # Normalization options
    normalize_target = data_config.get('normalize_target', False)
    singularity_percentile = data_config.get('singularity_percentile', 99.0)
    log_transform_target = data_config.get('log_transform_target', False)

    # Exclusion list (optional)
    exclusion_file = data_config.get('exclusion_file', None)

    # Check if using combined directories
    data_dirs = data_config.get('data_dirs', None)

    if data_dirs is not None and len(data_dirs) > 0:
        # Use combined dataset from multiple directories
        splits = create_combined_data_splits(
            data_dirs=data_dirs,
            train_split=data_config['train_split'],
            val_split=data_config['val_split'],
            test_split=data_config['test_split'],
            seed=seed,
            exclusion_file=exclusion_file,
        )

        train_dataset = CombinedPotentialFieldDataset(
            samples=splits['train'],
            add_spacing_channels=add_spacing_channels,
            add_analytical_solution=add_analytical_solution,
            add_distance_field=add_distance_field,
            normalize_target=normalize_target,
            singularity_percentile=singularity_percentile,
            log_transform_target=log_transform_target,
        )

        val_dataset = CombinedPotentialFieldDataset(
            samples=splits['val'],
            add_spacing_channels=add_spacing_channels,
            add_analytical_solution=add_analytical_solution,
            add_distance_field=add_distance_field,
            normalize_target=normalize_target,
            singularity_percentile=singularity_percentile,
            log_transform_target=log_transform_target,
        )

        test_dataset = CombinedPotentialFieldDataset(
            samples=splits['test'],
            add_spacing_channels=add_spacing_channels,
            add_analytical_solution=add_analytical_solution,
            add_distance_field=add_distance_field,
            normalize_target=normalize_target,
            singularity_percentile=singularity_percentile,
            log_transform_target=log_transform_target,
        )
    else:
        # Use single directory (original behavior)
        train_indices, val_indices, test_indices = create_data_splits(
            data_dir=data_config['data_dir'],
            train_split=data_config['train_split'],
            val_split=data_config['val_split'],
            test_split=data_config['test_split'],
            seed=seed,
            exclusion_file=exclusion_file,
        )

        train_dataset = PotentialFieldDataset(
            data_dir=data_config['data_dir'],
            sample_indices=train_indices,
            add_spacing_channels=add_spacing_channels,
            add_analytical_solution=add_analytical_solution,
            add_distance_field=add_distance_field,
            normalize_target=normalize_target,
            singularity_percentile=singularity_percentile,
            log_transform_target=log_transform_target,
        )

        val_dataset = PotentialFieldDataset(
            data_dir=data_config['data_dir'],
            sample_indices=val_indices,
            add_spacing_channels=add_spacing_channels,
            add_analytical_solution=add_analytical_solution,
            add_distance_field=add_distance_field,
            normalize_target=normalize_target,
            singularity_percentile=singularity_percentile,
            log_transform_target=log_transform_target,
        )

        test_dataset = PotentialFieldDataset(
            data_dir=data_config['data_dir'],
            sample_indices=test_indices,
            add_spacing_channels=add_spacing_channels,
            add_analytical_solution=add_analytical_solution,
            add_distance_field=add_distance_field,
            normalize_target=normalize_target,
            singularity_percentile=singularity_percentile,
            log_transform_target=log_transform_target,
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        drop_last=True if len(train_dataset) > data_config['batch_size'] else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
    )

    return train_loader, val_loader, test_loader
