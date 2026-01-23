"""Preprocess high-resolution data for training.

This script:
1. Downsamples 96x96x192 data to 48x48x96 using 2x2x2 average pooling
2. Saves downsampled data to a new directory
3. Records which samples are held out for super-resolution testing
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm


def downsample_3d_avg(arr: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downsample 3D array using average pooling.

    Args:
        arr: Input array of shape (..., D, H, W) or (D, H, W, C)
        factor: Downsampling factor (default 2)

    Returns:
        Downsampled array
    """
    if arr.ndim == 3:
        # (D, H, W) -> (D//factor, H//factor, W//factor)
        D, H, W = arr.shape
        new_shape = (D // factor, factor, H // factor, factor, W // factor, factor)
        return arr.reshape(new_shape).mean(axis=(1, 3, 5))
    elif arr.ndim == 4:
        # (D, H, W, C) -> (D//factor, H//factor, W//factor, C)
        D, H, W, C = arr.shape
        new_shape = (D // factor, factor, H // factor, factor, W // factor, factor, C)
        return arr.reshape(new_shape).mean(axis=(1, 3, 5))
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}")


def downsample_source_point(source_point: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downsample source point coordinates.

    Args:
        source_point: Source point in (z, y, x) or (x, y, z) format
        factor: Downsampling factor

    Returns:
        Downsampled source point
    """
    return source_point / factor


def process_sample(input_path: Path, output_path: Path, factor: int = 2):
    """Downsample a single sample and save.

    Args:
        input_path: Path to input npz file
        output_path: Path to output npz file
        factor: Downsampling factor
    """
    data = np.load(input_path)

    # Downsample fields
    sigma_ds = downsample_3d_avg(data['sigma'], factor)
    source_ds = downsample_3d_avg(data['source'], factor)
    mask_ds = (downsample_3d_avg(data['mask'].astype(float), factor) > 0.5).astype(data['mask'].dtype)
    u_ds = downsample_3d_avg(data['u'], factor)
    source_fem_ds = downsample_3d_avg(data['source_fem'], factor) if 'source_fem' in data else source_ds

    # Update spacing (doubles when downsampling by 2)
    spacing_ds = data['spacing'] * factor

    # Update source point
    source_point_ds = downsample_source_point(data['source_point'], factor)

    # Update grid shape
    grid_shape_ds = np.array([s // factor for s in data['grid_shape']])

    # Save downsampled data
    np.savez_compressed(
        output_path,
        sigma=sigma_ds,
        source=source_ds,
        mask=mask_ds,
        u=u_ds,
        source_fem=source_fem_ds,
        grid_shape=grid_shape_ds,
        spacing=spacing_ds,
        r_skin=data['r_skin'],
        length=data['length'],
        source_point=source_point_ds,
        meta_path=str(data['meta_path']),
        fem_u_path=str(data['fem_u_path']),
        fem_summary=str(data['fem_summary']),
    )


def main():
    parser = argparse.ArgumentParser(description="Downsample high-resolution data")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/voxel_96_96_192",
        help="Directory containing high-res samples",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/downsampled_highres",
        help="Directory to save downsampled samples",
    )
    parser.add_argument(
        "--n-holdout",
        type=int,
        default=100,
        help="Number of samples to hold out for super-resolution testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible holdout selection",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=2,
        help="Downsampling factor",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all high-res samples
    all_files = sorted(input_dir.glob("sample_*.npz"))
    n_total = len(all_files)
    print(f"Found {n_total} high-resolution samples in {input_dir}")

    # Randomly select holdout samples
    rng = np.random.default_rng(args.seed)
    all_indices = np.arange(n_total)
    rng.shuffle(all_indices)

    holdout_indices = sorted(all_indices[:args.n_holdout].tolist())
    downsample_indices = sorted(all_indices[args.n_holdout:].tolist())

    print(f"Holding out {len(holdout_indices)} samples for super-resolution testing")
    print(f"Downsampling {len(downsample_indices)} samples")

    # Save holdout information
    holdout_info = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_holdout": args.n_holdout,
        "n_downsampled": len(downsample_indices),
        "seed": args.seed,
        "factor": args.factor,
        "holdout_indices": holdout_indices,
        "holdout_files": [all_files[i].name for i in holdout_indices],
        "downsampled_indices": downsample_indices,
    }

    with open(output_dir / "preprocessing_info.json", "w") as f:
        json.dump(holdout_info, f, indent=2)

    # Downsample selected samples
    for i, idx in enumerate(tqdm(downsample_indices, desc="Downsampling")):
        input_path = all_files[idx]
        # Use sequential numbering for downsampled files
        output_path = output_dir / f"sample_{i:06d}.npz"
        process_sample(input_path, output_path, factor=args.factor)

    print(f"\nDownsampled {len(downsample_indices)} samples to {output_dir}")
    print(f"Holdout info saved to {output_dir / 'preprocessing_info.json'}")

    # Verify a sample
    print("\nVerifying first downsampled sample...")
    orig = np.load(all_files[downsample_indices[0]])
    ds = np.load(output_dir / "sample_000000.npz")
    print(f"  Original sigma shape: {orig['sigma'].shape}")
    print(f"  Downsampled sigma shape: {ds['sigma'].shape}")
    print(f"  Original spacing: {orig['spacing']}")
    print(f"  Downsampled spacing: {ds['spacing']}")


if __name__ == "__main__":
    main()
