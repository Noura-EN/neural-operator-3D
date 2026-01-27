#!/usr/bin/env python3
"""Visualize potential field along 1D fiber cross-sections.

This script generates plots showing the predicted vs ground truth potential
along Z-axis fiber cross-sections at multiple positions in the muscle volume.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import PotentialFieldDataset
from src.models.wrapper import build_model, get_device


def find_source_location(source: torch.Tensor) -> tuple:
    """Find the voxel location of the current source.

    Args:
        source: Source field tensor (1, D, H, W)

    Returns:
        Tuple of (d, h, w) indices of the source location
    """
    # Source is typically a point source - find max absolute value
    source_np = source.squeeze().cpu().numpy()
    idx = np.unravel_index(np.argmax(np.abs(source_np)), source_np.shape)
    return idx


def generate_fiber_positions(
    shape: tuple,
    source_location: tuple,
    num_fibers: int = 10,
) -> list:
    """Generate fiber positions in the HW plane.

    Args:
        shape: Volume shape (D, H, W)
        source_location: (d, h, w) of source
        num_fibers: Number of fibers to generate

    Returns:
        List of (h, w) positions for fibers
    """
    D, H, W = shape
    _, src_h, src_w = source_location

    positions = []

    # First fiber always through source
    positions.append((src_h, src_w))

    # Generate remaining fibers in a grid pattern
    remaining = num_fibers - 1

    # Create grid excluding borders (10% margin)
    h_margin = int(H * 0.1)
    w_margin = int(W * 0.1)

    h_positions = np.linspace(h_margin, H - h_margin - 1, int(np.ceil(np.sqrt(remaining))) + 1, dtype=int)
    w_positions = np.linspace(w_margin, W - w_margin - 1, int(np.ceil(np.sqrt(remaining))) + 1, dtype=int)

    # Generate grid positions
    for h in h_positions:
        for w in w_positions:
            if len(positions) >= num_fibers:
                break
            # Skip if too close to source
            if abs(h - src_h) < 3 and abs(w - src_w) < 3:
                continue
            positions.append((int(h), int(w)))
        if len(positions) >= num_fibers:
            break

    # If we still need more, add random positions
    while len(positions) < num_fibers:
        h = np.random.randint(h_margin, H - h_margin)
        w = np.random.randint(w_margin, W - w_margin)
        if (h, w) not in positions:
            positions.append((h, w))

    return positions[:num_fibers]


def extract_fiber_potential(
    volume: torch.Tensor,
    h: int,
    w: int,
    mask: torch.Tensor = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract potential values along a Z-axis fiber.

    Args:
        volume: Potential field (1, D, H, W)
        h: Height index
        w: Width index
        mask: Optional mask (1, D, H, W) - values outside mask shown as NaN

    Returns:
        Tuple of (potential values, mask values) along Z
    """
    values = volume[0, :, h, w].detach().cpu().numpy()

    if mask is not None:
        mask_vals = mask[0, :, h, w].detach().cpu().numpy()
        # Set values outside mask to NaN (creates gaps in plot)
        values = np.where(mask_vals > 0.5, values, np.nan)
        return values, mask_vals

    return values, np.ones_like(values)


def plot_fiber_comparison(
    pred: np.ndarray,
    target: np.ndarray,
    z_coords: np.ndarray,
    fiber_idx: int,
    h: int,
    w: int,
    is_source_fiber: bool,
    ax: plt.Axes,
    mask_vals: np.ndarray = None,
):
    """Plot predicted vs ground truth potential along a fiber.

    Args:
        pred: Predicted potential along fiber (NaN where outside mask)
        target: Ground truth potential along fiber (NaN where outside mask)
        z_coords: Z-axis coordinates (physical or normalized)
        fiber_idx: Fiber index for labeling
        h, w: Fiber position
        is_source_fiber: Whether this fiber passes through the source
        ax: Matplotlib axis
        mask_vals: Optional mask values along fiber
    """
    label_suffix = " (through source)" if is_source_fiber else ""

    # Plot with gaps where NaN (outside muscle)
    ax.plot(z_coords, target, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(z_coords, pred, 'r--', linewidth=2, label='Prediction', alpha=0.8)

    ax.set_xlabel('Z position (normalized)', fontsize=10)
    ax.set_ylabel('Potential (V)', fontsize=10)
    ax.set_title(f'Fiber {fiber_idx}: (h={h}, w={w}){label_suffix}', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Compute error only on valid (non-NaN) points
    valid = ~np.isnan(pred) & ~np.isnan(target)
    if valid.sum() > 0:
        pred_valid = pred[valid]
        target_valid = target[valid]
        rmse = np.sqrt(np.mean((pred_valid - target_valid) ** 2))
        rel_error = rmse / (np.std(target_valid) + 1e-8)

        # Calculate muscle coverage
        coverage = 100 * valid.sum() / len(valid)

        ax.text(0.02, 0.98, f'RMSE: {rmse:.4f}\nRel: {rel_error:.2%}\nMuscle: {coverage:.0f}%',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.02, 0.98, 'No valid points\n(outside muscle)',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.5))


def create_fiber_visualization(
    pred: torch.Tensor,
    target: torch.Tensor,
    source: torch.Tensor,
    sample_idx: int,
    resolution: str,
    output_dir: str,
    num_fibers: int = 10,
    mask: torch.Tensor = None,
):
    """Create fiber potential visualization for a sample.

    Args:
        pred: Predicted potential (1, 1, D, H, W)
        target: Ground truth potential (1, 1, D, H, W)
        source: Source field (1, 1, D, H, W)
        sample_idx: Sample index
        resolution: Resolution label (e.g., "base_48x48x96")
        output_dir: Output directory
        num_fibers: Number of fibers to visualize
        mask: Optional muscle mask (1, 1, D, H, W) - shows gaps outside muscle
    """
    # Remove batch dimension
    pred = pred.squeeze(0)  # (1, D, H, W)
    target = target.squeeze(0)
    source = source.squeeze(0)
    if mask is not None:
        mask = mask.squeeze(0)  # (1, D, H, W)

    D, H, W = pred.shape[1], pred.shape[2], pred.shape[3]

    # Find source location
    source_loc = find_source_location(source)

    # Generate fiber positions
    fiber_positions = generate_fiber_positions((D, H, W), source_loc, num_fibers)

    # Create Z coordinates (normalized)
    z_coords = np.linspace(-1, 1, D)

    # Create figure with subplots
    n_cols = 2
    n_rows = (num_fibers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if num_fibers > 1 else [axes]

    # Plot each fiber
    for i, (h, w) in enumerate(fiber_positions):
        pred_fiber, mask_vals = extract_fiber_potential(pred, h, w, mask)
        target_fiber, _ = extract_fiber_potential(target, h, w, mask)

        is_source = (h == fiber_positions[0][0] and w == fiber_positions[0][1])

        plot_fiber_comparison(
            pred_fiber, target_fiber, z_coords,
            fiber_idx=i + 1, h=h, w=w,
            is_source_fiber=is_source,
            ax=axes[i],
            mask_vals=mask_vals,
        )

    # Hide unused subplots
    for i in range(num_fibers, len(axes)):
        axes[i].set_visible(False)

    # Overall title
    fig.suptitle(f'Fiber Potential Profiles - Sample {sample_idx} ({resolution})',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'fiber_potential_sample_{sample_idx}_{resolution}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def run_evaluation(
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    num_fibers: int = 10,
    max_samples: int = None,
    include_highres: bool = True,
):
    """Run evaluation and generate fiber visualizations.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        output_dir: Output directory for visualizations
        num_fibers: Number of fibers per sample
        max_samples: Maximum number of samples to visualize (None for all)
        include_highres: Whether to also evaluate on high-res samples
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Build model
    print("Building model...")
    model = build_model(config)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Check if model uses analytical solution
    add_analytical = config.get('model', {}).get('add_analytical_solution', False)

    # Create output directories
    base_res_dir = os.path.join(output_dir, 'base_resolution')
    high_res_dir = os.path.join(output_dir, 'high_resolution')
    os.makedirs(base_res_dir, exist_ok=True)

    # ===== Base Resolution Evaluation =====
    print("\n" + "="*60)
    print("Evaluating on BASE RESOLUTION (48x48x96)")
    print("="*60)

    # Create test split indices manually
    data_dir = Path(config['data']['data_dir'])
    all_files = sorted(data_dir.glob("sample_*.npz"))
    n_total = len(all_files)

    train_split = config['data']['train_split']
    val_split = config['data']['val_split']

    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    # Test indices are after train and val
    test_indices = list(range(n_train + n_val, n_total))

    # Load base resolution dataset with test indices
    base_dataset = PotentialFieldDataset(
        data_dir=config['data']['data_dir'],
        sample_indices=test_indices,
        add_analytical_solution=add_analytical,
    )

    print(f"Test samples: {len(base_dataset)}")

    n_samples = min(len(base_dataset), max_samples) if max_samples else len(base_dataset)

    with torch.no_grad():
        for idx in tqdm(range(n_samples), desc="Base resolution"):
            batch = base_dataset[idx]

            # Prepare inputs
            sigma = batch['sigma'].unsqueeze(0).to(device)
            source = batch['source'].unsqueeze(0).to(device)
            coords = batch['coords'].unsqueeze(0).to(device)
            spacing = batch['spacing'].unsqueeze(0).to(device)
            target = batch['target'].unsqueeze(0).to(device)
            mask = batch['mask'].unsqueeze(0).to(device)

            analytical = None
            if add_analytical and 'analytical' in batch:
                analytical = batch['analytical'].unsqueeze(0).to(device)

            # Forward pass
            pred = model(sigma, source, coords, spacing, analytical=analytical)

            # Create visualization
            D, H, W = target.shape[2], target.shape[3], target.shape[4]
            resolution_str = f"base_{D}x{H}x{W}"

            save_path = create_fiber_visualization(
                pred, target, source,
                sample_idx=idx,
                resolution=resolution_str,
                output_dir=base_res_dir,
                num_fibers=num_fibers,
                mask=mask,
            )

            if idx < 3:  # Print first few
                print(f"  Saved: {save_path}")

    print(f"\nBase resolution visualizations saved to: {base_res_dir}")

    # ===== High Resolution Evaluation =====
    if include_highres:
        print("\n" + "="*60)
        print("Evaluating on HIGH RESOLUTION (96x96x192)")
        print("="*60)

        # Clear GPU memory before high-res
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        os.makedirs(high_res_dir, exist_ok=True)

        # Load high-res holdout info
        holdout_info_path = "data/downsampled_highres/preprocessing_info.json"
        highres_data_dir = "data/voxel_96_96_192"

        if os.path.exists(holdout_info_path) and os.path.exists(highres_data_dir):
            with open(holdout_info_path, 'r') as f:
                holdout_info = json.load(f)

            holdout_indices = holdout_info['holdout_indices']

            # Load high-res samples directly
            all_files = sorted(Path(highres_data_dir).glob("sample_*.npz"))
            highres_files = [all_files[i] for i in holdout_indices if i < len(all_files)]

            print(f"High-res holdout samples: {len(highres_files)}")

            n_highres = min(len(highres_files), max_samples) if max_samples else len(highres_files)

            oom_count = 0
            for idx, sample_file in enumerate(tqdm(highres_files[:n_highres], desc="High resolution")):
                try:
                    # Clear GPU memory before each sample
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Load sample manually
                    data = np.load(sample_file)

                    # High-res data has different format: (D, H, W, 6) vs (6, D, H, W)
                    sigma_raw = data['sigma']
                    if sigma_raw.shape[-1] == 6:  # (D, H, W, 6) format
                        sigma = torch.from_numpy(sigma_raw).float().permute(3, 0, 1, 2)
                    else:
                        sigma = torch.from_numpy(sigma_raw).float()

                    source_raw = data['source']
                    source = torch.from_numpy(source_raw).float()
                    if source.dim() == 3:
                        source = source.unsqueeze(0)  # Add channel dim

                    # Key is 'u' not 'potential' in high-res data
                    target_raw = data['u']
                    target = torch.from_numpy(target_raw).float()
                    if target.dim() == 3:
                        target = target.unsqueeze(0)  # Add channel dim

                    # Load mask
                    mask_raw = data['mask']
                    mask = torch.from_numpy(mask_raw.astype(np.float32))
                    if mask.dim() == 3:
                        mask = mask.unsqueeze(0)  # Add channel dim

                    spacing = torch.from_numpy(data['spacing']).float()

                    # Generate coordinates
                    D, H, W = target.shape[1], target.shape[2], target.shape[3]
                    z = torch.linspace(-1, 1, D)
                    y = torch.linspace(-1, 1, H)
                    x = torch.linspace(-1, 1, W)
                    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
                    coords = torch.stack([X, Y, Z], dim=0)

                    # Compute analytical if needed
                    analytical = None
                    if add_analytical:
                        # Find source position
                        source_np = source.squeeze().numpy()
                        src_idx = np.unravel_index(np.argmax(np.abs(source_np)), source_np.shape)
                        src_pos = np.array(src_idx) * spacing.numpy()

                        # Compute analytical solution (monopole)
                        z_phys = torch.arange(D).float() * spacing[0]
                        y_phys = torch.arange(H).float() * spacing[1]
                        x_phys = torch.arange(W).float() * spacing[2]
                        Z_p, Y_p, X_p = torch.meshgrid(z_phys, y_phys, x_phys, indexing='ij')

                        r = torch.sqrt(
                            (X_p - src_pos[2])**2 +
                            (Y_p - src_pos[1])**2 +
                            (Z_p - src_pos[0])**2
                        )

                        sigma_avg = sigma.mean()
                        I = 1.0
                        eps = 0.1 * spacing.min()

                        analytical = (I / (4 * np.pi * sigma_avg)) / (r + eps)
                        analytical = analytical.unsqueeze(0)

                    # Prepare for model
                    sigma = sigma.unsqueeze(0).to(device)
                    source = source.unsqueeze(0).to(device)
                    coords = coords.unsqueeze(0).to(device)
                    spacing = spacing.unsqueeze(0).to(device)
                    target = target.unsqueeze(0).to(device)
                    mask = mask.unsqueeze(0).to(device)
                    if analytical is not None:
                        analytical = analytical.unsqueeze(0).to(device)

                    # Forward pass
                    pred = model(sigma, source, coords, spacing, analytical=analytical)

                    # Create visualization
                    resolution_str = f"highres_{D}x{H}x{W}"

                    save_path = create_fiber_visualization(
                        pred, target, source,
                        sample_idx=idx,
                        resolution=resolution_str,
                        output_dir=high_res_dir,
                        num_fibers=num_fibers,
                        mask=mask,
                    )

                    if idx < 3:
                        print(f"  Saved: {save_path}")

                    # Explicit cleanup after each sample
                    del pred, sigma, source, coords, spacing, target, mask
                    if analytical is not None:
                        del analytical

                except torch.cuda.OutOfMemoryError:
                    oom_count += 1
                    print(f"\n  OOM on sample {idx}, skipping... (total OOM: {oom_count})")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    print(f"\n  Error on sample {idx}: {e}")
                    continue

            if oom_count > 0:
                print(f"\nNote: {oom_count} samples skipped due to OOM")

            print(f"\nHigh resolution visualizations saved to: {high_res_dir}")
        else:
            print("High-res holdout data not found. Skipping high-res evaluation.")

    print("\n" + "="*60)
    print("DONE")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Visualize fiber potential profiles")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/layers6_analytical_fno_20260123_144632/checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations/fiber_potentials",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--num-fibers",
        type=int,
        default=10,
        help="Number of fiber cross-sections per sample",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to visualize (default: all)",
    )
    parser.add_argument(
        "--no-highres",
        action="store_true",
        help="Skip high-resolution evaluation",
    )
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        num_fibers=args.num_fibers,
        max_samples=args.max_samples,
        include_highres=not args.no_highres,
    )


if __name__ == "__main__":
    main()
