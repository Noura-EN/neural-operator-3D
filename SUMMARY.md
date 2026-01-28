# Neural Operator 3D - Codebase Summary

This document provides an overview of the codebase architecture, data processing pipeline, model design, loss functions, and experimental results for the 3D potential field prediction project using Fourier Neural Operators (FNO).

## Table of Contents
1. [Problem Overview](#problem-overview)
2. [Data Processing Pipeline](#data-processing-pipeline)
3. [Model Architecture](#model-architecture)
4. [Loss Functions](#loss-functions)
5. [Configurable Parameters](#configurable-parameters)
6. [Experimental Results](#experimental-results)
7. [Regional Performance Analysis](#regional-performance-analysis)

---

## Problem Overview

The goal is to predict 3D potential fields (φ) from conductivity tensors (σ) and source fields (f) for electrical stimulation simulations in muscle tissue. The governing PDE is:

```
-∇·(σ∇φ) = f
```

**Key challenges:**
- Point source creates a singularity with values approaching infinity near the source
- Target values span many orders of magnitude (~10⁻⁵ to ~10¹)
- Model must generalize across different muscle geometries and source locations
- Far-from-source regions are harder to predict accurately due to smaller signal magnitudes

---

## Data Processing Pipeline

### Data Format (`src/data/loader.py`)

Each sample is stored as an NPZ file containing:

| Field | Shape | Description |
|-------|-------|-------------|
| `sigma` | (D, H, W, 6) | Conductivity tensor (symmetric 3x3, stored as 6 components) |
| `source` | (D, H, W) | Source field f (point source delta function) |
| `mask` | (D, H, W) | Binary muscle region mask |
| `u` | (D, H, W) | FEM ground truth potential |
| `spacing` | (3,) | Physical voxel spacing [dz, dy, dx] in mm |
| `source_point` | (3,) | Source location in voxel coordinates |

### Preprocessing Steps

1. **Tensor Conversion**: Convert from (D, H, W, C) to (C, D, H, W) format for PyTorch
2. **Coordinate Generation**: Create normalized coordinate grids [-1, 1] for spatial encoding:
   ```python
   coords = torch.stack([X, Y, Z], dim=0)  # Shape: (3, D, H, W)
   ```
3. **Analytical Solution** (optional): Compute monopole approximation Φ = 1/(4πσ_avg·r):
   - Uses average diagonal conductivity from muscle region
   - Provides physics-informed prior for the model
4. **Normalization** (optional): Per-sample target normalization excluding singularity region

### Data Splits
- **Train/Val/Test**: 70%/15%/15% (seed=42 for reproducibility)
- **352 samples** in current dataset (v1_medium)
- Typical grid size: ~48×23×23 voxels

---

## Model Architecture

### Overview (`src/models/wrapper.py`)

```
PotentialFieldModel
├── CombinedEncoder
│   ├── GeometryEncoder (processes conductivity tensor)
│   ├── Feature Fusion (concatenate + 1x1 conv)
│   └── SpacingConditioner (additive spacing bias)
└── FNOBackbone (Fourier Neural Operator)
```

### CombinedEncoder (`src/models/geometry.py`)

**GeometryEncoder:**
- 3D convolutional layers: σ → hidden features
- Default: 6 → 64 → 64 channels with BatchNorm + GELU
- Processes the spatially-varying conductivity tensor

**Feature Fusion:**
- Concatenates: geometry features (64) + source (1-2) + coords (3)
- 1x1 convolution to output dimension
- Total input: 68-69 channels → 64 output channels

**SpacingConditioner (Additive):**
- Learns spacing-dependent bias: `features + MLP(spacing)`
- MLP: 3 → 32 → 64 (with GELU activation)
- Critical for resolution-independent learning

### FNO Backbone (`src/models/fno.py`)

**Architecture:**
```
Input (B, 64, D, H, W)
    │
    ▼
Lifting Layer (1x1 Conv: 64 → width)
    │
    ▼
FNO Blocks × num_layers
    │   ├── SpectralConv3d (Fourier space multiplication)
    │   ├── Linear bypass (1x1 Conv)
    │   └── Residual: x = GELU(spectral + linear)
    │
    ▼
Projection (width → fc_dim → 1)
    │
    ▼
Output (B, 1, D, H, W)
```

**SpectralConv3d:**
- Performs convolution via FFT: `IFFT(FFT(x) × W)`
- Uses rFFT for memory efficiency
- Learnable complex weights for each Fourier mode quadrant
- Truncates to specified number of modes (frequency components)

**Default Parameters:**
- modes1/2/3 = 8 (Fourier modes per dimension)
- width = 32 (hidden channel width)
- num_layers = 6 (optimal, see ablations)
- fc_dim = 128 (decoder hidden dimension)

---

## Loss Functions

### CombinedLoss (`src/utils/masking.py`)

**Primary Components:**

| Loss | Default Weight | Description |
|------|----------------|-------------|
| MSE | 1.0 | Mean squared error with singularity exclusion |
| Gradient | 0.5 | Gradient consistency: MSE(∇pred, ∇target) |
| TV | 0.01 | Total Variation: encourages smoothness |
| PDE | 0.0 | Physics residual: -∇·(σ∇φ) - f ≈ 0 |
| Gradient Matching | 0.0 | Direct gradient MSE |

### Singularity Handling

**Radius-based masking** (default, optimal):
- Excludes spherical region of radius=3 voxels around source
- Prevents fitting numerical artifacts near point source

**Percentile-based masking** (experimental):
- Excludes voxels where |target| > X percentile threshold
- Alternative but found to be worse than radius-based

### Muscle Region Mask

Identifies muscle tissue by conductivity values:
- σ_xx = 0.2455, σ_yy = 0.2455, σ_zz = 1.2275 (anisotropic)
- Loss computed only within muscle region

---

## Configurable Parameters

### Best Configuration (from ablation studies)

```yaml
model:
  backbone: fno
  add_analytical_solution: true  # Critical: provides physics prior
  geometry_encoder:
    hidden_dim: 64
    num_layers: 2
  fno:
    modes1: 8
    modes2: 8
    modes3: 8
    width: 32
    num_layers: 6  # KEY: increased from 4

training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.00001
  scheduler: cosine
  early_stopping_patience: 10

loss:
  pde_weight: 0.0
  tv_weight: 0.01  # Small TV regularization helps smoothness
  gradient_matching_weight: 0.0

physics:
  mse:
    singularity_mask_radius: 3
```

### Key Parameter Guidelines

| Parameter | Range Tested | Optimal | Notes |
|-----------|-------------|---------|-------|
| FNO layers | 4-8 | **6** | More depth helps until ~6-7 |
| FNO modes | 2-16 | **8** | Higher modes overfit, lower underfit |
| FNO width | 32-64 | **32** | Wider doesn't help much |
| TV weight | 0-0.1 | **0.01** | Improves smoothness |
| Analytical solution | on/off | **on** | Significant improvement |
| Singularity radius | 0-5 | **3** | Smaller radius hurts accuracy |

---

## Experimental Results

### 352-Sample Dataset Results (v1_medium)

| Experiment | Changes vs Baseline | Rel L2 | L2 Norm Ratio | Grad Energy Ratio | Params |
|------------|---------------------|--------|---------------|-------------------|--------|
| baseline | modes=8, width=32, layers=4 | 0.217 | 1.00 | 0.96 | 8.5M |
| modes_10 | Increased modes to 10 | 0.181 | 0.93 | 0.94 | 16.5M |
| modes_12 | Increased modes to 12 | 0.263 | 0.99 | 1.04 | 28.5M |
| width_48 | Increased width to 48 | 0.192 | 0.89 | 0.92 | 19.0M |
| width_64 | Increased width to 64 | 0.190 | 0.95 | 0.88 | 33.7M |
| layers_5 | Increased layers to 5 | 0.229 | 0.91 | 0.89 | 10.6M |
| **layers_6** | **Increased layers to 6** | **0.157** | **0.97** | **0.97** | **12.7M** |
| modes10_layers6 | modes=10 + layers=6 | 0.225 | 0.92 | 0.90 | 24.7M |

### Key Findings

1. **Depth is the primary bottleneck**: layers=6 gives 27% improvement over baseline
2. **Combinations don't help**: modes10+layers6 is worse than layers=6 alone
3. **Scale accuracy**: L2 Norm Ratio near 1.0 indicates correct global magnitude
4. **Smoothness**: Gradient Energy Ratio near 1.0 means predictions match GT smoothness

### Metrics Explained

- **Rel L2**: Relative L2 error = ||pred - target||₂ / ||target||₂ (lower is better)
- **L2 Norm Ratio**: ||pred||₂ / ||target||₂ (1.0 = perfect scale)
- **Grad Energy Ratio**: mean(||∇pred||) / mean(||∇target||) (1.0 = matching smoothness)
- **Laplacian Ratio**: Similar for second derivatives (noise indicator)

---

## Regional Performance Analysis

### Region Definitions

- **Near region** (top 10%): Voxels where |target| > 90th percentile
  - High signal values, close to source
  - Model predicts well here
- **Far region** (bottom 90%): Voxels where |target| < 90th percentile
  - Low signal values, far from source
  - More challenging, lower relative accuracy

### Why Baseline is Optimal Everywhere

All attempts to improve far-region performance made things **worse**:

| Approach | Far-90 Rel L2 | Change vs Baseline |
|----------|---------------|-------------------|
| **Baseline (3-voxel mask)** | **0.1746** | - |
| Mask top 1% | 0.2352 | +35% worse |
| Mask top 10% | 0.3046 | +74% worse |
| Distance weight α=1 | 0.2453 | +41% worse |
| Distance weight α=5 | 0.4415 | +153% worse |
| PDE loss 0.1 | 0.2609 | +49% worse |

### Explanation

The strong near-source signals provide crucial learning gradients:

1. **Gradient magnitude**: Near-source regions have larger |∇L/∇pred| contributing more to parameter updates
2. **Field structure**: Learning the high-gradient near-source structure helps the model understand the global potential field shape
3. **Information flow**: Near-source accuracy cascades to far regions through field continuity
4. **Removing/downweighting strong signals**: Reduces useful gradient information, degrading learning everywhere

**Bottom line**: The model benefits from learning the full field structure. Artificially suppressing near-source signals hurts both near and far performance.

---

## File Structure

```
neural-operator-3D/
├── configs/
│   └── ablations/          # Experiment configurations
├── data/                   # Training data (sample_*.npz)
├── experiments/            # Experiment outputs
├── scripts/
│   ├── main.py             # Training script
│   ├── run_ablation.py     # Run experiments from configs
│   ├── eval_*.py           # Evaluation scripts
│   └── visualize_*.py      # Visualization scripts
└── src/
    ├── data/
    │   └── loader.py       # Data loading and preprocessing
    ├── models/
    │   ├── wrapper.py      # Main model class
    │   ├── geometry.py     # CombinedEncoder, SpacingConditioner
    │   ├── fno.py          # FNO backbone
    │   └── ...             # Other backbones (UNet, U-NO, etc.)
    └── utils/
        ├── masking.py      # Loss functions, masks
        ├── metrics.py      # Evaluation metrics
        └── visualization.py # Plotting utilities
```

---

## Running Experiments

### Training
```bash
python scripts/main.py --config configs/ablations/layers_6.yaml
```

### Evaluation
```bash
python scripts/eval_single.py experiments/layers6_analytical_fno_*/checkpoints/best_model.pt
```

### Ablation Studies
```bash
python scripts/run_ablation.py configs/ablations/layers_6.yaml
```

---

## Summary

The optimal configuration for this dataset is:
- **FNO with 6 layers** (key finding)
- **Analytical solution as input** (provides physics prior)
- **TV regularization** (improves smoothness)
- **3-voxel singularity mask** (excludes numerical artifacts)
- **Additive spacing conditioning** (resolution independence)

This achieves Test Rel L2 = 0.157 with near-perfect scale (0.97) and smoothness (0.97).
