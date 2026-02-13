# Results Summary

This document summarizes all experimental results for the 3D neural operator potential field prediction project.

## Problem

Predict 3D electrical potential fields from conductivity tensors and source fields for muscle tissue simulations. The governing PDE is:

```
-div(sigma * grad(phi)) = f
```

with Neumann zero-flux boundary conditions. The domain is a 3D voxel grid containing layered cylindrical tissue geometry (bone, muscle, fat, skin).

## Dataset

- **Low-resolution**: 901 samples at 48x48x96 (501 original + 400 downsampled from high-res)
- **High-resolution**: 500 samples at 96x96x192
- **Mixed-resolution training**: 901 low-res + 50 original high-res samples
- **Train/Val/Test split**: 70/15/15, seed=42

## Architecture Comparison (10 Random Seeds)

All models use 6 FNO layers, 8 Fourier modes, width 32, analytical monopole solution input, additive spacing conditioning, and TV regularization (weight=0.01).

### Base-Resolution Test Set (48x48x96)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy |
|-------|----------|--------|---------------|-------------|
| U-Net | Base-res | 0.530 +/- 0.106 | 1.016 +/- 0.137 | 1.138 +/- 0.095 |
| **FNO** | **Base-res** | **0.152 +/- 0.020** | **0.980 +/- 0.024** | **0.913 +/- 0.041** |
| FNO + GeomAttn | Base-res | 0.171 +/- 0.023 | 0.994 +/- 0.028 | 0.939 +/- 0.061 |
| TFNO | Base-res | 0.166 +/- 0.017 | 0.978 +/- 0.041 | 1.01 +/- 0.068 |
| **FNO** | **Mixed-res** | **0.144 +/- 0.016** | **0.985 +/- 0.038** | **0.895 +/- 0.039** |
| FNO + GeomAttn | Mixed-res | 0.165 +/- 0.019 | 0.972 +/- 0.033 | 0.938 +/- 0.068 |
| TFNO | Mixed-res | 0.170 +/- 0.021 | 0.971 +/- 0.032 | 1.05 +/- 0.121 |

### High-Resolution Test Set (96x96x192)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy |
|-------|----------|--------|---------------|-------------|
| U-Net | Base-res | 1.25 +/- 0.388 | 0.746 +/- 0.504 | 2.964 +/- 3.34 |
| FNO | Base-res | 0.495 +/- 0.159 | 0.752 +/- 0.171 | 0.885 +/- 0.075 |
| FNO + GeomAttn | Base-res | 0.504 +/- 0.147 | 0.869 +/- 0.230 | 0.938 +/- 0.205 |
| TFNO | Base-res | 0.447 +/- 0.165 | 1.14 +/- 0.196 | 1.23 +/- 0.142 |
| **FNO** | **Mixed-res** | **0.128 +/- 0.023** | **0.996 +/- 0.061** | **0.970 +/- 0.035** |
| FNO + GeomAttn | Mixed-res | 0.155 +/- 0.031 | 0.965 +/- 0.050 | 1.05 +/- 0.071 |
| TFNO | Mixed-res | 0.162 +/- 0.040 | 0.923 +/- 0.055 | 1.15 +/- 0.184 |

### Key Findings

1. **FNO is the best architecture** for both base-res (0.152) and super-resolution (0.128 with mixed training)
2. **Mixed-resolution training enables true super-resolution**: FNO achieves 0.128 at 2x resolution, better than its 0.144 at base resolution
3. **U-Net fails at super-resolution** (Rel L2 = 1.25) due to its fixed-resolution architecture
4. **FNO + GeomAttn and TFNO** perform similarly but don't improve over vanilla FNO
5. All FNO variants show near-perfect scale (L2 Norm Ratio ~ 1.0) and smoothness (Grad Energy ~ 1.0)

## MUAP Evaluation Results

Motor Unit Action Potential (MUAP) evaluation on the best model (FNO, mixed-resolution training, 10-seed ensemble):

| Test Set | Correlation | Rel L2 | Rel MSE | P2P Ratio |
|----------|-------------|--------|---------|-----------|
| Low-res (48x48x96) | 0.9927 +/- 0.0186 | 0.2034 +/- 0.1623 | 0.0677 +/- 0.1373 | 0.9743 +/- 0.2514 |
| High-res (96x96x192) | 0.9372 +/- 0.0328 | 0.3655 +/- 0.0869 | 0.1412 +/- 0.0630 | 0.8265 +/- 0.2125 |

### Key Findings

1. **Excellent correlation** at both resolutions (>0.93), indicating correct field pattern learning
2. **Near-perfect amplitude** at low-res (P2P ratio = 0.97), slight underestimation at high-res (0.83)
3. **Super-resolution degrades gracefully**: ~1.8x increase in Rel L2 from low to high resolution

## Super-Resolution Baselines

Comparison of super-resolution approaches on the high-res test set:

| Method | High-Res Rel L2 | vs Mixed Training |
|--------|-----------------|-------------------|
| **FNO mixed-res training** | **0.128 +/- 0.023** | -- |
| Trilinear interpolation (theoretical) | 0.272 | 2.1x worse |
| FNO standard + upsample | 0.383 +/- 0.073 | 3.0x worse |
| FNO standard (direct high-res) | 0.495 +/- 0.159 | 3.9x worse |

Mixed-resolution training achieves 3x better super-resolution than the standard + upsample approach, and even beats the theoretical trilinear interpolation limit (0.128 vs 0.272).

### Why Mixed-Resolution Works

- The **spacing MLP was the bottleneck**: it had never seen high-res spacing values during standard training
- Including just 50 high-res samples (5% of data) exposes the spacing MLP to both regimes
- FNO Fourier operations are inherently resolution-independent once spacing is properly handled
- The model truly learns resolution-invariant inference (not just better features), proven by the fact that forcing mixed-res models through downsample+upsample degrades performance back to standard levels

## Ensemble Results

Ensemble of 10 random seeds for FNO with mixed-resolution training:

| Test Set | Ensemble Rel L2 | Mean Individual | Improvement |
|----------|-----------------|-----------------|-------------|
| Low-res (48x48x96) | 0.1134 | 0.1441 +/- 0.0155 | 21.3% |
| High-res (96x96x192) | 0.0810 | 0.1280 +/- 0.0226 | 36.7% |

Ensembling provides substantial improvements, especially for super-resolution (36.7% error reduction).

## Model Complexity and Inference Time

| Model | Parameters | Inference Time (ms) |
|-------|------------|---------------------|
| Trilinear Interp. | 0 | <1 |
| FNO | 12.72M | 10.0 +/- 0.2 |
| FNO + GeomAttn | 12.74M | 15.0 +/- 0.2 |
| TFNO | 12.72M | 11.4 +/- 0.2 |
| UNet | 23.72M | 14.7 +/- 0.1 |

Inference time measured on NVIDIA GPU, 48x48x96 input, averaged over 50 samples.

## Historical Ablation Results

### Architecture Ablations (901 samples, single seed)

These experiments established the optimal FNO configuration before the multi-seed study.

| Experiment | Rel L2 | Notes |
|------------|--------|-------|
| Baseline (layers=4, modes=8) | 0.149 | Reference |
| layers=6 | 0.127 | 15% improvement |
| layers=7 | 0.182 | Worse (too deep) |
| layers=8 | 0.156 | Worse than layers=6 |
| modes=10 | 0.147 | Marginal improvement |
| Analytical solution input | 0.139 | 7% improvement |
| **layers=6 + analytical** | **0.118** | **21% improvement (best)** |

### Key Conclusions from Ablations

1. **Depth peaks at 6 layers**: More layers hurt (7, 8 are worse)
2. **Analytical monopole solution** as input provides consistent ~7% improvement
3. **Additive spacing conditioning** is essential for resolution awareness
4. **TV regularization** (weight=0.01) improves smoothness with minimal accuracy cost
5. **Singularity mask** (3-voxel radius) prevents fitting numerical artifacts near point source
6. **Combinations don't help**: modes10+layers6, width48+layers6 are worse than layers=6 alone

### Loss Function Experiments

All non-standard losses performed worse than the baseline MSE + TV combination:
- Gradient matching loss: Worse at base res, but best for zero-shot super-res (without mixed training)
- PDE residual loss: No improvement at any weight
- Percentile-based masking: Hurts performance everywhere
- Per-sample normalization: Makes learning harder, degrades all metrics

### Alternative Architecture Experiments

Early comparison (single seed, 501 samples):

| Model | Rel L2 | Parameters |
|-------|--------|------------|
| **TFNO** | **0.186** | 12.7M |
| LSM | 0.189 | 5.1M |
| U-NO | 0.237 | 24.4M |
| FNO | 0.255 | 12.7M |
| DeepONet | 0.854 | 12.0M |

Note: These early results used fewer samples and a single seed. The multi-seed study with 901 samples (above) is the definitive comparison.

### Spacing and Super-Resolution Experiments

Neither log-transform nor normalized spacing helped extrapolation. The solution is mixed-resolution training, not spacing transforms.

| Model | Base Rel L2 | Super-Res Rel L2 |
|-------|-------------|------------------|
| Baseline (raw spacing) | 0.118 | 0.329 |
| Log spacing | 0.155 | 0.557 |
| Normalized spacing | 0.170 | 0.506 |
| No spacing + analytical | 0.226 | 0.394 |
| **Mixed-res training** | **0.157** | **0.110** |

## Metrics Definitions

- **Rel L2**: Relative L2 error = ||pred - target||_2 / ||target||_2 (lower is better)
- **L2 Norm Ratio**: ||pred||_2 / ||target||_2 (1.0 = perfect scale)
- **Grad Energy Ratio**: mean(||grad pred||) / mean(||grad target||) (1.0 = matching smoothness)
- **Correlation**: Pearson correlation between predicted and ground truth MUAP signals
- **P2P Ratio**: Peak-to-peak amplitude ratio of predicted vs ground truth MUAPs

## Reproducing Results

### Training (single seed)
```bash
# FNO, standard training
python scripts/main.py --config configs/fno_standard.yaml --experiment-name fno_standard_seed42 --seed 42

# FNO, mixed-resolution training
python scripts/main.py --config configs/fno_mixed.yaml --experiment-name fno_mixed_seed42 --seed 42
```

### Multi-seed study
```bash
python scripts/run_ablations.py --max-parallel 2 --seeds 42 142 242 342 442 542 642 742 842 942
```

### MUAP evaluation
```bash
python scripts/evaluate_muap_accuracy.py
```
