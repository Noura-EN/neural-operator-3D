## Goal
Improve FNO performance on the dataset in `data/` by reducing:
1) prediction noise (lack of smoothness vs GT)
2) global scale error (right shape but wrong magnitude)
3) performance error (MSE error on validation/test data)
Investigate zero-shot super resolution capabilities of the neural operator.

I have added data so there are now 500 samples of 48x48x96 within `data` and 500 more samples at double that in a specific folder `voxel_96_96_192` within `data`. 

TODO:
0) Ask me for clarifications
1) Systematic ablations. Keep experiments comparable by starting with the current baseline and produce a clear results table to add to this file. Downsample the higher resolution data to 48x48x96 and use as training and testing data
   - try deeper models now that there is more data
   - add tv regularisation
   - increase fourier modes
   - add an analytical solution using the muscle conductivity as $sigma_avg$ in a wider region the singularity instead of the small singularity mask
   - try combinations of the best options of the above
2) Once you have found the best model, train it on 48x48x96 samples (the data that is already at this resolution and part of the downsampled higher res data) and test its one-shot super-resolution capabilities on the remaining non-downsampled 96x96x192 higher resolution data. There should be NO interpolation.

## Current Best / Key Findings

### With 901 samples (combined dataset)
**NEW BEST: layers6_analytical** (Rel L2 = 0.118, Grad ratio = 0.87, Laplacian = 0.91)
- Combined 501 original + 400 downsampled high-res samples
- layers=6 + analytical monopole solution input: **Rel L2 = 0.118** (21% better than baseline!)
- layers=6 alone: Rel L2 = 0.127 (15% better than baseline)
- Analytical solution as input (layers=4): Rel L2 = 0.139 (7% better)
- **Depth sweet spot is layers=6**: layers_7 (0.182) and layers_8 (0.156) are worse

### With 352 samples (v1_medium)
**PREVIOUS BEST: layers_6** (Rel L2 = 0.157, Grad ratio = 0.97)
- layers=6: Rel L2 = 0.157 (27% better than baseline)
- modes=10: Rel L2 = 0.181 (17% better than baseline)
- Combinations hurt: modes10+layers6 = 0.225 (worse than either alone)

### With 100 samples (previous dataset)
- Best was additive_tv: Rel L2 = 0.185, Grad ratio = 0.98
- More capacity didn't help (data-limited regime)

## Hard Constraints (do not deviate unless explicitly justified)
- Dataset: 500 samples in `data/` with 500 more samples you will need to downsample.
- Train protocol per experiment:
  - max_epochs=100
  - early stopping patience=10, delta=1e-6
  - keep train/val/test split fixed across experiments (same random seed)
  - 70/15/15 split
- Each experiment must write to its own directory under `experiments/` with a descriptive name

## Metrics (must be reported for every experiment)
Report on:
- Entire domain (no mask)
- Muscle region only (evaluation mask), regardless of whether mask is used in training loss

At minimum report:
- MSE (and/or RMSE) on entire domain + muscle-only
- Relative scale error: compare predicted vs GT norm (L2 norm ratio)
- Smoothness/noise proxy:
  - gradient energy ratio: mean ||grad pred|| / mean ||grad gt||
  - Laplacian energy ratio: mean ||laplacian pred|| / mean ||laplacian gt||

Also log:
- best val metric, test metric at best checkpoint
- grad_weight, any mask usage, any loss weights, and spacing encoding choice

## Diagnosis Targets (what "noise" and "scale off" mean)
- Noise problem: pred has higher high-frequency content than GT (gradient/Laplacian ratios >> 1)
- Scale problem: L2 norm ratio far from 1 even when correlation/shape is reasonable

---

## Overall Results Table so far

### 901-Sample Combined Dataset Results

| Experiment | Changes vs Baseline | Rel L2 | L2 Norm Ratio | Grad Energy Ratio | Laplacian Ratio | Notes |
|------------|---------------------|--------|---------------|-------------------|-----------------|-------|
| baseline_901 | None (modes=8, width=32, layers=4) | 0.149 | 0.95 | 0.928 | 1.53 | Reference with combined data |
| layers_6_901 | Increased layers to 6 | 0.127 | 1.00 | 0.889 | 1.05 | 15% better, perfect scale |
| layers_7_901 | Increased layers to 7 | 0.182 | 0.93 | 0.949 | 1.37 | Worse than baseline |
| layers_8_901 | Increased layers to 8 | 0.156 | 1.01 | 0.920 | 1.15 | Worse than layers_6 |
| modes_10_901 | Increased Fourier modes to 10 | 0.147 | 0.93 | 0.923 | 1.25 | Marginal improvement |
| analytical_solution | Analytical monopole input (layers=4) | 0.139 | 0.99 | 0.922 | 1.15 | 7% better than baseline |
| **layers6_analytical** | **layers=6 + analytical input** | **0.118** | **0.97** | **0.871** | **0.91** | **BEST: 21% improvement!** |

### Key Findings (901 samples)

1. **layers6_analytical is the new best**:
   - Test Rel L2 = 0.118 (21% better than baseline, 8% better than layers_6 alone)
   - L2 Norm Ratio = 0.97 (near-perfect scale)
   - Laplacian Ratio = 0.91 (smoothest predictions yet!)

2. **Depth peaks at layers=6**:
   - layers_6: 0.127 (best single-factor improvement)
   - layers_7: 0.182 (worse than baseline!)
   - layers_8: 0.156 (worse than layers_6)

3. **Analytical solution as input helps**:
   - Monopole solution Φ = I/(4πσ_avg·r) provides useful prior
   - With layers=4: 7% improvement (0.139 vs 0.149)
   - Combined with layers=6: another 8% improvement

4. **modes=10 provides marginal benefit**:
   - Only 1% improvement over baseline (0.147 vs 0.149)
   - Not worth the extra parameters

### Comparison: Dataset Size Scaling

| Setting | 100 samples | 352 samples | 901 samples |
|---------|-------------|-------------|-------------|
| Baseline (layers=4) | 0.199 | 0.217 | 0.149 |
| layers=6 | Worse | 0.157 | 0.127 |
| Best overall | 0.185 (additive_tv) | 0.157 (layers_6) | **0.118** (layers6_analytical) |

**Conclusion**: With 9x more data (vs 100 samples), the FNO can now leverage both deeper architecture (layers=6) and physics-informed inputs (analytical solution) for significant accuracy gains.

---

### Zero-Shot Super-Resolution Results (96x96x192)

Testing trained models on 2x resolution (96x96x192) without retraining:

| Model | Training Rel L2 | Super-Res Rel L2 | Degradation | L2 Norm Ratio | Grad Ratio |
|-------|-----------------|------------------|-------------|---------------|------------|
| baseline_901 | 0.149 | 0.520 | 3.5x | 0.61 | 1.20 |
| **layers6_analytical** | **0.118** | **0.327** | **2.8x** | **0.87** | **0.87** |

**Key Findings**:
1. **FNO shows limited zero-shot super-resolution**: Both models degrade significantly at 2x resolution
2. **layers6_analytical generalizes better**: 2.8x degradation vs 3.5x for baseline
3. **Deeper + analytical input helps resolution transfer**: The combination provides more robust features
4. **Scale issues emerge at high-res**: L2 norm ratio drops (0.87 for best model), indicating under-prediction

**Why limited super-resolution?**
- Training data was exclusively 48x48x96 resolution
- Fixed Fourier modes (8x8x8) may limit frequency representation at higher resolution
- Consider training on mixed resolutions or increasing modes for better transfer

---

### 352-Sample Dataset Results

| Experiment | Changes vs Baseline | Rel L2 | L2 Norm Ratio | Grad Energy Ratio | Laplacian Ratio | Params | Notes |
|------------|---------------------|--------|---------------|-------------------|-----------------|--------|-------|
| baseline_352samples | None (modes=8, width=32, layers=4) | 0.217 | 1.00 | 0.96 | 1.40 | 8.5M | Reference |
| modes_10 | Increased Fourier modes to 10 | 0.181 | 0.93 | 0.94 | 1.34 | 16.5M | 17% better, slight underscale |
| modes_12 | Increased Fourier modes to 12 | 0.263 | 0.99 | 1.04 | 1.82 | 28.5M | Overfitting |
| width_48 | Increased width to 48 | 0.192 | 0.89 | 0.92 | 1.29 | 19.0M | 11% better |
| width_64 | Increased width to 64 | 0.190 | 0.95 | 0.88 | 1.26 | 33.7M | 12% better, smoothest |
| layers_5 | Increased layers to 5 | 0.229 | 0.91 | 0.89 | 1.47 | 10.6M | Worse than baseline |
| **layers_6** | **Increased layers to 6** | **0.157** | **0.97** | **0.97** | **1.23** | **12.7M** | **BEST: 27% improvement!** |
| modes10_layers6 | modes=10 + layers=6 | 0.225 | 0.92 | 0.90 | 1.30 | 24.7M | Combination hurts |
| width48_layers6 | width=48 + layers=6 | 0.236 | 0.97 | 1.04 | 1.76 | 28.5M | Combination hurts |

### Key Findings (352 samples)

1. **Depth is the key bottleneck**: With more data, deeper models (layers=6) now work and give the best results. Previously with 100 samples, layers=6 was worse.

2. **layers=6 is the new optimal**:
   - Test Rel L2 = 0.157 (27% better than baseline!)
   - L2 Norm Ratio = 0.97 (near-perfect scale)
   - Gradient Energy Ratio = 0.97 (almost exactly matches GT smoothness)
   - Only 12.7M parameters (efficient)

3. **modes=10 helps but less than layers=6**:
   - Test Rel L2 = 0.181 (17% better than baseline)
   - But 16.5M parameters vs 12.7M for layers=6

4. **Combinations don't help**:
   - modes10 + layers6: 0.225 (worse than either alone)
   - width48 + layers6: 0.236 (worse than either alone)
   - Too many parameters leads to overfitting

5. **Width improvements are moderate**:
   - width=48: Rel L2 = 0.192 (11% improvement)
   - width=64: Rel L2 = 0.190 (12% improvement, smoothest)
   - But layers=6 beats both with fewer parameters

### Comparison: 100 vs 352 Samples

| Setting | 100 samples | 352 samples | Change |
|---------|-------------|-------------|--------|
| Baseline (modes=8, width=32, layers=4) | 0.199 | 0.217 | Different test set |
| layers=6 | Worse | **0.157** | Now works! |
| modes=10 | 0.232 (worse) | 0.181 (better) | Now helps |
| width=64 | 0.241 (worse) | 0.190 (better) | Now helps |

**Conclusion**: With 3.5x more data, we've exited the data-limited regime. Deeper/larger models that previously overfit now generalize better.

---

## Previous Results (100-Sample Dataset)

### Summary Metrics (Test Set)

| Experiment | Changes vs Baseline | Rel L2 | L2 Norm Ratio | Grad Energy Ratio | Laplacian Ratio | Notes |
|------------|---------------------|--------|---------------|-------------------|-----------------|-------|
| baseline_new_metrics | None (prior baseline) | 0.199 | 0.996 | 1.34 | 2.81 | Near-perfect scale, but noisy |
| no_singularity_mask | Removed singularity exclusion | 0.240 | 0.966 | 1.37 | 2.89 | 20% worse - singularity mask helps |
| with_muscle_mask | Added muscle mask to loss | 0.967 | 0.130 | 0.40 | 0.42 | **FAILED**: Predictions ~8x too small |
| normalized_loss | MSE/(gt^2+eps) pointwise | 0.985 | 0.082 | 0.24 | 0.31 | FAILED: Predictions ~12x too small |
| logcosh_loss | log(cosh(pred-target)) | 0.380 | 0.726 | 1.03 | 2.26 | Low noise but 27% too small |
| spectral_threshold_0.01 | Penalize high-freq FFT | 0.482 | 0.890 | 2.06 | 4.70 | Made noise WORSE |
| pde_residual_0.1 | True PDE loss weight=0.1 | 0.300 | 0.909 | 1.48 | 3.41 | No improvement |
| mse_logcosh_0.1 | MSE + 0.1*log-cosh hybrid | 0.235 | 0.926 | 1.29 | 2.64 | Slight noise reduction, 7% scale off |
| tv_0.01 | MSE + 0.01*TV regularizer | 0.264 | 0.985 | 1.15 | 1.84 | Good noise reduction, near-perfect scale |
| mse_logcosh_tv | MSE + 0.1*logcosh + 0.01*TV | 0.240 | 0.950 | 1.05 | 1.34 | Good smoothness, 5% scale off |
| fno_modes_6 | Reduced Fourier modes to 6 | 0.263 | 1.070 | 1.49 | 2.84 | Worse accuracy AND noise |
| fno_modes_4 | Reduced Fourier modes to 4 | 0.380 | 0.964 | 1.50 | 3.01 | Much worse - underfitting |
| fno_modes_2 | Reduced Fourier modes to 2 | 0.677 | 0.994 | 1.43 | 2.90 | Severe underfitting |
| no_spacing_conditioning | Removed spacing MLP modulation | 0.223 | 0.988 | 1.41 | 2.93 | 12% worse - spacing conditioning helps |
| physical_coords | Coords scaled by spacing | 0.219 | 0.995 | 1.43 | 2.93 | 10% worse - normalized coords better |
| spacing_channels | Spacing as explicit channels | 0.274 | 0.984 | 1.37 | 2.78 | 38% worse - don't use |
| physical_coords_no_cond | Physical coords, no conditioning | 0.238 | 0.984 | 1.46 | 2.81 | 20% worse - baseline approach is best |
| **spacing_additive** | Additive spacing conditioning | 0.175 | 0.976 | 1.28 | 2.26 | **12% better than mult!** |
| spacing_film | FiLM spacing conditioning | 0.299 | 0.940 | 1.70 | 4.64 | 50% worse than baseline |
| spacing_gate | Gated spacing conditioning | 0.257 | 0.917 | 1.44 | 2.92 | 29% worse than baseline |
| spacing_combined | Combined (mult+add) conditioning | 0.232 | 1.030 | 1.56 | 3.43 | Worse than additive alone |
| fno_modes_10 | Increased Fourier modes to 10 | 0.232 | 0.936 | 1.50 | 3.10 | More capacity doesn't help |
| fno_modes_12 | Increased Fourier modes to 12 | 0.210 | 1.007 | 1.37 | 3.13 | Slight improvement, still overfits |
| fno_modes_16 | Increased Fourier modes to 16 | 0.260 | 0.922 | 1.18 | 2.87 | Best smoothness for modes alone |
| wider_decoder | fc_dim=256 (vs 128) | 0.343 | 1.091 | 1.69 | 3.81 | More decoder capacity doesn't help |
| wider_model | width=64 (vs 32) | 0.241 | 1.031 | 1.64 | 3.20 | More FNO capacity doesn't help |
| **additive_tv** | Additive spacing + TV loss | **0.185** | 0.964 | **0.98** | **1.40** | **BEST for 100 samples** |

---
