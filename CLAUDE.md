## Goal
Improve FNO performance on the dataset in `data/` by reducing:
1) prediction noise (lack of smoothness vs GT)
2) global scale error (right shape but wrong magnitude)
3) performance error (MSE error on validation/test data)

Do this via systematic ablations. Keep experiments comparable and produce a clear results table.

## Current Best / Key Findings

### With 352 samples (v1_medium dataset)
**NEW BEST: layers_6** (Rel L2 = 0.157, Grad ratio = 0.97)
- With 3.5x more data, deeper models now work!
- layers=6: Rel L2 = 0.157 (27% better than baseline!)
- modes=10: Rel L2 = 0.181 (17% better than baseline)
- Wider models (width 48, 64): ~10% improvement
- **Combinations hurt**: modes10+layers6 = 0.225 (worse than either alone)
- The sweet spot is layers=6 with baseline width=32 and modes=8

### With 100 samples (previous dataset)
- Best was additive_tv: Rel L2 = 0.185, Grad ratio = 0.98
- More capacity didn't help (data-limited regime)

## Hard Constraints (do not deviate unless explicitly justified)
- Dataset: 352 samples in `data/` (v1_medium)
- Train protocol per experiment:
  - max_epochs=100
  - early stopping patience=10, delta=1e-6
  - keep train/val/test split fixed across experiments (same random seed)
  - 70/15/15 split = 246 train, 52 val, 54 test
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

## Overall Results Table

### 352-Sample Dataset Results (v1_medium)

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

## Recommendations

### For 352-sample dataset (current):
1. **Use layers=6 as new default**:
   - Rel L2 = 0.157 (best accuracy)
   - Near-perfect scale and smoothness
   - Efficient (12.7M params)

2. **Update config**:
   ```yaml
   model:
     fno:
       modes1: 8  # Keep at 8
       modes2: 8
       modes3: 8
       width: 32  # Keep at 32
       num_layers: 6  # Increase from 4 to 6
   ```

3. **Don't combine improvements**: Each optimal setting works best alone. Combining modes=10 + layers=6 is worse than layers=6 alone.

### Next steps to explore:
- Test layers=7 or layers=8 (may continue improving)
- Try layers=6 with TV regularization
- Data augmentation if more accuracy needed
