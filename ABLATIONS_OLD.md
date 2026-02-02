# Comprehensive Ablation Study Results

Auto-generated overnight experiments. Baseline: Muscle Rel L2 = 0.1359

## Results Table

| Experiment | Muscle Rel L2 | Muscle Far | L2 Norm | Grad Ratio | Laplacian | Epoch | vs Baseline |
|------------|---------------|------------|---------|------------|-----------|-------|-------------|
| baseline_layers6 | 0.1359 | 0.1340 | 0.9857 | 1.0210 | 2.39 | 46 | - |

## Phase 1: Loss Function Ablations

| grad_weight_0.5 | 0.1912 | 0.1894 | 0.9318 | 0.9639 | 3.48 | 27 | ✗ +40.7% |
| grad_weight_1.0 | 0.1518 | 0.1580 | 0.9799 | 1.0046 | 3.06 | 32 | ✗ +11.7% |
| grad_weight_2.0 | 0.2214 | 0.1975 | 1.0975 | 1.2483 | 4.10 | 13 | ✗ +62.9% |
| grad_weight_3.0 | 0.3811 | 0.4397 | 1.1754 | 1.0810 | 3.43 | 29 | ✗ +180.4% |
| grad_weight_5.0 | 0.2391 | 0.1786 | 1.0790 | 1.2356 | 3.51 | 13 | ✗ +76.0% |
| pde_weight_0.01 | 0.2068 | 0.1896 | 1.0778 | 1.2097 | 4.02 | 35 | ✗ +52.2% |
| pde_weight_0.05 | 0.3666 | 0.3697 | 1.2605 | 1.3859 | 4.99 | 13 | ✗ +169.8% |
| pde_weight_0.1 | 0.2351 | 0.2582 | 1.0097 | 0.9792 | 2.76 | 22 | ✗ +73.0% |
| pde_weight_0.2 | 0.4756 | 0.5601 | 1.3023 | 1.2599 | 3.97 | 10 | ✗ +249.9% |
| pde_weight_0.5 | 0.3004 | 0.2743 | 1.0677 | 1.2728 | 4.07 | 23 | ✗ +121.1% |
| tv_weight_0.0 | 0.3169 | 0.3083 | 1.2203 | 1.9817 | 15.72 | 13 | ✗ +133.2% |
| tv_weight_0.005 | 0.2016 | 0.1849 | 1.0815 | 1.2219 | 4.52 | 35 | ✗ +48.3% |
| tv_weight_0.02 | FAILED | - | - | - | - | - | TV regularization |
| tv_weight_0.05 | FAILED | - | - | - | - | - | TV regularization |
| tv_weight_0.1 | 0.4391 | 0.4335 | 0.7795 | 0.5957 | 0.98 | 36 | ✗ +223.1% |
| laplacian_weight_0.1 | 0.1992 | 0.1996 | 0.9295 | 1.0507 | 3.46 | 27 | ✗ +46.6% |
| laplacian_weight_0.5 | 0.3407 | 0.3602 | 0.9897 | 1.2079 | 4.02 | 19 | ✗ +150.7% |
| laplacian_weight_1.0 | FAILED | - | - | - | - | - | Laplacian matching |
| radius_15 | SKIPPED | - | - | - | - | - | Config missing |

## Phase 2: Architecture Ablations

| layers_4 | 0.2085 | 0.2102 | 0.9294 | 1.1256 | 3.69 | 25 | ✗ +53.4% |
| layers_5 | 0.4296 | 0.4797 | 1.2502 | 1.2957 | 4.92 | 11 | ✗ +216.1% |
| layers_7 | 0.2652 | 0.2978 | 0.9164 | 1.0736 | 2.61 | 27 | ✗ +95.2% |
| layers_8 | 0.2411 | 0.2423 | 0.9303 | 1.2170 | 4.52 | 10 | ✗ +77.4% |
| width_24 | 0.1601 | 0.1442 | 1.0004 | 1.1726 | 3.98 | 23 | ✗ +17.8% |
| width_48 | FAILED | - | - | - | - | - | FNO width |
| width_64 | FAILED | - | - | - | - | - | FNO width |
| modes_6 | 0.1789 | 0.1592 | 1.0612 | 1.1931 | 4.75 | 35 | ✗ +31.7% |
| modes_10 | 0.2391 | 0.2815 | 1.1162 | 1.1395 | 3.36 | 29 | ✗ +75.9% |
| modes_12 | FAILED | - | - | - | - | - | Fourier modes |
| modes_16 | 0.2358 | 0.1963 | 0.9945 | 1.2822 | 6.55 | 19 | ✗ +73.5% |
| fc_dim_64 | 0.1442 | 0.1396 | 0.9541 | 1.0465 | 2.48 | 33 | ✗ +6.1% |
| fc_dim_256 | 0.2067 | 0.2060 | 0.9242 | 1.0823 | 5.07 | 25 | ✗ +52.1% |
| fc_dim_512 | 0.3132 | 0.3695 | 1.1574 | 1.1405 | 4.00 | 23 | ✗ +130.5% |

## Phase 3: Training Ablations

| lr_0.0001 | FAILED | - | - | - | - | - | Learning rate |
| lr_0.0005 | 0.2030 | 0.1995 | 0.9916 | 1.1893 | 3.69 | 19 | ✗ +49.3% |
| lr_0.002 | 0.2925 | 0.3187 | 0.9307 | 1.1931 | 4.27 | 18 | ✗ +115.2% |
| lr_0.005 | 0.3693 | 0.3077 | 1.0680 | 1.9292 | 10.39 | 13 | ✗ +171.7% |
| weight_decay_0.0 | FAILED | - | - | - | - | - | Weight decay |
| weight_decay_0.00001 | FAILED | - | - | - | - | - | Weight decay |
| weight_decay_0.0001 | 0.3939 | 0.3941 | 0.9936 | 1.3093 | 5.12 | 17 | ✗ +189.8% |
| weight_decay_0.001 | 0.3885 | 0.4406 | 0.9289 | 1.1582 | 3.51 | 19 | ✗ +185.8% |
| epochs_200 | FAILED | - | - | - | - | - | Longer training |

## Phase 4: Input Ablations

| no_analytical | 0.1664 | 0.1683 | 0.9190 | 1.0558 | 3.59 | 33 | ✗ +22.4% |
| no_spacing_cond | 0.2379 | 0.2171 | 0.8991 | 1.1138 | 4.57 | 11 | ✗ +75.1% |
| geo_layers_1 | 0.2912 | 0.3317 | 1.1466 | 1.0813 | 3.21 | 25 | ✗ +114.2% |
| geo_layers_3 | 0.1975 | 0.2147 | 0.9206 | 1.1001 | 3.37 | 21 | ✗ +45.3% |
| geo_layers_4 | 0.2878 | 0.3325 | 1.1535 | 1.2188 | 5.21 | 22 | ✗ +111.8% |
| geo_dim_32 | 0.1998 | 0.1988 | 0.9996 | 1.0314 | 3.33 | 19 | ✗ +47.0% |
| geo_dim_128 | 0.1518 | 0.1428 | 0.9921 | 1.1092 | 3.95 | 18 | ✗ +11.7% |

## Phase 5: Alternative Architectures

| tfno_baseline | FAILED | - | - | - | - | - | TFNO backbone |
| tfno_layers8 | FAILED | - | - | - | - | - | TFNO 8 layers |
| lsm_baseline | FAILED | - | - | - | - | - | LSM backbone |

## Phase 6: Best Combinations

| best_combo | SKIPPED | - | - | - | - | - | Config missing |

## Phase 7: Fine-tuning


## Final Summary

Completed at Fri 30 Jan 09:33:46 GMT 2026

### Top 10 Experiments (by Muscle Rel L2)

