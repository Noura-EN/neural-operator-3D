# Ablation Study Results

## Summary

- **Models**: UNet, FNO (with/without analytical), FNO+GeomAttn, TFNO
- **Training Types**: Standard (901 samples), Mixed (901 + 50 high-res)
- **Seeds**: 10 seeds per configuration for uncertainty quantification
- **Test Sets**: Low-res (48x48x96) and High-res (96x96x192)


## Model Complexity and Inference Time

| Model | Parameters | Inference Time (ms) | Notes |
|-------|------------|---------------------|-------|
| Trilinear Interp. | 0 | <1 | Naive baseline |
| UNet | 23.72M | 14.7 ± 0.1 | Largest model |
| FNO (analytical) | 12.72M | 10.0 ± 0.2 | Best accuracy |
| FNO (no analytical) | 12.72M | 10.0 ± 0.1 | Same as FNO |
| FNO + GeomAttn | 12.74M | 15.0 ± 0.2 | +50% slower due to attention |
| TFNO | 12.72M | 11.4 ± 0.2 | +14% slower due to normalization |

*Inference time measured on NVIDIA GPU, 48×48×96 input, averaged over 50 samples.*


## Super-Resolution Baselines

### Baseline 1: Trilinear Interpolation (Theoretical Upper Bound)
Downsample high-res GT to low-res, then upsample with trilinear interpolation.
This represents the **best possible** result from interpolation alone.

| Metric | Value |
|--------|-------|
| High-Res Rel L2 | 0.2718 |
| L2 Norm Ratio | 1.0000 |
| Grad Energy Ratio | 0.6311 |

### Baseline 2: Standard Training + Trilinear Upsample (Practical Baseline)
Train FNO on low-res data, predict at low-res, upsample prediction to high-res.
This is the **practical alternative** to mixed-resolution training.

| Model | High-Res Rel L2 (10 seeds) |
|-------|----------------------------|
| FNO (no analytical) + upsample | 0.3833 ± 0.0725 |
| FNO (with analytical) + upsample | 0.4088 ± 0.0651 |

### Comparison

| Method | High-Res Rel L2 | vs Mixed Training |
|--------|-----------------|-------------------|
| FNO mixed-res training | **0.128 ± 0.023** | — |
| Trilinear (theoretical) | 0.272 | 2.1× worse |
| FNO standard + upsample | 0.383 ± 0.073 | 3.0× worse |
| FNO standard (direct) | 0.446 ± 0.084 | 3.5× worse |

**Key findings**:
1. Standard + upsample (0.38) is slightly better than direct high-res (0.45) — the model performs better on inputs from its training distribution
2. Mixed-resolution training (0.13) achieves **3× better** super-resolution than standard + upsample
3. Mixed training even beats the theoretical interpolation limit (0.13 vs 0.27)

### Baseline 3: Mixed Training + Downsample/Upsample
Tests whether mixed-res models learn better features or truly learn resolution-invariant inference.

| Model | Direct High-Res | Downsample + Upsample |
|-------|-----------------|----------------------|
| FNO (no analytical) mixed | 0.1291 ± 0.0305 | 0.4101 ± 0.0535 |
| FNO (with analytical) mixed | 0.1280 ± 0.0226 | 0.3835 ± 0.0917 |

**Critical insight**: Mixed-res models perform **3× worse** when forced through downsample+upsample (0.38-0.41 vs 0.13). This proves:
- The super-resolution ability **requires high-res inputs** — it's not just better learned features
- Mixed-res training teaches the model to **natively process high-res data**
- When denied high-res inputs, mixed models perform similarly to standard models (~0.40 vs ~0.38)


## Seed 42 Results

### Low-Resolution Test Set (48x48x96)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 0.7536 | 1.3443 | 1.2285 | N/A | 0.000004 |
| fno_analytical | standard | 0.1176 | 0.9670 | 0.8714 | N/A | 0.000000 |
| fno_analytical | mixed | 0.1362 | 0.9912 | 0.8822 | N/A | 0.000000 |
| fno_no_analytical | standard | 0.1268 | 1.0041 | 0.8893 | N/A | 0.000000 |
| fno_no_analytical | mixed | 0.1333 | 1.0109 | 0.9067 | N/A | 0.000000 |
| fno_geom_attn | standard | 0.1600 | 1.0019 | 0.9565 | N/A | 0.000001 |
| fno_geom_attn | mixed | 0.1948 | 0.9410 | 1.0877 | N/A | 0.000001 |
| tfno | standard | 0.1813 | 1.0009 | 0.9691 | N/A | 0.000000 |
| tfno | mixed | 0.1711 | 0.9425 | 0.9185 | N/A | 0.000000 |

### High-Resolution Test Set (96x96x192)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 1.0301 | 0.4029 | 0.8618 | N/A | 0.000014 |
| fno_analytical | standard | 0.3206 | 0.8482 | 0.8565 | N/A | 0.000000 |
| fno_analytical | mixed | 0.1166 | 0.9436 | 0.9536 | N/A | 0.000000 |
| fno_no_analytical | standard | 0.4569 | 0.7241 | 0.9729 | N/A | 0.000000 |
| fno_no_analytical | mixed | 0.1023 | 0.9950 | 1.0017 | N/A | 0.000000 |
| fno_geom_attn | standard | 0.4407 | 0.6938 | 0.7218 | N/A | 0.000001 |
| fno_geom_attn | mixed | 0.1866 | 0.9939 | 1.1627 | N/A | 0.000001 |
| tfno | standard | 0.4157 | 0.9930 | 1.0169 | N/A | 0.000001 |
| tfno | mixed | 0.2339 | 0.8068 | 0.9651 | N/A | 0.000000 |

## Seed 142 Results

### Low-Resolution Test Set (48x48x96)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 0.5052 | 1.0233 | 1.1134 | N/A | 0.000007 |
| fno_analytical | standard | 0.1399 | 0.9735 | 0.8886 | N/A | 0.000000 |
| fno_analytical | mixed | 0.1268 | 0.9933 | 0.8689 | N/A | 0.000000 |
| fno_no_analytical | standard | 0.2159 | 1.0247 | 1.0488 | N/A | 0.000002 |
| fno_no_analytical | mixed | 0.1440 | 0.9843 | 0.9124 | N/A | 0.000000 |
| fno_geom_attn | standard | 0.1660 | 1.0209 | 0.9766 | N/A | 0.000001 |
| fno_geom_attn | mixed | 0.1756 | 1.0062 | 1.0836 | N/A | 0.000001 |
| tfno | standard | 0.1517 | 0.9723 | 0.9263 | N/A | 0.000000 |
| tfno | mixed | 0.2074 | 1.0513 | 1.2094 | N/A | 0.000001 |

### High-Resolution Test Set (96x96x192)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 1.0668 | 0.4437 | 1.5617 | N/A | 0.000008 |
| fno_analytical | standard | 0.6576 | 0.8452 | 0.8315 | N/A | 0.000003 |
| fno_analytical | mixed | 0.1029 | 1.0196 | 0.9610 | N/A | 0.000000 |
| fno_no_analytical | standard | 0.4857 | 0.8592 | 1.2562 | N/A | 0.000004 |
| fno_no_analytical | mixed | 0.1423 | 0.9888 | 1.0969 | N/A | 0.000000 |
| fno_geom_attn | standard | 0.3378 | 0.9127 | 0.9124 | N/A | 0.000004 |
| fno_geom_attn | mixed | 0.1621 | 1.0390 | 1.1396 | N/A | 0.000002 |
| tfno | standard | 0.3249 | 0.9987 | 1.0661 | N/A | 0.000001 |
| tfno | mixed | 0.2081 | 0.9176 | 1.3705 | N/A | 0.000001 |

## Seed 242 Results

### Low-Resolution Test Set (48x48x96)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 0.5539 | 1.0443 | 1.2211 | N/A | 0.000004 |
| fno_analytical | standard | 0.1801 | 0.9915 | 0.9864 | N/A | 0.000000 |
| fno_analytical | mixed | 0.1351 | 1.0185 | 0.9017 | N/A | 0.000000 |
| fno_no_analytical | standard | 0.1377 | 0.9446 | 0.8703 | N/A | 0.000000 |
| fno_no_analytical | mixed | 0.1662 | 1.0096 | 0.9427 | N/A | 0.000000 |
| fno_geom_attn | standard | 0.1652 | 1.0124 | 0.9702 | N/A | 0.000001 |
| fno_geom_attn | mixed | 0.1667 | 0.9852 | 1.0445 | N/A | 0.000001 |
| tfno | standard | 0.1654 | 0.9973 | 1.0564 | N/A | 0.000001 |
| tfno | mixed | 0.1697 | 0.9596 | 1.0660 | N/A | 0.000000 |

### High-Resolution Test Set (96x96x192)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 1.1601 | 0.6813 | 2.9325 | N/A | 0.000012 |
| fno_analytical | standard | 0.5912 | 0.5125 | 0.9877 | N/A | 0.000001 |
| fno_analytical | mixed | 0.1119 | 1.0360 | 0.9697 | N/A | 0.000000 |
| fno_no_analytical | standard | 0.5290 | 0.5444 | 0.8771 | N/A | 0.000001 |
| fno_no_analytical | mixed | 0.1576 | 1.0811 | 1.0570 | N/A | 0.000000 |
| fno_geom_attn | standard | 0.4611 | 1.1567 | 0.8997 | N/A | 0.000001 |
| fno_geom_attn | mixed | 0.1809 | 0.9427 | 1.0702 | N/A | 0.000001 |
| tfno | standard | 0.3799 | 0.9955 | 1.2073 | N/A | 0.000001 |
| tfno | mixed | 0.1439 | 1.0374 | 1.1943 | N/A | 0.000000 |

## Seed 342 Results

### Low-Resolution Test Set (48x48x96)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 0.5647 | 1.0088 | 1.0451 | N/A | 0.000003 |
| fno_analytical | standard | 0.1801 | 0.9741 | 0.9016 | N/A | 0.000001 |
| fno_analytical | mixed | 0.1283 | 0.9772 | 0.8881 | N/A | 0.000000 |
| fno_no_analytical | standard | 0.1825 | 0.9418 | 0.9186 | N/A | 0.000001 |
| fno_no_analytical | mixed | 0.1442 | 1.0022 | 0.9245 | N/A | 0.000000 |
| fno_geom_attn | standard | 0.2479 | 0.8557 | 0.7962 | N/A | 0.000001 |
| fno_geom_attn | mixed | 0.2012 | 0.9151 | 0.9630 | N/A | 0.000001 |
| tfno | standard | 0.1752 | 1.0411 | 1.1412 | N/A | 0.000001 |
| tfno | mixed | 0.1686 | 0.9515 | 1.0812 | N/A | 0.000000 |

### High-Resolution Test Set (96x96x192)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 1.0916 | 0.4839 | 1.7564 | N/A | 0.000011 |
| fno_analytical | standard | 0.5779 | 0.5750 | 0.8785 | N/A | 0.000001 |
| fno_analytical | mixed | 0.1243 | 0.9473 | 0.9401 | N/A | 0.000000 |
| fno_no_analytical | standard | 0.3108 | 0.8844 | 1.0196 | N/A | 0.000001 |
| fno_no_analytical | mixed | 0.1250 | 0.9274 | 0.9834 | N/A | 0.000000 |
| fno_geom_attn | standard | 0.5640 | 0.5556 | 0.6548 | N/A | 0.000003 |
| fno_geom_attn | mixed | 0.1814 | 0.9055 | 1.0501 | N/A | 0.000001 |
| tfno | standard | 0.8887 | 1.6881 | 1.4999 | N/A | 0.000001 |
| tfno | mixed | 0.1706 | 0.9092 | 1.1858 | N/A | 0.000000 |

## Seed 442 Results

### Low-Resolution Test Set (48x48x96)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 0.5325 | 0.8149 | 1.1641 | N/A | 0.000007 |
| fno_analytical | standard | 0.1469 | 0.9268 | 0.8646 | N/A | 0.000000 |
| fno_analytical | mixed | 0.1305 | 1.0238 | 0.9524 | N/A | 0.000000 |
| fno_no_analytical | standard | 0.1519 | 0.9808 | 0.9357 | N/A | 0.000000 |
| fno_no_analytical | mixed | 0.1485 | 0.9159 | 0.8874 | N/A | 0.000000 |
| fno_geom_attn | standard | 0.1630 | 1.0116 | 1.0132 | N/A | 0.000001 |
| fno_geom_attn | mixed | 0.1971 | 0.9171 | 0.9853 | N/A | 0.000001 |
| tfno | standard | 0.1615 | 0.9798 | 1.0649 | N/A | 0.000001 |
| tfno | mixed | 0.1911 | 0.9342 | 1.3240 | N/A | 0.000001 |

### High-Resolution Test Set (96x96x192)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 1.2494 | 0.8070 | 1.8321 | N/A | 0.000040 |
| fno_analytical | standard | 0.7761 | 0.7341 | 0.9897 | N/A | 0.000001 |
| fno_analytical | mixed | 0.1056 | 1.0034 | 1.0204 | N/A | 0.000000 |
| fno_no_analytical | standard | 0.3827 | 0.8148 | 1.1383 | N/A | 0.000001 |
| fno_no_analytical | mixed | 0.1675 | 0.8890 | 1.0225 | N/A | 0.000000 |
| fno_geom_attn | standard | 0.7617 | 1.0025 | 1.2031 | N/A | 0.000002 |
| fno_geom_attn | mixed | 0.1961 | 0.8542 | 0.9587 | N/A | 0.000000 |
| tfno | standard | 0.4525 | 1.2273 | 1.2413 | N/A | 0.000001 |
| tfno | mixed | 0.2048 | 0.8827 | 1.5832 | N/A | 0.000001 |

## Seed 542 Results

### Low-Resolution Test Set (48x48x96)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 0.5634 | 1.1038 | 1.3312 | N/A | 0.000005 |
| fno_analytical | standard | 0.1585 | 0.9991 | 0.9592 | N/A | 0.000001 |
| fno_analytical | mixed | 0.1538 | 0.9917 | 0.9416 | N/A | 0.000001 |
| fno_no_analytical | standard | 0.1509 | 0.9149 | 0.8676 | N/A | 0.000000 |
| fno_no_analytical | mixed | 0.1739 | 0.9818 | 0.9330 | N/A | 0.000001 |
| fno_geom_attn | standard | 0.2525 | 0.9651 | 0.9701 | N/A | 0.000005 |
| fno_geom_attn | mixed | 0.1668 | 0.9328 | 0.8275 | N/A | 0.000001 |
| tfno | standard | 0.1472 | 0.9522 | 0.9366 | N/A | 0.000000 |
| tfno | mixed | 0.1458 | 0.9699 | 0.9700 | N/A | 0.000000 |

### High-Resolution Test Set (96x96x192)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 1.1687 | 0.7523 | 1.6261 | N/A | 0.000027 |
| fno_analytical | standard | 0.3322 | 0.8513 | 0.8860 | N/A | 0.000003 |
| fno_analytical | mixed | 0.1219 | 1.0434 | 1.0187 | N/A | 0.000000 |
| fno_no_analytical | standard | 0.4423 | 0.8475 | 1.0783 | N/A | 0.000002 |
| fno_no_analytical | mixed | 0.1849 | 1.0903 | 1.1683 | N/A | 0.000001 |
| fno_geom_attn | standard | 0.7700 | 0.6741 | 0.9679 | N/A | 0.000002 |
| fno_geom_attn | mixed | 0.1297 | 1.0044 | 0.9147 | N/A | 0.000001 |
| tfno | standard | 0.4478 | 1.1262 | 1.0882 | N/A | 0.000000 |
| tfno | mixed | 0.1363 | 0.9179 | 0.9909 | N/A | 0.000000 |

## Seed 642 Results

### Low-Resolution Test Set (48x48x96)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 0.5492 | 0.9475 | 1.0835 | N/A | 0.000011 |
| fno_analytical | standard | 0.1546 | 1.0216 | 0.9344 | N/A | 0.000000 |
| fno_analytical | mixed | 0.1652 | 0.9589 | 0.8963 | N/A | 0.000001 |
| fno_no_analytical | standard | 0.1748 | 0.9743 | 0.9698 | N/A | 0.000001 |
| fno_no_analytical | mixed | 0.1239 | 1.0056 | 0.9199 | N/A | 0.000000 |
| fno_geom_attn | standard | 0.2016 | 0.9348 | 0.7660 | N/A | 0.000001 |
| fno_geom_attn | mixed | 0.1451 | 0.9524 | 0.8866 | N/A | 0.000001 |
| tfno | standard | 0.1978 | 0.8855 | 0.9539 | N/A | 0.000000 |
| tfno | mixed | 0.1555 | 0.9931 | 1.0033 | N/A | 0.000000 |

### High-Resolution Test Set (96x96x192)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 1.0398 | 0.3644 | 1.1970 | N/A | 0.000014 |
| fno_analytical | standard | 0.6158 | 0.4881 | 0.7560 | N/A | 0.000001 |
| fno_analytical | mixed | 0.1429 | 0.9424 | 0.9777 | N/A | 0.000001 |
| fno_no_analytical | standard | 0.6303 | 0.4934 | 0.9551 | N/A | 0.000002 |
| fno_no_analytical | mixed | 0.1092 | 1.0064 | 1.0504 | N/A | 0.000000 |
| fno_geom_attn | standard | 0.5188 | 0.6945 | 0.7119 | N/A | 0.000006 |
| fno_geom_attn | mixed | 0.0941 | 0.9810 | 1.0077 | N/A | 0.000000 |
| tfno | standard | 0.4350 | 1.1764 | 1.2016 | N/A | 0.000000 |
| tfno | mixed | 0.1282 | 0.9342 | 1.0650 | N/A | 0.000000 |

## Seed 742 Results

### Low-Resolution Test Set (48x48x96)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 0.5481 | 1.0571 | 1.0809 | N/A | 0.000004 |
| fno_analytical | standard | 0.1504 | 0.9753 | 0.8789 | N/A | 0.000000 |
| fno_analytical | mixed | 0.1739 | 0.8988 | 0.8101 | N/A | 0.000000 |
| fno_no_analytical | standard | 0.1498 | 0.9556 | 0.9711 | N/A | 0.000000 |
| fno_no_analytical | mixed | 0.1308 | 0.9961 | 0.8890 | N/A | 0.000000 |
| fno_geom_attn | standard | 0.2049 | 1.0558 | 1.1662 | N/A | 0.000002 |
| fno_geom_attn | mixed | 0.1755 | 0.9588 | 0.9293 | N/A | 0.000000 |
| tfno | standard | 0.1405 | 0.9684 | 0.9716 | N/A | 0.000000 |
| tfno | mixed | 0.1542 | 0.9637 | 0.9611 | N/A | 0.000000 |

### High-Resolution Test Set (96x96x192)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 2.3964 | 2.1857 | 12.8013 | N/A | 0.000664 |
| fno_analytical | standard | 0.3894 | 0.7055 | 0.8320 | N/A | 0.000001 |
| fno_analytical | mixed | 0.1456 | 0.9210 | 0.9294 | N/A | 0.000000 |
| fno_no_analytical | standard | 0.3943 | 0.8317 | 1.0566 | N/A | 0.000000 |
| fno_no_analytical | mixed | 0.1071 | 1.0200 | 0.9799 | N/A | 0.000000 |
| fno_geom_attn | standard | 0.3394 | 1.1483 | 1.3041 | N/A | 0.000003 |
| fno_geom_attn | mixed | 0.1304 | 0.9763 | 1.0581 | N/A | 0.000000 |
| tfno | standard | 0.2893 | 1.0976 | 1.3355 | N/A | 0.000001 |
| tfno | mixed | 0.1140 | 0.9540 | 1.0576 | N/A | 0.000000 |

## Seed 842 Results

### Low-Resolution Test Set (48x48x96)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 0.3939 | 0.9238 | 1.1220 | N/A | 0.000004 |
| fno_analytical | standard | 0.1711 | 0.9963 | 0.9596 | N/A | 0.000001 |
| fno_analytical | mixed | 0.1526 | 1.0370 | 0.9343 | N/A | 0.000000 |
| fno_no_analytical | standard | 0.1206 | 0.9447 | 0.9192 | N/A | 0.000000 |
| fno_no_analytical | mixed | 0.1167 | 0.9856 | 0.8967 | N/A | 0.000000 |
| fno_geom_attn | standard | 0.1711 | 0.9817 | 0.9221 | N/A | 0.000001 |
| fno_geom_attn | mixed | 0.1689 | 0.9831 | 0.9346 | N/A | 0.000001 |
| tfno | standard | 0.1564 | 0.9573 | 1.0068 | N/A | 0.000000 |
| tfno | mixed | 0.1398 | 0.9871 | 0.9869 | N/A | 0.000000 |

### High-Resolution Test Set (96x96x192)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 1.1964 | 0.7789 | 2.9418 | N/A | 0.000053 |
| fno_analytical | standard | 0.3808 | 1.0001 | 0.9879 | N/A | 0.000002 |
| fno_analytical | mixed | 0.1827 | 1.1324 | 1.0132 | N/A | 0.000000 |
| fno_no_analytical | standard | 0.4230 | 0.8924 | 1.0139 | N/A | 0.000001 |
| fno_no_analytical | mixed | 0.0899 | 0.9883 | 0.9844 | N/A | 0.000000 |
| fno_geom_attn | standard | 0.4258 | 0.6610 | 0.8899 | N/A | 0.000000 |
| fno_geom_attn | mixed | 0.1430 | 0.9746 | 1.0677 | N/A | 0.000001 |
| tfno | standard | 0.5370 | 1.0748 | 1.2522 | N/A | 0.000000 |
| tfno | mixed | 0.1128 | 0.9564 | 1.0368 | N/A | 0.000000 |

## Seed 942 Results

### Low-Resolution Test Set (48x48x96)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 0.3317 | 0.8923 | 0.9902 | N/A | 0.000003 |
| fno_analytical | standard | 0.1252 | 0.9767 | 0.8860 | N/A | 0.000000 |
| fno_analytical | mixed | 0.1386 | 0.9619 | 0.8789 | N/A | 0.000000 |
| fno_no_analytical | standard | 0.1322 | 0.9654 | 0.8851 | N/A | 0.000000 |
| fno_no_analytical | mixed | 0.1341 | 0.9580 | 0.8760 | N/A | 0.000000 |
| fno_geom_attn | standard | 0.2165 | 1.0244 | 1.1129 | N/A | 0.000004 |
| fno_geom_attn | mixed | 0.1700 | 0.9837 | 0.9488 | N/A | 0.000001 |
| tfno | standard | 0.1815 | 1.0230 | 1.0835 | N/A | 0.000001 |
| tfno | mixed | 0.1974 | 0.9517 | 0.9630 | N/A | 0.000001 |

### High-Resolution Test Set (96x96x192)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian | MSE (Muscle) |
|-------|----------|--------|---------------|-------------|-----------|--------------|
| unet | standard | 1.1094 | 0.5591 | 2.1327 | N/A | 0.000012 |
| fno_analytical | standard | 0.3089 | 0.9580 | 0.8468 | N/A | 0.000002 |
| fno_analytical | mixed | 0.1252 | 0.9717 | 0.9174 | N/A | 0.000000 |
| fno_no_analytical | standard | 0.4059 | 0.9800 | 1.1488 | N/A | 0.000001 |
| fno_no_analytical | mixed | 0.1052 | 1.0155 | 1.0006 | N/A | 0.000000 |
| fno_geom_attn | standard | 0.4162 | 1.1949 | 1.1167 | N/A | 0.000000 |
| fno_geom_attn | mixed | 0.1422 | 0.9803 | 1.0809 | N/A | 0.000000 |
| tfno | standard | 0.2987 | 1.0596 | 1.3865 | N/A | 0.000001 |
| tfno | mixed | 0.1721 | 0.9178 | 1.0508 | N/A | 0.000001 |


## Aggregated Results (Mean ± Std over 10 seeds)

### Low-Resolution Test Set (Aggregated)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian |
|-------|----------|--------|---------------|-------------|-----------|
| unet | standard | 0.5296 ± 0.1058 | 1.0160 ± 0.1367 | 1.1380 ± 0.0952 | N/A |
| fno_analytical | standard | 0.1524 ± 0.0202 | 0.9802 ± 0.0237 | 0.9131 ± 0.0410 | N/A |
| fno_analytical | mixed | 0.1441 ± 0.0155 | 0.9852 ± 0.0377 | 0.8954 ± 0.0392 | N/A |
| fno_no_analytical | standard | 0.1543 ± 0.0277 | 0.9651 ± 0.0307 | 0.9275 ± 0.0536 | N/A |
| fno_no_analytical | mixed | 0.1416 ± 0.0170 | 0.9850 ± 0.0276 | 0.9088 ± 0.0205 | N/A |
| fno_geom_attn | standard | 0.1949 ± 0.0335 | 0.9864 ± 0.0541 | 0.9650 ± 0.1161 | N/A |
| fno_geom_attn | mixed | 0.1762 ± 0.0163 | 0.9576 ± 0.0297 | 0.9691 ± 0.0795 | N/A |
| tfno | standard | 0.1659 ± 0.0170 | 0.9778 ± 0.0408 | 1.0110 ± 0.0682 | N/A |
| tfno | mixed | 0.1701 ± 0.0213 | 0.9705 ± 0.0321 | 1.0483 ± 0.1212 | N/A |

### High-Resolution Test Set (Aggregated)

| Model | Training | Rel L2 | L2 Norm Ratio | Grad Energy | Laplacian |
|-------|----------|--------|---------------|-------------|-----------|
| *Trilinear Interp.* | *baseline* | *0.2718* | *1.0000* | *0.6311* | *N/A* |
| *FNO (no analytical)* | *std + upsample* | *0.3833 ± 0.0725* | — | — | *N/A* |
| *FNO (analytical)* | *std + upsample* | *0.4088 ± 0.0651* | — | — | *N/A* |
| unet | standard | 1.2509 ± 0.3877 | 0.7459 ± 0.5040 | 2.9643 ± 3.3392 | N/A |
| fno_analytical | standard | 0.4951 ± 0.1587 | 0.7518 ± 0.1711 | 0.8853 ± 0.0753 | N/A |
| **fno_analytical** | **mixed** | **0.1280 ± 0.0226** | **0.9961 ± 0.0610** | **0.9701 ± 0.0353** | **N/A** |
| fno_no_analytical | standard | 0.4461 ± 0.0835 | 0.7872 ± 0.1478 | 1.0517 ± 0.1039 | N/A |
| **fno_no_analytical** | **mixed** | **0.1291 ± 0.0305** | **1.0002 ± 0.0577** | **1.0345 ± 0.0575** | **N/A** |
| fno_geom_attn | standard | 0.5036 ± 0.1468 | 0.8694 ± 0.2296 | 0.9382 ± 0.2049 | N/A |
| fno_geom_attn | mixed | 0.1546 ± 0.0306 | 0.9652 ± 0.0499 | 1.0511 ± 0.0713 | N/A |
| tfno | standard | 0.4469 ± 0.1645 | 1.1437 ± 0.1961 | 1.2295 ± 0.1424 | N/A |
| tfno | mixed | 0.1625 ± 0.0402 | 0.9234 ± 0.0554 | 1.1500 ± 0.1837 | N/A |

**Key Finding**: Mixed-resolution training achieves 2× better super-resolution accuracy than trilinear interpolation (0.128 vs 0.272 Rel L2), while standard training performs worse than the naive baseline.


## Ensemble Results (10-Seed Ensemble)

Ensemble predictions were computed by averaging the predictions from all 10 seeds for the best model (FNO with analytical solution, mixed-resolution training).

### Ensemble vs Individual Model Performance

| Test Set | Ensemble Rel L2 | Mean Individual Rel L2 | Improvement |
|----------|-----------------|------------------------|-------------|
| Low-res (48×48×96) | **0.1134** | 0.1441 ± 0.0155 | 21.3% |
| High-res (96×96×192) | **0.0810** | 0.1280 ± 0.0226 | 36.7% |

### Additional Ensemble Metrics

| Metric | Low-res Ensemble | High-res Ensemble |
|--------|------------------|-------------------|
| Rel L2 | **0.1134** | **0.0810** |
| L2 Norm Ratio | 0.9826 | 0.9927 |
| Grad Energy Ratio | 0.8645 | 0.9125 |

**Key Finding**: Ensembling 10 seeds provides substantial improvements, especially for super-resolution where it achieves **36.7% reduction** in relative L2 error. The ensemble high-res Rel L2 of **0.0810** is the best result achieved in this study.