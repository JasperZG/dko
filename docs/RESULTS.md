# DKO Benchmark Results

**Date:** 2026-01-27
**Status:** Complete (54/54 experiments)

## Overview

This document presents benchmark results comparing DKO (Distribution Kernel Operators) against baseline methods for molecular property prediction across three datasets.

### Models Evaluated

| Model | Description |
|-------|-------------|
| **dko** | Full DKO with 2nd order features (mean μ + covariance Σ) |
| **dko_first_order** | DKO with 1st order features only (mean μ) |
| **attention** | Attention-based conformer aggregation |
| **mean_ensemble** | Simple averaging of conformer predictions |
| **single_conformer** | Lowest energy conformer only |
| **deepsets** | DeepSets permutation-invariant aggregation |

### Datasets

| Dataset | Task | Size | Property |
|---------|------|------|----------|
| ESOL | Regression | 1,128 | Water solubility (log S) |
| FreeSolv | Regression | 642 | Hydration free energy |
| Lipophilicity | Regression | 4,200 | Octanol/water partition coefficient |

---

## Results by Dataset

### ESOL (Water Solubility)

| Rank | Model | RMSE | MAE | R² | Pearson |
|------|-------|------|-----|-----|---------|
| 1 | **attention** | **1.903 ± 0.005** | 1.464 | +0.113 | 0.390 |
| 2 | mean_ensemble | 2.016 ± 0.070 | 1.437 | +0.003 | 0.331 |
| 3 | dko_first_order | 2.033 ± 0.053 | 1.486 | -0.013 | 0.270 |
| 4 | single_conformer | 2.396 ± 0.117 | 1.718 | -0.409 | 0.248 |
| 5 | deepsets | 2.535 ± 0.483 | 1.981 | -0.632 | 0.398 |
| 6 | dko | 2.539 ± 0.091 | 1.870 | -0.581 | 0.069 |

**Winner:** Attention (RMSE=1.903, only model with positive R²)

### FreeSolv (Hydration Free Energy)

| Rank | Model | RMSE | MAE | R² | Pearson |
|------|-------|------|-----|-----|---------|
| 1 | **single_conformer** | **4.089 ± 0.112** | 2.998 | -0.204 | 0.165 |
| 2 | attention | 4.125 ± 0.096 | 3.179 | -0.225 | 0.205 |
| 3 | mean_ensemble | 4.164 ± 0.066 | 3.056 | -0.248 | 0.123 |
| 4 | deepsets | 4.349 ± 0.234 | 3.439 | -0.365 | 0.211 |
| 5 | dko_first_order | 4.547 ± 0.060 | 3.441 | -0.488 | 0.151 |
| 6 | dko | 5.576 ± 0.222 | 4.441 | -1.240 | 0.040 |

**Winner:** Single conformer (RMSE=4.089, all models have negative R²)

### Lipophilicity (Partition Coefficient)

| Rank | Model | RMSE | MAE | R² | Pearson |
|------|-------|------|-----|-----|---------|
| 1 | **dko_first_order** | **1.135 ± 0.003** | 0.914 | +0.051 | 0.258 |
| 2 | attention | 1.140 ± 0.010 | 0.914 | +0.042 | 0.214 |
| 3 | mean_ensemble | 1.158 ± 0.012 | 0.934 | +0.011 | 0.238 |
| 4 | dko | 1.165 ± 0.009 | 0.938 | -0.000 | 0.125 |
| 5 | deepsets | 1.176 ± 0.028 | 0.935 | -0.019 | 0.191 |
| 6 | single_conformer | 1.190 ± 0.020 | 0.955 | -0.044 | 0.097 |

**Winner:** DKO First Order (RMSE=1.135, highest R²=0.051)

---

## Overall Rankings

| Rank | Model | Avg Rank | Dataset Ranks | Notes |
|------|-------|----------|---------------|-------|
| 1 | **attention** | **1.67** | [1, 2, 2] | Best overall, consistent |
| 2 | mean_ensemble | 2.67 | [2, 3, 3] | Simple but effective |
| 3 | dko_first_order | 3.00 | [3, 5, 1] | Won on Lipophilicity |
| 4 | single_conformer | 3.67 | [4, 1, 6] | Won on FreeSolv |
| 5 | deepsets | 4.67 | [5, 4, 5] | Inconsistent |
| 6 | dko | 5.33 | [6, 6, 4] | Last or near-last |

---

## DKO Analysis: 1st Order vs 2nd Order

### Per-Seed Results

**DKO 2nd Order (μ + Σ):**
| Dataset | Seed 42 | Seed 123 | Seed 456 | Average |
|---------|---------|----------|----------|---------|
| ESOL | RMSE=2.64, R²=-0.71 | RMSE=2.55, R²=-0.60 | RMSE=2.42, R²=-0.44 | RMSE=2.54, R²=-0.58 |
| FreeSolv | RMSE=5.89, R²=-1.49 | RMSE=5.38, R²=-1.08 | RMSE=5.47, R²=-1.15 | RMSE=5.58, R²=-1.24 |
| Lipophilicity | RMSE=1.16, R²=+0.01 | RMSE=1.16, R²=+0.01 | RMSE=1.18, R²=-0.02 | RMSE=1.16, R²=-0.00 |

**DKO 1st Order (μ only):**
| Dataset | Seed 42 | Seed 123 | Seed 456 | Average |
|---------|---------|----------|----------|---------|
| ESOL | RMSE=1.99, R²=+0.03 | RMSE=2.11, R²=-0.09 | RMSE=2.00, R²=+0.02 | RMSE=2.03, R²=-0.01 |
| FreeSolv | RMSE=4.57, R²=-0.50 | RMSE=4.46, R²=-0.43 | RMSE=4.61, R²=-0.53 | RMSE=4.55, R²=-0.49 |
| Lipophilicity | RMSE=1.14, R²=+0.04 | RMSE=1.13, R²=+0.05 | RMSE=1.13, R²=+0.05 | RMSE=1.13, R²=+0.05 |

### Head-to-Head Comparison

| Dataset | DKO 2nd Order | DKO 1st Order | Winner | Improvement |
|---------|---------------|---------------|--------|-------------|
| ESOL | RMSE=2.54 | RMSE=2.03 | **1st Order** | 20% better |
| FreeSolv | RMSE=5.58 | RMSE=4.55 | **1st Order** | 18% better |
| Lipophilicity | RMSE=1.16 | RMSE=1.13 | **1st Order** | 3% better |

**DKO 1st Order beats DKO 2nd Order on ALL datasets.**

---

## Key Findings

### 1. DKO 2nd Order Model is Broken
- Consistently ranks last or near-last on all datasets
- Negative R² everywhere (worse than predicting the mean)
- The covariance (Σ) features are hurting, not helping
- Likely causes: numerical instability in covariance processing, PCA issues

### 2. DKO 1st Order Works But Isn't Best
- Competitive performance (won on Lipophilicity)
- Average rank of 3.0 across datasets
- Outperformed by simpler attention baseline

### 3. Attention is the Best Overall Method
- Ranked #1 or #2 on every dataset
- Simple attention-based conformer aggregation is effective
- Highest R² on ESOL (+0.113)

### 4. All Models Struggle with R²
- Most models have near-zero or negative R²
- Even winners barely beat mean prediction
- Suggests features or training need improvement

### 5. Simple Baselines are Competitive
- mean_ensemble: consistent #2-3 ranking
- single_conformer: won on FreeSolv
- Complex models don't guarantee better results

---

## Recommendations

1. **Fix DKO 2nd Order:** The covariance processing needs debugging. Numerical stability issues with large covariance matrices (1024x1024) cause the model to learn nothing useful.

2. **Improve Features:** Near-zero R² across all models suggests the geometric features may not capture property-relevant information well.

3. **Use Attention as Baseline:** For now, attention-based aggregation is the best approach for conformer ensemble modeling.

4. **Hyperparameter Tuning:** All models used default settings. Grid search could improve results.

---

## Experimental Setup

- **Seeds:** 42, 123, 456 (3 runs per model per dataset)
- **Epochs:** 300 max with early stopping (patience=30)
- **Optimizer:** AdamW (lr=1e-5 for DKO, 1e-4 for others)
- **Batch Size:** 32
- **Mixed Precision:** Disabled for DKO (numerical stability)
- **Hardware:** Single NVIDIA GPU

---

## Files

- `results/benchmark/benchmark_results.json` - Full results with all metrics
- `results/benchmark/benchmark_summary.json` - Aggregated statistics
- `results/benchmark/logs/` - Per-experiment training logs
- `results/benchmark/checkpoints/` - Model checkpoints (every 25 epochs + best)
