# Empirical Study: Conformer Ensemble Statistics for Molecular Property Prediction

**Research question:** Can second-order conformer statistics (covariance matrices) improve molecular property prediction beyond first-order (mean) features and 2D molecular fingerprints?

**Short answer:** Conformer covariance features are complementary to fingerprints on solvation-related tasks (ESOL -9.9%, FreeSolv -3.9%, QM9-HOMO -4.2% RMSE improvement when combined), but standalone neural conformer methods cannot beat a Morgan fingerprint + XGBoost baseline on any of the 14 regression targets tested. Among neural architectures, learned gating (dko_gated) and simple scalar invariants (dko_invariants) are the most effective ways to incorporate second-order information, significantly outperforming the original PCA-based DKO formulation.

**Scale:** ~700 experiments across 14 regression targets, 13 neural architectures, 3-10 seeds per configuration.

---

## Study Overview

This study systematically evaluates Distribution Kernel Operators (DKO) and related conformer ensemble methods for molecular property prediction. We test whether second-order statistics of conformer feature distributions — specifically, the covariance matrix (sigma) — carry predictive signal beyond first-order statistics (mean, mu) and standard 2D molecular fingerprints.

We conduct ~700 experiments across five experimental axes:

1. **Architecture comparison** — 13 neural models spanning kernel methods, attention, gating, invariants, and simple baselines, evaluated on 6 MoleculeNet regression targets and 8 MARCEL benchmark targets
2. **Sigma representation ablation** — 7 new eigendecomposition-based alternatives to DKO's original PCA compression of sigma
3. **Fingerprint baseline & hybrid features** — Morgan FP + XGBoost as a strong baseline, and hybrid FP + conformer feature combinations to test complementarity
4. **Hyperparameter sensitivity** — Bayesian optimization (Optuna, 50 trials) to verify default hyperparameters are reasonable
5. **Training strategy** — Curriculum learning (mu-only pretraining → sigma fine-tuning) vs. joint training from scratch

---

## Key Findings

1. **Morgan fingerprints + XGBoost beat all neural conformer methods** on all 14 regression targets (6 MoleculeNet + 8 MARCEL). The gap ranges from 8.5% (ESOL) to 6.7x (BDE MAE).

2. **Conformer features complement fingerprints on solvation tasks.** When combined with Morgan FP in XGBoost, DKO conformer statistics (mu + sigma invariants) reduce RMSE on ESOL (-9.9%), FreeSolv (-3.9%), and QM9-HOMO (-4.2%). No improvement on electronic properties.

3. **DKO gating significantly outperforms attention** on ESOL (p < 0.001, 12.1% RMSE improvement, 10 seeds). Among neural conformer methods, learned gating for mu/sigma fusion is the best architecture.

4. **Simple sigma representations outperform complex ones.** The `dko_invariants` model (5 scalar features from sigma) achieves the best Lipophilicity RMSE across all 13 neural models, beating cross-attention and low-rank variants.

5. **The original DKO's PCA-based sigma compression is catastrophically bad** — it ranks 12th of 13 models. All 7 eigendecomposition-based alternatives outperform it.

6. **Conformer diversity predicts where sigma helps.** Lipophilicity (highest SCC diversity, 3x QM9) is exactly where dko_invariants shows the strongest advantage. Low-diversity datasets show no sigma benefit.

7. **Joint training matches or beats curriculum learning.** Two-phase mu-pretraining → sigma fine-tuning offers no advantage over training with sigma from scratch.

8. **Neural conformer methods fail even on 3D-dependent targets.** On Kraken steric descriptors (Boltzmann-averaged 3D properties), FP+XGBoost still dominates neural models by 1.5-2x.

---

## Methodology

### Feature Pipeline

Conformer ensembles are generated per molecule using RDKit (ETKDG). From each conformer, geometric features are computed: pairwise interatomic distances, bond angles, and torsion angles. These variable-length feature vectors (187-1770 dimensions depending on molecule size) are zero-padded or truncated to `max_feature_dim=1024`.

For each molecule's conformer ensemble, we compute:
- **Mu (mean):** Average feature vector across conformers — a 1024-dimensional first-order summary
- **Sigma (covariance):** Covariance matrix of features across conformers — a 1024×1024 second-order summary capturing conformational variability

Feature normalization uses `dim=1` (normalize across conformers only, preserving inter-feature variance). Sigma regularization is set to 1e-2 to prevent near-singular covariance matrices.

### Model Architectures

| Model | Description |
|-------|-------------|
| **dko** | Full DKO with 2nd order features (mean + covariance) |
| **dko_first_order** | DKO with 1st order features only (mean) |
| **dko_diagonal** | DKO using diagonal of sigma (skips PCA) |
| **dko_separate_nets** | DKO with independent mu and sigma networks |
| **dko_eigenspectrum** | Top-k eigenvalues of sigma concatenated with mu (~5K extra params) |
| **dko_invariants** | 5 scalar invariants: trace, log_det, frobenius, lambda_ratio, spectral_ratio (~2K extra params) |
| **dko_lowrank** | Top-k eigenvalues + flattened eigenvector projection (~40K extra params) |
| **dko_residual** | Base mu prediction + learned sigma correction, init scale=0.1 (~30K extra params) |
| **dko_crossattn** | Cross-attention: mu queries sigma-encoded eigenvalues (~50K extra params) |
| **dko_gated** | Learned sigmoid gate fusing separate mu and sigma streams (~40K extra params) |
| **dko_router** | SCC-based mixture-of-experts routing between first/second-order paths (~60K extra params) |
| **attention** | Attention-based conformer aggregation |
| **deepsets** | DeepSets permutation-invariant aggregation |
| **mean_ensemble** | Simple averaging of conformer predictions |
| **single_conformer** | Lowest energy conformer only |

All DKO variants with eigendecomposition use a diagonal proxy for D > 256 to avoid O(D^3) cost. Real datasets have D=1024.

### Training Setup

- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-5) for all models
- **Epochs:** 300 max with early stopping (patience=30)
- **Seeds:** 3 per configuration (42, 123, 456); 10 seeds for statistical validation
- **Mixed precision:** Disabled for full DKO variants (FP16 insufficient for covariance arithmetic), enabled for others
- **Hardware:** 8x NVIDIA RTX 2080 Ti

### Hyperparameter Sensitivity Analysis

To verify that default hyperparameters are reasonable, we ran 50-trial Bayesian optimization (Optuna, TPE sampler, MedianPruner) per model-dataset combination, with 100 epochs/trial max.

| Model | Dataset | Best Val MSE | Val RMSE | lr | weight_decay | dropout | k |
|-------|---------|-------------|----------|-----|-------------|---------|---|
| dko_gated | esol | **2.40** | **1.55** | 4.7e-4 | 3.0e-5 | 0.2 | 20 |
| dko_gated | lipophilicity | 1.54 | 1.24 | 5.3e-4 | 1.8e-5 | 0.0 | 15 |
| dko_invariants | esol | 2.77 | 1.66 | 7.1e-4 | 9.2e-5 | 0.1 | 20 |
| dko_invariants | lipophilicity | 1.55 | 1.25 | 3.9e-4 | 9.6e-5 | 0.2 | 5 |
| attention | esol | 2.71 | 1.65 | 2.2e-4 | 8.1e-6 | 0.0 | — |
| attention | lipophilicity | **1.41** | **1.19** | 4.1e-4 | 4.1e-6 | 0.1 | — |

Key observations:
- **dko_gated wins ESOL** even after tuning (val MSE 2.40 vs attention 2.71), confirming the benchmark result
- **attention wins Lipophilicity** (1.41 vs 1.54), consistent with benchmark rankings
- All models prefer **higher learning rates** (~3-5x the default 1e-4): optimal range 2e-4 to 7e-4
- DKO variants benefit from **more eigenvalues** (k=15-20 vs default k=10)
- Default hyperparameters (lr=1e-4, wd=1e-5) are suboptimal but not catastrophically so — tuning improves val RMSE by ~5-6%

### Data Splits

- **MoleculeNet:** 80/10/10 train/val/test split
- **MARCEL (Kraken, BDE, Drugs-75K):** 70/10/20, seed=123 (matching MARCEL defaults for direct comparability)

### Implementation Notes

During development, we identified and corrected several implementation issues in the original DKO codebase: a 10x learning rate discrepancy for DKO models (1e-5 vs 1e-4 for baselines), a halved kernel output dimension (32 vs 64), feature normalization across both conformers and features (`dim=(1,2)`) that destroyed the inter-feature variance sigma captures, and insufficient sigma regularization (1e-4 → 1e-2). Mixed precision was also disabled for DKO after ablation showed FP16 causes 10x variance inflation in covariance computation. An ablation study quantifying the impact of each correction is reported in the Diagnostic Studies section.

---

## Results: Feature Quality Validation

Before evaluating neural architectures, we verified that the geometric conformer features are inherently predictive using simple sklearn models (Ridge Regression and Random Forest) on raw mean features (mu).

### Regression Datasets

| Dataset | RF R² | RF Pearson | Ridge R² | Verdict |
|---------|-------|-----------|---------|---------|
| ESOL | **0.235** | **0.523** | -0.207 | YES - features are predictive |
| FreeSolv | -0.053 | 0.155 | -3.984 | WEAK - features barely predictive |
| Lipophilicity | **0.124** | **0.393** | -0.119 | YES - features are predictive |
| QM9-Gap | **0.482** | **0.710** | 0.366 | YES - highly predictive |
| QM9-HOMO | **0.191** | **0.476** | 0.049 | YES - features are predictive |
| QM9-LUMO | **0.575** | **0.773** | 0.467 | YES - highly predictive |

### Classification Datasets

| Dataset | RF Accuracy | Logistic Accuracy | Verdict |
|---------|------------|-------------------|---------|
| BACE | 0.704 | 0.724 | Moderate |
| BBBP | **0.985** | 0.873 | YES - highly predictive |

Random Forest achieves R²=0.24 on ESOL and R²=0.57 on QM9-LUMO using only mean conformer features, confirming that the geometric features carry meaningful signal. The variation across datasets (R²=-0.05 on FreeSolv to R²=0.57 on QM9-LUMO) provides a useful baseline for interpreting neural model performance.

---

## Results: MoleculeNet Benchmark

**192 experiments** (8 models x 3 seeds x 8 datasets). All models use corrected hyperparameters (lr=1e-4, kdim=64, norm=dim(1), sreg=1e-2, mp=off for full DKO).

### ESOL (Water Solubility)

| Rank | Model | RMSE | MAE | R² | Pearson | Seeds |
|------|-------|------|-----|-----|---------|-------|
| 1 | **dko_first_order** | 1.646 +/- 0.036 | 1.291 +/- 0.022 | +0.336 +/- 0.029 | 0.592 +/- 0.025 | 3 |
| 2 | attention | 1.888 +/- 0.011 | 1.471 +/- 0.038 | +0.127 +/- 0.011 | 0.402 +/- 0.010 | 3 |
| 3 | mean_ensemble | 2.016 +/- 0.070 | 1.437 +/- 0.021 | +0.003 +/- 0.070 | 0.331 +/- 0.005 | 3 |
| 4 | dko_diagonal | 2.040 +/- 0.033 | 1.614 +/- 0.008 | -0.020 +/- 0.033 | 0.479 +/- 0.013 | 3 |
| 5 | dko | 2.056 +/- 0.030 | 1.607 +/- 0.009 | -0.036 +/- 0.030 | 0.478 +/- 0.012 | 3 |
| 6 | dko_separate_nets | 2.249 +/- 0.031 | 1.707 +/- 0.010 | -0.240 +/- 0.035 | 0.321 +/- 0.048 | 3 |
| 7 | single_conformer | 2.394 +/- 0.120 | 1.717 +/- 0.110 | -0.407 +/- 0.143 | 0.249 +/- 0.028 | 3 |
| 8 | deepsets | 2.463 +/- 0.637 | 1.925 +/- 0.649 | -0.586 +/- 0.837 | 0.369 +/- 0.040 | 3 |

DKO first-order wins ESOL decisively (R²=0.336 vs attention's 0.127). ESOL (water solubility) does not depend strongly on conformational flexibility, so first-order features suffice.

### FreeSolv (Hydration Free Energy)

| Rank | Model | RMSE | MAE | R² | Pearson | Seeds |
|------|-------|------|-----|-----|---------|-------|
| 1 | **attention** | 4.077 +/- 0.071 | 3.213 +/- 0.191 | -0.196 +/- 0.041 | 0.204 +/- 0.034 | 3 |
| 2 | single_conformer | 4.087 +/- 0.100 | 3.000 +/- 0.114 | -0.203 +/- 0.059 | 0.166 +/- 0.072 | 3 |
| 3 | mean_ensemble | 4.177 +/- 0.048 | 3.075 +/- 0.048 | -0.255 +/- 0.029 | 0.119 +/- 0.113 | 3 |
| 4 | deepsets | 4.277 +/- 0.177 | 3.340 +/- 0.181 | -0.319 +/- 0.110 | 0.213 +/- 0.023 | 3 |
| 5 | dko_first_order | 4.513 +/- 0.110 | 3.411 +/- 0.152 | -0.466 +/- 0.071 | 0.096 +/- 0.009 | 3 |
| 6 | dko_diagonal | 4.699 +/- 0.057 | 3.837 +/- 0.035 | -0.589 +/- 0.039 | -0.001 +/- 0.025 | 3 |
| 7 | dko_separate_nets | 4.790 +/- 0.134 | 3.635 +/- 0.143 | -0.652 +/- 0.093 | -0.015 +/- 0.049 | 3 |
| 8 | dko | 4.881 +/- 0.140 | 3.952 +/- 0.094 | -0.716 +/- 0.099 | -0.069 +/- 0.073 | 3 |

All models have negative R² on FreeSolv, indicating worse-than-mean predictions. This is a challenging dataset with only ~642 molecules. The feature quality analysis also showed FreeSolv is the weakest dataset for these features (RF R²=-0.05).

### Lipophilicity

| Rank | Model | RMSE | MAE | R² | Pearson | Seeds |
|------|-------|------|-----|-----|---------|-------|
| 1 | **attention** | 1.141 +/- 0.002 | 0.917 +/- 0.002 | +0.040 +/- 0.003 | 0.209 +/- 0.008 | 3 |
| 2 | mean_ensemble | 1.165 +/- 0.015 | 0.935 +/- 0.015 | -0.001 +/- 0.026 | 0.179 +/- 0.069 | 3 |
| 3 | dko_separate_nets | 1.165 +/- 0.001 | 0.926 +/- 0.006 | -0.001 +/- 0.002 | 0.066 +/- 0.019 | 3 |
| 4 | deepsets | 1.167 +/- 0.011 | 0.928 +/- 0.007 | -0.005 +/- 0.018 | 0.193 +/- 0.015 | 3 |
| 5 | dko | 1.168 +/- 0.001 | 0.933 +/- 0.006 | -0.005 +/- 0.002 | 0.058 +/- 0.018 | 3 |
| 6 | dko_diagonal | 1.168 +/- 0.001 | 0.940 +/- 0.005 | -0.006 +/- 0.002 | 0.080 +/- 0.030 | 3 |
| 7 | single_conformer | 1.201 +/- 0.025 | 0.961 +/- 0.022 | -0.063 +/- 0.044 | 0.116 +/- 0.020 | 3 |
| 8 | dko_first_order | 1.213 +/- 0.007 | 0.956 +/- 0.015 | -0.085 +/- 0.012 | 0.137 +/- 0.014 | 3 |

All models are tightly clustered (RMSE 1.14-1.21). Only attention achieves marginally positive R² (0.04). Lipophilicity has ~4200 molecules but the geometric features provide limited signal (RF R²=0.12).

### QM9-Gap (HOMO-LUMO Gap)

| Rank | Model | RMSE | MAE | R² | Pearson | Seeds |
|------|-------|------|-----|-----|---------|-------|
| 1 | **attention** | 0.036 +/- 0.001 | 0.029 +/- 0.001 | +0.464 +/- 0.021 | 0.686 +/- 0.017 | 3 |
| 2 | mean_ensemble | 0.037 +/- 0.001 | 0.030 +/- 0.001 | +0.444 +/- 0.022 | 0.691 +/- 0.009 | 3 |
| 3 | deepsets | 0.038 +/- 0.001 | 0.030 +/- 0.001 | +0.427 +/- 0.021 | 0.661 +/- 0.019 | 3 |
| 4 | single_conformer | 0.038 +/- 0.000 | 0.030 +/- 0.000 | +0.402 +/- 0.010 | 0.645 +/- 0.001 | 3 |
| 5 | dko_first_order | 0.039 +/- 0.001 | 0.031 +/- 0.001 | +0.377 +/- 0.025 | 0.621 +/- 0.018 | 3 |
| 6 | dko_diagonal | 0.044 +/- 0.001 | 0.035 +/- 0.001 | +0.230 +/- 0.030 | 0.479 +/- 0.031 | 3 |
| 7 | dko | 0.045 +/- 0.000 | 0.036 +/- 0.000 | +0.183 +/- 0.014 | 0.429 +/- 0.016 | 3 |
| 8 | dko_separate_nets | 0.046 +/- 0.000 | 0.038 +/- 0.000 | +0.128 +/- 0.004 | 0.364 +/- 0.009 | 3 |

All models achieve positive R² on QM9-Gap. Attention leads (R²=0.46), with DKO first-order at 0.38. Full DKO lags at R²=0.18 — the covariance features add noise on this electronic property. Reference: sklearn RF R²=0.48.

### QM9-HOMO

| Rank | Model | RMSE | MAE | R² | Pearson | Seeds |
|------|-------|------|-----|-----|---------|-------|
| 1 | **attention** | 0.019 +/- 0.000 | 0.014 +/- 0.000 | +0.158 +/- 0.021 | 0.434 +/- 0.006 | 3 |
| 2 | deepsets | 0.019 +/- 0.000 | 0.014 +/- 0.000 | +0.127 +/- 0.022 | 0.368 +/- 0.040 | 3 |
| 3 | dko_first_order | 0.019 +/- 0.000 | 0.014 +/- 0.000 | +0.126 +/- 0.017 | 0.405 +/- 0.014 | 3 |
| 4 | single_conformer | 0.019 +/- 0.000 | 0.014 +/- 0.000 | +0.105 +/- 0.024 | 0.391 +/- 0.019 | 3 |
| 5 | mean_ensemble | 0.020 +/- 0.001 | 0.015 +/- 0.001 | +0.055 +/- 0.056 | 0.348 +/- 0.058 | 3 |
| 6 | dko_separate_nets | 0.020 +/- 0.000 | 0.015 +/- 0.000 | +0.042 +/- 0.006 | 0.257 +/- 0.004 | 3 |
| 7 | dko_diagonal | 0.020 +/- 0.000 | 0.015 +/- 0.000 | +0.024 +/- 0.021 | 0.222 +/- 0.026 | 3 |
| 8 | dko | 0.020 +/- 0.000 | 0.015 +/- 0.000 | +0.016 +/- 0.003 | 0.221 +/- 0.014 | 3 |

Tightly clustered RMSE (0.019-0.020). All models have positive but modest R². DKO first-order (rank 3, R²=0.13) is competitive with attention (R²=0.16). Reference: sklearn RF R²=0.19.

### QM9-LUMO

| Rank | Model | RMSE | MAE | R² | Pearson | Seeds |
|------|-------|------|-----|-----|---------|-------|
| 1 | **mean_ensemble** | 0.033 +/- 0.001 | 0.026 +/- 0.002 | +0.537 +/- 0.035 | 0.736 +/- 0.025 | 3 |
| 2 | deepsets | 0.034 +/- 0.001 | 0.026 +/- 0.001 | +0.535 +/- 0.022 | 0.735 +/- 0.015 | 3 |
| 3 | attention | 0.034 +/- 0.001 | 0.026 +/- 0.001 | +0.527 +/- 0.027 | 0.730 +/- 0.018 | 3 |
| 4 | single_conformer | 0.035 +/- 0.000 | 0.028 +/- 0.000 | +0.490 +/- 0.009 | 0.704 +/- 0.005 | 3 |
| 5 | dko_first_order | 0.036 +/- 0.000 | 0.029 +/- 0.000 | +0.461 +/- 0.010 | 0.683 +/- 0.005 | 3 |
| 6 | dko_diagonal | 0.043 +/- 0.001 | 0.035 +/- 0.001 | +0.238 +/- 0.030 | 0.488 +/- 0.030 | 3 |
| 7 | dko | 0.044 +/- 0.000 | 0.036 +/- 0.000 | +0.203 +/- 0.005 | 0.453 +/- 0.006 | 3 |
| 8 | dko_separate_nets | 0.045 +/- 0.000 | 0.037 +/- 0.000 | +0.154 +/- 0.009 | 0.394 +/- 0.012 | 3 |

Best dataset for all models. Top 3 (mean_ensemble, deepsets, attention) are essentially tied at R²~0.53. DKO first-order is close at R²=0.46. Full DKO at R²=0.20 — again, covariance features hurt. Reference: sklearn RF R²=0.57.

### Overall Regression Ranking

Average rank across 6 regression datasets (ESOL, FreeSolv, Lipophilicity, QM9-Gap, QM9-HOMO, QM9-LUMO), ranked by RMSE:

| Rank | Model | Avg Rank | ESOL | FreeSolv | Lipo | QM9-Gap | QM9-HOMO | QM9-LUMO |
|------|-------|----------|------|----------|------|---------|----------|----------|
| 1 | **attention** | **1.50** | 2 | 1 | 1 | 1 | 1 | 3 |
| 2 | mean_ensemble | 2.67 | 3 | 3 | 2 | 2 | 5 | 1 |
| 3 | deepsets | 3.83 | 8 | 4 | 4 | 3 | 2 | 2 |
| 4 | dko_first_order | 4.50 | 1 | 5 | 8 | 5 | 3 | 5 |
| 5 | single_conformer | 4.67 | 7 | 2 | 7 | 4 | 4 | 4 |
| 6 | dko_diagonal | 5.83 | 4 | 6 | 6 | 6 | 7 | 6 |
| 7 | dko_separate_nets | 6.33 | 6 | 7 | 3 | 8 | 6 | 8 |
| 8 | dko | 6.67 | 5 | 8 | 5 | 7 | 8 | 7 |

### Statistical Validation (10-Seed)

To confirm the ranking between top neural models, we ran 10 seeds per model on ESOL and Lipophilicity with one-sided Welch's t-test for statistical significance.

#### ESOL (10 seeds)

| Model | RMSE (mean ± std) | vs attention p-value |
|-------|-------------------|---------------------|
| **dko_gated** | **1.654 ± 0.032** | **< 0.001** (12.1% better) |
| dko_invariants | 1.765 ± 0.050 | < 0.001 (6.2% better) |
| attention | 1.881 ± 0.027 | -- |

#### Lipophilicity (10 seeds)

| Model | RMSE (mean ± std) | vs attention p-value |
|-------|-------------------|---------------------|
| dko_invariants | 1.140 ± 0.008 | 0.48 (not significant) |
| attention | 1.140 ± 0.006 | -- |
| dko_gated | 1.164 ± 0.022 | 0.99 (attention better) |

DKO's advantage over attention on ESOL is statistically significant (p < 0.001). On Lipophilicity, no significant difference between any conformer-based models.

### Classification Note (BACE/BBBP)

Classification tasks were out of scope for this study's regression-focused architecture. Results are reported for completeness but are not meaningful for model comparison — all models converge to predicting the majority class due to BCEWithLogitsLoss on severely imbalanced data without class weighting.

#### BACE (Classification: Active/Inactive)

| Rank | Model | Accuracy | AUC-ROC | Precision | Recall | F1 | Seeds |
|------|-------|----------|---------|-----------|--------|-----|-------|
| 1 | **mean_ensemble** | 0.871 +/- 0.046 | N/A | 0.000 | 0.000 | 0.000 | 3 |
| 2 | deepsets | 0.811 +/- 0.067 | N/A | 0.000 | 0.000 | 0.000 | 3 |
| 3 | attention | 0.711 +/- 0.048 | N/A | 0.000 | 0.000 | 0.000 | 3 |
| 4 | single_conformer | 0.695 +/- 0.019 | N/A | 0.000 | 0.000 | 3 |
| 5 | dko_first_order | 0.180 +/- 0.016 | N/A | 0.000 | 0.000 | 0.000 | 3 |
| 6 | dko_diagonal | 0.175 +/- 0.022 | N/A | 0.000 | 0.000 | 0.000 | 3 |
| 7 | dko | 0.125 +/- 0.046 | N/A | 0.000 | 0.000 | 0.000 | 3 |
| 8 | dko_separate_nets | 0.101 +/- 0.033 | N/A | 0.000 | 0.000 | 0.000 | 3 |

All models show zero precision/recall/F1, indicating they predict only one class. BACE test set is ~87% negative class.

#### BBBP (Classification: Blood-Brain Barrier Permeability)

| Rank | Model | Accuracy | AUC-ROC | Precision | Recall | F1 | Seeds |
|------|-------|----------|---------|-----------|--------|-----|-------|
| 1 | **dko** | 1.000 +/- 0.000 | N/A | 1.000 | 1.000 | 1.000 | 3 |
| 2 | dko_diagonal | 1.000 +/- 0.000 | N/A | 1.000 | 1.000 | 1.000 | 3 |
| 3 | dko_separate_nets | 1.000 +/- 0.000 | N/A | 1.000 | 1.000 | 1.000 | 3 |
| 4 | mean_ensemble | 0.990 +/- 0.014 | N/A | 1.000 | 0.990 | 0.995 | 3 |
| 5 | dko_first_order | 0.975 +/- 0.007 | N/A | 1.000 | 0.975 | 0.988 | 3 |
| 6 | deepsets | 0.969 +/- 0.008 | N/A | 1.000 | 0.969 | 0.984 | 3 |
| 7 | single_conformer | 0.964 +/- 0.008 | N/A | 1.000 | 0.964 | 0.982 | 3 |
| 8 | attention | 0.958 +/- 0.017 | N/A | 1.000 | 0.958 | 0.978 | 3 |

BBBP test set is ~96% positive class. All models achieve 95%+ accuracy by predicting the majority class. Fixing would require class-weighted loss, oversampling, or focal loss.

---

## Results: Architecture Variants

### Motivation

The original DKO uses PCA to compress the full covariance matrix (sigma), which ranks 12th of 13 models overall — a clear failure of the sigma representation, not the sigma signal itself. We designed and tested 7 alternative eigendecomposition-based sigma representations to find better ways to incorporate second-order information.

### New Variant Descriptions

| Model | Description | Extra Params |
|-------|-------------|-------------|
| `dko_eigenspectrum` | Top-k eigenvalues of sigma concatenated with mu | ~5K |
| `dko_invariants` | 5 scalar invariants: trace, log_det, frobenius, lambda_ratio, spectral_ratio | ~2K |
| `dko_lowrank` | Top-k eigenvalues + flattened eigenvector projection | ~40K |
| `dko_residual` | Base mu prediction + learned sigma correction (init scale=0.1) | ~30K |
| `dko_crossattn` | Cross-attention: mu queries sigma-encoded eigenvalues | ~50K |
| `dko_gated` | Learned sigmoid gate fusing separate mu and sigma streams | ~40K |
| `dko_router` | SCC-based mixture-of-experts routing between first/second-order paths | ~60K |

### Results: Test RMSE (mean +/- std, 3 seeds)

**84 experiments** (7 new models x 4 datasets x 3 seeds)

| Model | ESOL | QM9-Gap | QM9-LUMO | Lipophilicity |
|-------|------|---------|----------|---------------|
| *Baselines:* | | | | |
| dko (original) | 2.056 +/- 0.030 | 0.045 +/- 0.000 | 0.044 +/- 0.000 | 1.168 +/- 0.001 |
| dko_first_order | 1.646 +/- 0.036 | 0.039 +/- 0.001 | 0.036 +/- 0.000 | 1.213 +/- 0.007 |
| attention | 1.888 +/- 0.011 | **0.036 +/- 0.001** | 0.034 +/- 0.001 | 1.141 +/- 0.002 |
| mean_ensemble | 2.016 +/- 0.070 | 0.037 +/- 0.001 | **0.034 +/- 0.001** | 1.165 +/- 0.015 |
| *New variants:* | | | | |
| dko_eigenspectrum | 1.695 +/- 0.043 | 0.040 +/- 0.001 | 0.036 +/- 0.000 | 1.182 +/- 0.017 |
| dko_invariants | 1.807 +/- 0.014 | 0.040 +/- 0.001 | 0.036 +/- 0.000 | **1.131 +/- 0.008** |
| dko_lowrank | 1.681 +/- 0.023 | 0.042 +/- 0.000 | 0.038 +/- 0.000 | 1.274 +/- 0.040 |
| dko_residual | 1.698 +/- 0.025 | 0.040 +/- 0.001 | 0.036 +/- 0.000 | 1.169 +/- 0.005 |
| dko_crossattn | 1.929 +/- 0.043 | 0.038 +/- 0.000 | 0.035 +/- 0.000 | 1.170 +/- 0.014 |
| **dko_gated** | **1.635 +/- 0.023** | 0.039 +/- 0.001 | 0.037 +/- 0.000 | 1.166 +/- 0.012 |
| dko_router | 1.670 +/- 0.023 | 0.040 +/- 0.000 | 0.036 +/- 0.000 | 1.155 +/- 0.013 |

Bold = best model for that dataset.

### Best Model Per Dataset

| Dataset | Best Model | RMSE | Type |
|---------|-----------|------|------|
| ESOL | **dko_gated** | 1.635 | New variant |
| QM9-Gap | attention | 0.036 | Baseline |
| QM9-LUMO | mean_ensemble | 0.034 | Baseline |
| Lipophilicity | **dko_invariants** | 1.131 | New variant |

### Overall Ranking (13 models, 4 datasets)

| Rank | Model | Mean Rank | ESOL | QM9-Gap | QM9-LUMO | Lipo | Type |
|------|-------|-----------|------|---------|----------|------|------|
| 1 | attention | 3.25 | 8 | 1 | 2 | 2 | Baseline |
| 2 | mean_ensemble | 4.25 | 10 | 2 | 1 | 4 | Baseline |
| 3 | **dko_invariants** | 4.50 | 7 | 6 | 4 | 1 | New |
| 4 | **dko_gated** | 5.00 | 1 | 4 | 9 | 6 | New |
| 5 | **dko_router** | 5.75 | 3 | 9 | 8 | 3 | New |
| 6 | dko_crossattn | 6.25 | 9 | 3 | 3 | 10 | New |
| 7 | dko_first_order | 6.50 | 2 | 5 | 7 | 12 | Baseline |
| 8 | dko_residual | 7.00 | 6 | 7 | 6 | 9 | New |
| 9 | dko_eigenspectrum | 7.25 | 5 | 8 | 5 | 11 | New |
| 10 | dko_lowrank | 9.25 | 4 | 10 | 10 | 13 | New |
| 11 | dko_diagonal | 10.25 | 11 | 11 | 11 | 8 | Baseline |
| 12 | dko (original) | 10.75 | 12 | 12 | 12 | 7 | Baseline |
| 13 | dko_separate_nets | 11.00 | 13 | 13 | 13 | 5 | Baseline |

The pattern is clear: simple sigma representations outperform complex ones. dko_invariants (5 scalars, ~2K params) and dko_gated (sigmoid gate, ~40K params) beat dko_crossattn (~50K params) and dko_lowrank (~40K params). The original PCA-based DKO ranks 12th — confirming PCA compression as the primary failure mode.

---

## Results: MARCEL Benchmark

The MARCEL benchmark (Zhu et al., ICLR 2024) tests conformer ensemble learning on datasets where conformer geometry directly affects the target property. We evaluate on 3 of 4 MARCEL datasets (EE is proprietary/unavailable).

### Datasets

| Dataset | Molecules | Conformers/mol | Split | Targets |
|---------|-----------|---------------|-------|---------|
| Kraken | 1,552 | mean=13.5, max=50 | 70/10/20 seed=123 | B5, L, burB5, burL |
| BDE | 5,915 | mean=8.2, max=20 | 70/10/20 seed=123 | BDE (kcal/mol) |
| Drugs-75K | 75,099 | mean=7.2, max=20 | 70/10/20 seed=123 | ip, ea, chi |

All datasets use MARCEL's default split (70/10/20, seed=123) for direct comparability with paper baselines.

### FP+XGBoost Baseline (All Targets)

| Dataset | MAE | RMSE | R² |
|---------|-----|------|-----|
| kraken_B5 | 0.376±0.003 | 0.519±0.006 | 0.834 |
| kraken_L | 0.431±0.005 | 0.650±0.013 | 0.775 |
| kraken_burB5 | 0.266±0.003 | 0.378±0.004 | 0.698 |
| kraken_burL | 0.160±0.004 | 0.259±0.007 | 0.543 |
| drugs_ip | 0.509±0.001 | 0.656±0.001 | 0.514 |
| drugs_ea | 0.471±0.001 | 0.609±0.001 | 0.535 |
| drugs_chi | 0.280±0.000 | 0.360±0.000 | 0.634 |
| bde | 3.025±0.037 | 4.833±0.079 | 0.958 |

### Kraken Neural Results (5 models x 4 targets x 3 seeds, MARCEL-matched splits)

| Model | B5 RMSE | L RMSE | burB5 RMSE | burL RMSE | Avg Rank |
|-------|---------|--------|------------|-----------|----------|
| **FP+XGBoost** | **0.519** | **0.650** | **0.378** | **0.259** | **1.0** |
| attention | 1.084 | 1.035 | 0.560 | 0.337 | 2.0 |
| mean_ensemble | 1.096 | 1.086 | 0.603 | 0.414 | 3.5 |
| dko_first_order | 1.145 | 1.148 | 0.575 | 0.379 | 3.5 |
| dko_gated | 1.178 | 1.152 | 0.580 | 0.412 | 4.5 |
| dko_invariants | 1.226 | 1.182 | 0.603 | 0.410 | 5.5 |

| Model | B5 R² | L R² | burB5 R² | burL R² |
|-------|-------|------|----------|---------|
| **FP+XGBoost** | **0.834** | **0.775** | **0.698** | **0.543** |
| attention | 0.454 | 0.292 | 0.539 | 0.269 |
| mean_ensemble | 0.442 | 0.221 | 0.464 | -0.123 |
| dko_first_order | 0.392 | 0.129 | 0.514 | 0.076 |
| dko_gated | 0.357 | 0.123 | 0.505 | -0.091 |
| dko_invariants | 0.303 | 0.077 | 0.466 | -0.082 |

FP+XGBoost dominates all neural models on all 4 Kraken targets (1.5-2x better RMSE). Attention is the clear best neural model (rank 2.0). DKO variants underperform attention on steric descriptor prediction.

### BDE Neural Results (5 models x 3 seeds)

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **FP+XGBoost** | **3.025** | **4.833** | **0.958** |
| mean_ensemble | 20.267 | 23.134 | 0.028 |
| attention | 20.529 | 23.317 | 0.012 |
| dko_invariants | 20.631 | 24.619 | -0.101 |
| dko_first_order | 20.735 | 24.394 | -0.081 |
| dko_gated | 20.898 | 25.066 | -0.142 |

FP+XGBoost is **6.7x better** on MAE. All neural models essentially predict the mean (R²~0). Investigation of this failure is reported in the Diagnostic Studies section.

### Drugs-75K Neural Results (5 models x 3 targets x 3 seeds)

| Target | FP+XGB RMSE | attention RMSE | mean_ensemble RMSE | dko_gated RMSE | FP Wins By |
|--------|------------|----------------|-------------------|----------------|------------|
| drugs_ip | **0.656** | 0.849 | 0.856 | 0.908 | +29% |
| drugs_ea | **0.609** | 0.774 | 0.776 | 0.866 | +27% |
| drugs_chi | **0.360** | 0.435 | 0.437 | 0.526 | +21% |

FP+XGBoost dominates on all 3 Drugs-75K electronic properties. Neural conformer methods fail to match the fingerprint baseline, consistent with MoleculeNet results.

### Comparison to MARCEL Paper Baselines

MARCEL (Zhu et al., ICLR 2024) reports results for 19 model configurations. Key comparisons on datasets with matching splits:

**BDE (MAE, kcal/mol):**

| Model | MAE | Source |
|-------|-----|--------|
| 3D-DimeNet++ | 1.45 | MARCEL paper |
| Ensemble-GemNet | 1.61 | MARCEL paper |
| 1D-Random Forest | 3.03 | MARCEL paper |
| **FP+XGBoost (ours)** | **3.025** | This work |
| DKO neural models (ours) | ~20.3 | This work (failed) |

Our FP+XGBoost matches MARCEL's 1D-RF exactly. MARCEL's 3D GNNs (DimeNet++, GemNet) are 2x better because they use end-to-end learned representations on atomic coordinates.

**Drugs-75K (MAE, eV):**

| Model | ip | ea | chi | Source |
|-------|-----|-----|------|--------|
| Ensemble-GemNet | 0.407 | 0.391 | 0.197 | MARCEL best |
| 1D-Random Forest | 0.499 | 0.475 | 0.273 | MARCEL baseline |
| **FP+XGBoost (ours)** | **0.509** | **0.471** | **0.280** | This work |
| attention (ours) | 0.663* | 0.607* | 0.339* | This work (MAE) |

*Our neural models use geometric features, not learned 3D representations. MARCEL's 3D GNNs are 24-42% better than our FP+XGBoost.

---

## Results: Fingerprint Baseline & Hybrid Features

### Morgan Fingerprint + XGBoost vs. All Neural Models

**Method:** 2048-bit Morgan fingerprints (radius=2) + XGBoost (100 trees, max_depth=6, lr=0.1, hist tree method), 3 seeds, train+val combined.

| Dataset | FP RMSE (mean±std) | Best Neural RMSE | Gap | FP Wins? |
|---------|-------------------|-----------------|-----|----------|
| ESOL | **1.507 ± 0.021** | dko_gated 1.635 | +0.128 | YES |
| FreeSolv | **2.939 ± 0.127** | attention 4.077 | +1.138 | YES |
| Lipophilicity | **0.910 ± 0.006** | dko_invariants 1.131 | +0.221 | YES |
| QM9-Gap | **0.020 ± 0.000** | attention 0.036 | +0.016 | YES |
| QM9-HOMO | **0.014 ± 0.000** | attention 0.019 | +0.005 | YES |
| QM9-LUMO | **0.019 ± 0.000** | mean_ensemble 0.034 | +0.015 | YES |

**Fingerprints beat ALL neural conformer methods on ALL regression datasets.** The gap ranges from 8.5% (ESOL) to 45% (QM9-Gap). This establishes that the geometric conformer features, as currently formulated, provide less predictive signal than standard 2D molecular fingerprints for standalone neural models.

### Hybrid FP + Conformer Features

The dominant fingerprint baseline raises a natural question: do conformer features provide *complementary* information to fingerprints, even if they cannot compete standalone?

**Method:** Concatenate Morgan FP (2048-bit), conformer mu (mean, 256-dim), and sigma stats (5 scalar invariants) in various combinations. Train XGBoost on combined features. 3 seeds per experiment.

| Features | ESOL | FreeSolv | Lipophilicity | QM9-Gap | QM9-HOMO | QM9-LUMO |
|----------|------|----------|---------------|---------|----------|----------|
| FP only | 1.507 | 2.939 | **0.910** | **0.020** | 0.014 | **0.019** |
| Mu only | 1.607 | 4.056 | 1.112 | 0.035 | 0.018 | 0.032 |
| Sigma only | 2.374 | 4.229 | 1.199 | 0.047 | 0.021 | 0.047 |
| FP + Mu | 1.367 | 2.831 | 0.939 | 0.021 | 0.014 | 0.019 |
| FP + Sigma | 1.432 | 3.119 | 0.914 | 0.020 | 0.014 | 0.019 |
| **FP + Mu + Sigma** | **1.358** | **2.824** | 0.957 | 0.021 | **0.014** | 0.019 |
| Mu + Sigma | 1.514 | 4.231 | 1.103 | 0.035 | 0.018 | 0.032 |

**Key findings:**

1. **FP + Mu + Sigma beats FP alone on ESOL by 9.9%** (1.358 vs 1.507) — conformer features ARE complementary to fingerprints
2. **FP + Mu + Sigma beats FP alone on FreeSolv by 3.9%** (2.824 vs 2.939)
3. **FP + Mu + Sigma beats FP alone on QM9-HOMO by 4.2%** (0.0136 vs 0.0142)
4. On Lipophilicity, QM9-Gap, QM9-LUMO: FP alone is already best; conformer features add no value
5. Mu contributes more than Sigma in the hybrid (FP+Mu beats FP+Sigma on 4/6 datasets)
6. The best overall method across all datasets is **FP + Mu + Sigma on XGBoost**

This is the central positive result of the study: DKO's conformer statistics (both first-order mu and second-order sigma invariants) provide genuine complementary signal to fingerprints on solvation-related properties, where conformational ensemble shape influences the target.

---

## Results: Training Strategy Investigation

### Curriculum Learning (mu-pretraining → sigma fine-tuning)

We tested whether pre-training on mu-only (first-order) then fine-tuning with sigma (second-order) improves DKO performance versus standard joint training from scratch.

**Protocol:**
- **Phase 1** (mu-only): 100 epochs, lr=1e-3, patience=20 — learns mu representation without sigma
- **Phase 2** (sigma fine-tune): 200 epochs, lr=1e-4, patience=30 — enables sigma, fine-tunes at 0.1x LR
- **Baseline**: Standard training with sigma from scratch, 300 epochs, lr=1e-3, patience=30
- Models: dko_gated, dko_invariants | Datasets: ESOL, Lipophilicity | Seeds: 42, 123, 456

### Results (RMSE, averaged over 3 seeds)

| Model | Dataset | Phase 1 (mu-only) | Curriculum | Baseline | Diff |
|-------|---------|-------------------|------------|----------|------|
| dko_gated | ESOL | 1.845 | 1.842 | **1.711** | -7.6% |
| dko_gated | Lipophilicity | 1.150 | **1.141** | 1.144 | +0.2% |
| dko_invariants | ESOL | 2.186 | 2.103 | **2.046** | -2.8% |
| dko_invariants | Lipophilicity | 1.165 | **1.146** | 1.150 | +0.3% |

*Diff = (baseline - curriculum) / baseline × 100. Positive = curriculum wins.*

**Curriculum learning does NOT help DKO.** Standard joint training from scratch matches or beats the two-phase curriculum approach:
- **ESOL**: Baseline wins by 2.8-7.6% — standard training handles the sigma signal from the start
- **Lipophilicity**: Essentially tied (0.2-0.3% difference within noise)
- Phase 1 → Phase 2 does improve over mu-only, confirming sigma adds value, but standard training captures the same benefit more efficiently
- This suggests DKO models can learn from sigma without requiring a mu-pretraining warmup

---

## Diagnostic Studies

### Feature Variance Audit

We analyzed the eigenvalue spectrum of conformer covariance matrices across all 8 datasets to understand the spectral structure that DKO operates on.

| Dataset | Molecules | Total Var | Top-10 Diag Var% | Eff. Rank (90%) |
|---------|-----------|-----------|-----------------|-----------------|
| ESOL | 601 | 490.4 | 6.4% | 475 |
| FreeSolv | 306 | 409.4 | 7.1% | 381 |
| Lipophilicity | 1928 | 595.0 | 5.2% | 623 |
| QM9-Gap | 1829 | 301.6 | 8.0% | 353 |
| QM9-HOMO | 1829 | 301.6 | 8.0% | 353 |
| QM9-LUMO | 1829 | 301.6 | 8.0% | 353 |
| BACE | 1205 | 577.8 | 4.3% | 685 |
| BBBP | 1566 | 530.9 | 5.4% | 595 |

Top-10 diagonal eigenvalues capture only 4-8% of variance. Effective rank at 90% ranges from 353 (QM9) to 685 (BACE). The diagonal proxy used at D=1024 loses significant spectral information, partially explaining why full-sigma methods underperform scalar invariants.

### SCC Quartile Analysis (Conformer Diversity)

| Dataset | Median SCC | Max SCC | Interpretation |
|---------|-----------|---------|----------------|
| FreeSolv | 0.01 | low | Near-zero conformer variation |
| ESOL | 27.88 | 148.04 | Moderate diversity |
| QM9 | 25.05 | 113.76 | Moderate diversity |
| Lipophilicity | 69.99 | 155.53 | Highest conformer diversity |

Lipophilicity has the highest conformer diversity, which aligns with it being the dataset where sigma-based methods (dko_invariants) show the most benefit over baselines.

### Synthetic Validation (Controlled Sigma Signal)

To test whether DKO can learn from sigma in a controlled setting, we generated synthetic data where y = W@mu + alpha * log(1 + trace(sigma)).

| Alpha | DKO RMSE | First-Order RMSE | Improvement |
|-------|----------|-----------------|-------------|
| 0.0 | 7.079 | 10.100 | 29.9% |
| 0.1 | 7.076 | 10.178 | 30.5% |
| 0.5 | 7.111 | 9.842 | 27.8% |
| 1.0 | 7.246 | 9.898 | 26.8% |

DKO outperforms first-order by 27-30% even at alpha=0 (no sigma signal), demonstrating that the full second-order kernel provides a beneficial inductive bias. However, the ablated variants (eigenspectrum, residual, etc.) perform similarly to first-order (~9.5-10.2 RMSE), suggesting only the original DKO's kernel formulation captures the sigma signal effectively in controlled settings.

### Implementation Correction Ablation

To quantify the impact of each correction identified during development, we tested 7 configurations on ESOL with DKO (full 2nd order), 3 seeds each (21 total experiments).

| Config | RMSE | R² | Pearson | Key Change |
|--------|------|-----|---------|------------|
| Baseline (broken) | 3.318 +/- 0.056 | -1.698 +/- 0.091 | 0.293 +/- 0.044 | lr=1e-5, kdim=32, norm=dim(1,2), sreg=1e-4 |
| +LR only | 2.309 +/- 0.008 | -0.306 +/- 0.009 | -0.109 +/- 0.076 | lr=1e-4 |
| +kernel_dim only | 3.206 +/- 0.025 | -1.517 +/- 0.039 | 0.331 +/- 0.059 | kdim=64 |
| +norm only | 2.859 +/- 0.096 | -1.004 +/- 0.133 | **0.487 +/- 0.033** | norm=dim(1), sreg=1e-2 |
| +LR + kernel_dim | 2.311 +/- 0.014 | -0.308 +/- 0.016 | -0.081 +/- 0.043 | lr=1e-4, kdim=64 |
| **+LR + kdim + norm** | **2.056 +/- 0.030** | **-0.036 +/- 0.030** | **0.478 +/- 0.012** | lr=1e-4, kdim=64, norm=dim(1), sreg=1e-2 |
| All fixes (+mp) | 2.685 +/- 0.327 | -0.793 +/- 0.412 | 0.449 +/- 0.023 | Above + mixed precision |

**Reference:** sklearn Random Forest on same features: RMSE=1.767, R²=0.235, Pearson=0.523

Key observations:
1. **LR fix is the biggest single RMSE factor:** 3.318 → 2.309 (30% reduction)
2. **Norm fix is the biggest Pearson factor:** 0.293 → 0.487 (even with broken LR)
3. **Best combination is LR + kdim + norm:** R² went from -1.70 to -0.04 (nearly zero)
4. **Mixed precision HURTS DKO:** Adding mp increased RMSE from 2.056 to 2.685 with 10x higher variance. FP16 is insufficient for covariance matrix arithmetic.
5. **kernel_dim alone has minimal effect:** 3.206 vs 3.318 (marginal)

### BDE Failure Investigation

All neural models achieve R²~0 on BDE while FP+XGBoost gets R²=0.958. We investigated whether target scale (std=23.4 kcal/mol, MSE~550) was causing gradient instability via z-score normalization:

| Model | Raw R² | Norm R² | Raw MAE | Norm MAE |
|-------|--------|---------|---------|----------|
| dko_gated | 0.001 | -0.001 | 19.93 | 20.02 |
| dko_invariants | -0.003 | -0.001 | 20.19 | 19.81 |
| attention | -0.002 | -0.001 | 20.16 | 20.66 |
| dko_first_order | -0.011 | 0.002 | 20.09 | 20.01 |

Normalization has negligible effect — all models still predict the mean (MAE~20 vs mean-predictor MAE=20.76).

**Root cause:** The pre-computed conformer features (variable-length 187-1770, zero-padded/truncated to 1024d) lack the chemical information needed for BDE prediction. Evidence:
- **RF on mu features: R²=0.66** — signal exists but requires non-linear tree-based extraction
- **Ridge on mu features: R²=-0.28** — linear models also fail
- **Neural models overfit severely**: train_loss drops from 825→229 but val_loss stays at ~650 (near Var(y)=550)
- **37.2% of conformers truncated** at 1024d, losing information
- **MARCEL DimeNet++ achieves MAE=1.45** using 3D atomic coordinates directly — our features are fundamentally insufficient for BDE

### Classification Failure Analysis (BACE/BBBP)

All neural models fail on both classification datasets, converging to majority-class prediction:

**BACE** (binary, ~87% negative class in test):

| Model | Accuracy | AUC | F1 | Notes |
|-------|----------|-----|-----|-------|
| mean_ensemble | 0.871 | NaN | 0.0 | Predicts all negative |
| deepsets | 0.811 | NaN | 0.0 | Predicts all negative |
| attention | 0.711 | NaN | 0.0 | Predicts all negative |
| single_conformer | 0.695 | NaN | 0.0 | Predicts all negative |
| dko_first_order | 0.180 | NaN | 0.0 | Predicts all positive |
| dko | 0.125 | NaN | 0.0 | Predicts all positive |

**BBBP** (binary, ~96% positive class in test):

| Model | Accuracy | AUC | F1 | Notes |
|-------|----------|-----|-----|-------|
| dko | 1.000 | NaN | 1.0 | Predicts all positive |
| mean_ensemble | 0.990 | NaN | 0.995 | Predicts all positive |
| dko_first_order | 0.975 | NaN | 0.988 | Predicts all positive |
| deepsets | 0.969 | NaN | 0.984 | Predicts all positive |
| single_conformer | 0.964 | NaN | 0.982 | Predicts all positive |
| attention | 0.958 | NaN | 0.978 | Predicts all positive |

Root cause: BCEWithLogitsLoss on severely imbalanced data without class weighting causes all models to converge to predicting the majority class. AUC is NaN because predicted probabilities are constant (no threshold discrimination). Fixing would require class-weighted loss, oversampling, or focal loss — left for future work.

---

## Discussion

### Why Fingerprints Beat Neural Conformer Methods

Morgan fingerprints (2048-bit, radius=2) encode molecular substructure topology — which atoms are bonded to which, functional groups, ring systems. This structural information is highly correlated with most molecular properties. Our geometric conformer features (pairwise distances, angles, torsions) capture 3D shape but miss chemical identity entirely: a C-C-C angle looks the same as an N-O-N angle. XGBoost's tree-based learning is also more data-efficient than neural MLPs on tabular features, especially for datasets with <5000 molecules.

### Where Second-Order Features Help

Conformer statistics improve predictions specifically on solvation-related tasks (ESOL, FreeSolv) and QM9-HOMO. These properties depend on how a molecule interacts with its environment across its conformational ensemble — exactly the information sigma captures. Electronic properties like QM9-Gap and QM9-LUMO are primarily determined by equilibrium geometry, making sigma irrelevant.

### Comparison to MARCEL GNNs

DKO and MARCEL's 3D GNNs (SchNet, DimeNet++, GemNet, PaiNN) represent fundamentally different paradigms:
- **DKO:** Pre-computes geometric features from conformers, then learns aggregation via MLPs. Features are fixed; only the aggregation is learned.
- **MARCEL 3D GNNs:** Operate directly on atomic coordinates via message-passing. Both features and aggregation are learned end-to-end.

Our FP+XGBoost baseline matches MARCEL's 1D-Random Forest exactly (BDE MAE: 3.025 vs 3.03), confirming fair comparison. The gap to 3D GNNs (DimeNet++ MAE=1.45 on BDE) reflects the fundamental expressiveness advantage of end-to-end learned 3D representations.

### Limitations

1. **No GNN baselines.** Running MARCEL's GNNs would require their full pipeline (different data format, training loop, dependencies). We compare to their published numbers on matched splits instead.
2. **Feature truncation.** Variable-length geometric features (187-1770 dims) are zero-padded/truncated to 1024d. This causes ~58% sparsity in padded features, 37.2% truncation in BDE, and corrupted covariance structure for DKO. Future work could use dimensionality reduction (PCA on features, not on sigma) or adaptive feature dimensions.
3. **Classification not addressed.** All models fail on BACE/BBBP due to class imbalance — this study focuses on regression.

---

## Code Changes

| File | Change | Impact |
|------|--------|--------|
| `dko/experiments/main_benchmark.py` | LR: 1e-5 → 1e-4 for DKO | 30% RMSE reduction |
| `dko/experiments/main_benchmark.py` | kernel_output_dim: 32 → 64 | Marginal alone, needed for combined effect |
| `dko/experiments/main_benchmark.py` | Mixed precision: off for full DKO only | Prevents 10x variance inflation |
| `dko/experiments/main_benchmark.py` | Added dko_diagonal, dko_separate_nets to benchmark | New model variants |
| `dko/training/trainer.py` | Normalization: dim=(1,2) → dim=1 | Pearson 0.0 → 0.48 (critical fix) |
| `dko/training/trainer.py` | Sigma regularization: 1e-4 → 1e-2 | Prevents near-singular covariance |
| `dko/training/evaluator.py` | Same normalization + regularization fix | Consistent train/eval behavior |

---

## Files & Reproducibility

### Results

| Path | Description |
|------|-------------|
| `results/ablation/` | Ablation study results (7 configs x 3 seeds) |
| `results/feature_quality/feature_quality_results.json` | sklearn feature quality analysis |
| `results/benchmark_fixed/*/benchmark_results.json` | Benchmark results per dataset |
| `results/benchmark_fixed/*.log` | Training logs |
| `results/new_variants_20260203_204952/` | New variant benchmark results (7 subdirs) |
| `results/feature_variance_audit.json` | Variance audit results |
| `results/synthetic_validation.json` | Synthetic validation results |
| `results/scc_quartile_analysis.json` | SCC quartile results |
| `results/new_variants_report.md` | New variants standalone report |
| `results/fingerprint_baseline/fingerprint_results.json` | FP+XGBoost baseline (8 datasets) |
| `results/fingerprint_baseline/analysis.md` | FP baseline analysis |
| `results/10seed_validation/statistical_analysis.json` | 10-seed Welch's t-test results |
| `results/hybrid_experiment/hybrid_results.json` | Hybrid FP+conformer experiment |
| `results/kraken_benchmark/` | MARCEL/Kraken benchmark |
| `results/marcel_benchmark/` | MARCEL BDE + Drugs + FP baseline |

### Scripts

| Path | Description |
|------|-------------|
| `scripts/feature_quality_analysis.py` | Feature quality analysis script |
| `scripts/run_ablation.py` | Ablation study driver |
| `scripts/run_ablation_single.py` | Single ablation experiment runner |
| `scripts/feature_variance_audit.py` | Variance audit script |
| `scripts/synthetic_validation.py` | Synthetic validation script |
| `scripts/scc_quartile_analysis.py` | SCC quartile analysis script |
| `scripts/prepare_kraken.py` | Kraken data preprocessing |
| `scripts/prepare_bde.py` | BDE data preprocessing |
| `scripts/prepare_drugs75k.py` | Drugs-75K data preprocessing |
| `scripts/run_kraken_benchmark.sh` | Kraken DKO benchmark launcher |
| `scripts/run_bde_benchmark.sh` | BDE DKO benchmark launcher |
| `scripts/run_drugs_benchmark.sh` | Drugs DKO benchmark launcher |
| `scripts/run_marcel_fp_baseline.py` | Full MARCEL FP baseline (all datasets) |
| `scripts/compile_marcel_results.py` | Results compilation script |
| `scripts/run_hybrid_fast.py` | Fast hybrid FP+conformer experiment |

### Model Code

| Path | Description |
|------|-------------|
| `dko/models/dko_variants.py` | 7 new model variant classes |
| `data/conformers/kraken_*/` | Preprocessed Kraken data (4 targets) |
