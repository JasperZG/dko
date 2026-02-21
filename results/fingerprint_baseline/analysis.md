# Fingerprint Baseline Analysis

**Date:** 2026-02-05
**Model:** Morgan Fingerprints (ECFP4, radius=2, 2048 bits) + XGBoost

## Summary

The fingerprint baseline **outperforms all conformer-based models** on every dataset tested.

## Results Comparison

| Dataset | FP+XGBoost | Best Conformer Model | Δ RMSE | Winner |
|---------|------------|---------------------|--------|--------|
| ESOL | **1.506** ± 0.021 | dko_gated: 1.635 ± 0.023 | -0.129 | FP |
| Lipophilicity | **0.910** ± 0.006 | dko_invariants: 1.131 ± 0.008 | -0.221 | FP |
| QM9-Gap | **0.0203** ± 0.0001 | attention: 0.0364 ± 0.0007 | -0.0161 | FP |
| QM9-LUMO | **0.0187** ± 0.0001 | mean_ensemble: 0.0335 ± 0.001 | -0.0148 | FP |
| FreeSolv | **2.939** ± 0.127 | attention: 4.077 | -1.138 | FP |
| QM9-HOMO | **0.0142** ± 0.00002 | attention: 0.019 | -0.005 | FP |
| BACE | **0.274** ± 0.003 | mean_ensemble: 0.871 | -0.597 | FP |
| BBBP | **0.207** ± 0.003 | dko: 1.000 | -0.793 | FP |

## Implications

1. **Conformer features don't add value over fingerprints** for these datasets
2. The 1024-dim conformer features from pretrained models are less informative than 2048-bit Morgan fingerprints
3. This questions the premise of the entire DKO line of research

## Possible Explanations

1. **Feature quality**: The pretrained conformer features may not capture task-relevant information
2. **Dataset characteristics**: These benchmarks may not require 3D conformer information
3. **Model capacity**: XGBoost with 100 trees may have more effective capacity than shallow MLPs
4. **Overfitting**: Conformer models with 200 epochs may overfit small datasets

## Recommendations

1. **Investigate feature quality**: Compare raw conformer features vs fingerprints on held-out tasks
2. **Use MARCEL benchmark**: These datasets explicitly require conformer geometry
3. **Try hybrid approach**: Concatenate fingerprints + conformer features
4. **Increase training data**: Test on larger datasets (full QM9, ChEMBL)

## Raw Data

See `fingerprint_results.json` for full per-seed results.
