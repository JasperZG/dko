# DKO Error Report - Issues to Fix

**Date**: 2026-01-26
**Status**: Training pipeline partially working, needs fixes

## Summary

The DKO benchmark pipeline has been debugged to a partially working state. The `DKOFirstOrder` model (first-order features only) trains successfully, but the full `DKO` model (with second-order covariance features) still produces NaN losses.

---

## Working Components

1. **Dataset Preparation**: All 8 datasets prepared with precomputed conformers
   - esol, freesolv, lipophilicity (regression)
   - bace, bbbp (classification)
   - qm9_homo, qm9_lumo, qm9_gap (regression)

2. **DKOFirstOrder Model**: Trains successfully with reasonable metrics
   - Uses only first-order (mean) features
   - Training loss converges (~0.7-1.0 on regression tasks)

3. **Baseline Models**: Should work (not fully tested)
   - mean_ensemble, single_conformer, deepsets, attention

---

## Critical Issues

### Issue 1: Full DKO Model NaN Losses

**File**: `dko/models/dko.py`

**Symptom**: Training produces NaN losses after a few batches

**Root Cause**: Numerical instability in second-order (covariance) feature processing

**Details**:
- The covariance matrices (sigma) are extracted from conformer ensembles
- PCA is applied to reduce dimensionality: D*(D+1)/2 -> ~18-50 components
- After PCA + kernel network + PSD constraint (K = LL^T), values explode

**Attempted Fixes** (in dko.py):
1. Added sigma normalization before PCA (line ~200)
2. Added mu normalization in forward pass (line ~410)
3. Added kernel_features normalization after PSD constraint (line ~450)

**Still Needed**:
- Debug why normalization isn't preventing NaN
- Check if PCA transform produces extreme values
- Consider using batch normalization in kernel network output
- May need to clamp values or use different architecture

---

### Issue 2: PyTorch 2.0.1 Compatibility

**Files**: `dko/training/trainer.py`

**Symptom**: Import errors and API differences

**Fixes Applied**:
```python
# Handle different PyTorch versions for mixed precision
try:
    from torch.amp import GradScaler, autocast
    AUTOCAST_DEVICE_ARG = True  # torch.amp.autocast takes device_type arg
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    AUTOCAST_DEVICE_ARG = False  # torch.cuda.amp.autocast doesn't

# GradScaler initialization differs between versions
if AUTOCAST_DEVICE_ARG:
    scaler = GradScaler('cuda')
else:
    scaler = GradScaler()  # No device arg in 2.0.x
```

---

### Issue 3: DKO Model Signature Mismatch

**Files**: `dko/training/trainer.py`, `dko/training/evaluator.py`

**Symptom**: DKO models expect `(mu, sigma)` but trainer passes `(features, mask)`

**Fix Applied**: Added `_is_dko_model()` check and `_compute_mu_sigma()` function to compute distribution statistics from conformer features:

```python
def _is_dko_model(self, model):
    model_class_name = model.__class__.__name__
    return model_class_name in ['DKO', 'DKOFirstOrder', 'DKOFull', 'DKONoPSD']

def _compute_mu_sigma(self, features, mask=None):
    # Normalize features
    feat_mean = features.mean()
    feat_std = features.std().clamp(min=1e-6)
    features = (features - feat_mean) / feat_std

    # Compute weighted mean (mu)
    # Compute weighted covariance (sigma)
    ...
```

---

### Issue 4: Evaluator Device Parameter

**File**: `dko/experiments/main_benchmark.py`

**Symptom**: `device` was passed to evaluator.evaluate() as positional arg

**Fix Applied**:
```python
# Before (wrong):
evaluator = Evaluator(task_type=task_type)
test_metrics = evaluator.evaluate(model, test_loader, device)

# After (correct):
evaluator = Evaluator(task_type=task_type, device=device)
test_metrics = evaluator.evaluate(model, test_loader)
```

---

### Issue 5: Feature Padding

**File**: `dko/data/features.py`

**Symptom**: "all input arrays must have the same shape" error

**Fix Applied**: Added padding in `AugmentedBasisConstructor.construct()`:
```python
# Pad features to same length
max_len = max(len(f) for f in features_list)
padded_features = []
for f in features_list:
    if len(f) < max_len:
        padded = np.pad(f, (0, max_len - len(f)), mode='constant')
        padded_features.append(padded)
    else:
        padded_features.append(f)
```

---

## Configuration Notes

### DKO-Specific Settings (main_benchmark.py)

```python
is_dko = model_name in ["dko", "dko_first_order"]
if is_dko:
    # Reduced from 64 to prevent gradient explosion
    model_config["kernel_output_dim"] = 32

# Lower learning rate for DKO (large gradients)
base_lr = 1e-5 if is_dko else 1e-4
```

### Classification Datasets

```python
task_type = "classification" if dataset_name in [
    "bace", "herg", "cyp3a4", "tox21", "bbbp"
] else "regression"
```

---

## Testing Commands

### Working (DKOFirstOrder):
```bash
CUDA_VISIBLE_DEVICES=1 python -m dko.experiments.main_benchmark \
    --datasets esol freesolv lipophilicity \
    --models dko_first_order mean_ensemble single_conformer \
    --seeds 42 123 456 \
    --output-dir results/benchmark
```

### Broken (Full DKO):
```bash
# This produces NaN losses - needs debugging
CUDA_VISIBLE_DEVICES=1 python -m dko.experiments.main_benchmark \
    --datasets esol \
    --models dko \
    --seeds 42
```

---

## Priority Fixes Needed

1. **HIGH**: Fix full DKO model NaN issue
   - Debug PCA transform output ranges
   - Add gradient clipping specifically for sigma branch
   - Consider alternative architectures for second-order features

2. **MEDIUM**: Add comprehensive logging
   - Log files are created but empty (logging not properly configured)
   - Add intermediate value logging for debugging

3. **LOW**: Multi-GPU support for DKO
   - Currently disabled due to PCA fitting issues with DataParallel
   - Need to fit PCA before wrapping model

---

## Environment

- Python 3.10
- PyTorch 2.0.1+cu118
- CUDA 11.8
- Conda environment: `nest`
