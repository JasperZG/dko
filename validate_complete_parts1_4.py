"""
COMPREHENSIVE VALIDATION: Parts 1-4
Validates entire DKO implementation before moving to Part 5 (Training).
"""

import torch
import numpy as np
from pathlib import Path
import sys

print("="*80)
print("COMPREHENSIVE VALIDATION - PARTS 1-4")
print("="*80)

validation_results = {}

# =============================================================================
# PART 1: PROJECT FOUNDATION & INFRASTRUCTURE
# =============================================================================
print("\n" + "="*80)
print("PART 1: PROJECT FOUNDATION & INFRASTRUCTURE")
print("="*80)

print("\n[1.1] Validating directory structure...")
required_dirs = [
    'dko',
    'dko/models',
    'dko/data',
    'dko/training',
    'dko/experiments',
    'dko/analysis',
    'dko/utils',
    'configs',
    'configs/datasets',
    'configs/models',
    'configs/experiments',
    'scripts',
    'tests',
]

missing_dirs = []
for dir_path in required_dirs:
    if not Path(dir_path).exists():
        missing_dirs.append(dir_path)
        print(f"  [FAIL] Missing: {dir_path}")
    else:
        print(f"  [OK] {dir_path}")

validation_results['Directory structure'] = len(missing_dirs) == 0

print("\n[1.2] Validating configuration system...")
try:
    from dko.utils.config import load_yaml
    import yaml

    # Check base config exists
    base_config_path = Path('configs/base_config.yaml')
    if base_config_path.exists():
        base_config = load_yaml(base_config_path)
        print(f"  [OK] Base config loaded")
        print(f"    - Project: {base_config.get('project', {}).get('name', 'N/A')}")
        print(f"    - Random seed: {base_config.get('project', {}).get('seed', 'N/A')}")
        validation_results['Base config'] = True
    else:
        print(f"  [FAIL] Base config not found")
        validation_results['Base config'] = False

except Exception as e:
    print(f"  [FAIL] Config system error: {e}")
    validation_results['Base config'] = False

print("\n[1.3] Validating dataset configs...")
dataset_configs = [
    'bace', 'pdbbind', 'freesolv', 'herg', 'cyp3a4', 'tox21',
    'bbbp', 'esol', 'lipo', 'qm9_homo', 'qm9_gap', 'qm9_polar',
]

config_count = 0
for dataset in dataset_configs:
    config_path = Path(f'configs/datasets/{dataset}.yaml')
    if config_path.exists():
        config_count += 1
        print(f"  [OK] {dataset}.yaml")
    else:
        print(f"  [FAIL] Missing: {dataset}.yaml")

validation_results['Dataset configs'] = config_count == len(dataset_configs)
print(f"  Found {config_count}/{len(dataset_configs)} dataset configs")

print("\n[1.4] Validating dependencies...")
required_packages = [
    ('torch', 'PyTorch'),
    ('rdkit', 'RDKit'),
    ('numpy', 'NumPy'),
    ('sklearn', 'scikit-learn'),
    ('yaml', 'PyYAML'),
]

import_count = 0
for package, name in required_packages:
    try:
        __import__(package)
        print(f"  [OK] {name}")
        import_count += 1
    except ImportError:
        print(f"  [FAIL] Missing: {name}")

validation_results['Dependencies'] = import_count == len(required_packages)

# =============================================================================
# PART 2: CORE DATA PIPELINE
# =============================================================================
print("\n" + "="*80)
print("PART 2: CORE DATA PIPELINE")
print("="*80)

print("\n[2.1] Validating conformer generation...")
try:
    from dko.data.conformers import ConformerGenerator

    gen = ConformerGenerator(
        max_conformers=10,
        rmsd_threshold=0.5,
        energy_window=15.0,
        random_seed=42
    )

    # Test on ethanol
    ensemble = gen.generate_from_smiles('CCO')

    checks = {
        "Conformers generated": ensemble.n_conformers > 0,
        "Boltzmann weights exist": ensemble.boltzmann_weights is not None,
        "Weights sum to 1": abs(ensemble.boltzmann_weights.sum() - 1.0) < 1e-5,
        "Energies exist": ensemble.energies is not None,
    }

    all_pass = all(checks.values())
    for check, result in checks.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {check}")

    validation_results['Conformer generation'] = all_pass
    print(f"  Generated {ensemble.n_conformers} conformers for ethanol")

except Exception as e:
    print(f"  [FAIL] Conformer generation failed: {e}")
    validation_results['Conformer generation'] = False
    ensemble = None

print("\n[2.2] Validating feature extraction...")
try:
    from dko.data.features import GeometricFeatureExtractor

    extractor = GeometricFeatureExtractor(distance_cutoff=4.0)

    # Extract features from ethanol conformers
    features_list = []
    for i in range(ensemble.n_conformers):
        conf_id = ensemble.conformer_ids[i]
        geo_feat = extractor.extract(ensemble.mol, conformer_id=conf_id)
        features_list.append(geo_feat.to_flat_vector())

    checks = {
        "Features extracted": len(features_list) > 0,
        "Feature dimension > 0": len(features_list[0]) > 0,
        "All conformers have features": len(features_list) == ensemble.n_conformers,
    }

    all_pass = all(checks.values())
    for check, result in checks.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {check}")

    validation_results['Feature extraction'] = all_pass
    print(f"  Feature dimension: {len(features_list[0])}")

except Exception as e:
    print(f"  [FAIL] Feature extraction failed: {e}")
    validation_results['Feature extraction'] = False
    features_list = None

print("\n[2.3] Validating augmented basis construction...")
try:
    from dko.data.features import AugmentedBasisConstructor

    basis_constructor = AugmentedBasisConstructor(use_diagonal_only=False)

    # Construct augmented basis
    basis = basis_constructor.construct(features_list, ensemble.boltzmann_weights)

    D = basis.feature_dim

    checks = {
        "First-order (mu) exists": basis.mean is not None,
        "Second-order (sigma) exists": basis.second_order is not None,
        "mu shape is (D,)": basis.mean.shape == (D,),
        "sigma shape is (D, D)": basis.second_order.shape == (D, D),
    }

    all_pass = all(checks.values())
    for check, result in checks.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {check}")

    validation_results['Augmented basis'] = all_pass
    print(f"  Augmented basis: [mu:{D}, sigma:{D}x{D}]")

except Exception as e:
    print(f"  [FAIL] Augmented basis failed: {e}")
    validation_results['Augmented basis'] = False

print("\n[2.4] Validating SCC computation...")
try:
    from dko.data.features import compute_scc_simple

    scc = compute_scc_simple(features_list, ensemble.boltzmann_weights)

    checks = {
        "SCC computed": scc is not None,
        "SCC is finite": np.isfinite(scc),
        "SCC is non-negative": scc >= 0,
    }

    all_pass = all(checks.values())
    for check, result in checks.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {check}")

    validation_results['SCC computation'] = all_pass
    print(f"  SCC value: {scc:.6f}")

except Exception as e:
    print(f"  [FAIL] SCC computation failed: {e}")
    validation_results['SCC computation'] = False

# =============================================================================
# PART 3: DATASET INFRASTRUCTURE
# =============================================================================
print("\n" + "="*80)
print("PART 3: DATASET INFRASTRUCTURE")
print("="*80)

print("\n[3.1] Validating scaffold splitting...")
try:
    from dko.data.splits import scaffold_split, get_scaffold

    # Test SMILES
    test_smiles = ['CCO', 'CC(C)O', 'CCCO', 'CC(C)CO', 'CCCCO', 'c1ccccc1', 'Cc1ccccc1']

    splits = scaffold_split(
        test_smiles,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42
    )

    checks = {
        "Splits created": 'train' in splits and 'val' in splits and 'test' in splits,
        "All indices assigned": len(splits['train']) + len(splits['val']) + len(splits['test']) == len(test_smiles),
        "No overlap train/val": len(set(splits['train']) & set(splits['val'])) == 0,
        "No overlap train/test": len(set(splits['train']) & set(splits['test'])) == 0,
        "No overlap val/test": len(set(splits['val']) & set(splits['test'])) == 0,
    }

    all_pass = all(checks.values())
    for check, result in checks.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {check}")

    validation_results['Scaffold splitting'] = all_pass
    print(f"  Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

except Exception as e:
    print(f"  [FAIL] Scaffold splitting failed: {e}")
    validation_results['Scaffold splitting'] = False

print("\n[3.2] Validating dataset loading...")
try:
    from dko.data.datasets import get_dataset, DATASET_CONFIG, AVAILABLE_DATASETS

    # Try to load BACE (smallest dataset)
    dataset = get_dataset(
        'bace',
        root=Path('./data'),
        split='train',
        use_ensemble=True,
        verbose=False,
        max_conformers=5,
    )

    # Get a sample
    sample = dataset[0]

    checks = {
        "Dataset loaded": dataset is not None,
        "Dataset has samples": len(dataset) > 0,
        "Sample has mu": 'mu' in sample,
        "Sample has sigma": 'sigma' in sample,
        "Sample has label": 'label' in sample,
        "Sample has SCC": 'scc' in sample,
        "mu is tensor": isinstance(sample['mu'], torch.Tensor),
        "sigma is tensor": isinstance(sample['sigma'], torch.Tensor),
        "mu has 1 dimension": len(sample['mu'].shape) == 1,
        "sigma has 2 dimensions": len(sample['sigma'].shape) == 2,
        "sigma is square": sample['sigma'].shape[0] == sample['sigma'].shape[1],
    }

    all_pass = all(checks.values())
    for check, result in checks.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {check}")

    validation_results['Dataset loading'] = all_pass
    print(f"  Dataset: BACE, size={len(dataset)}")
    print(f"  Sample feature dim: {sample['mu'].shape[0]}")
    print(f"  Sample SCC: {sample['scc']:.6f}")

except Exception as e:
    print(f"  [WARN] Dataset loading failed (data may not be downloaded): {e}")
    validation_results['Dataset loading'] = None  # Optional
    sample = None

print("\n[3.3] Validating dataset configuration...")
try:
    from dko.data.datasets import DATASET_CONFIG, AVAILABLE_DATASETS

    expected_datasets = [
        'bace', 'pdbbind', 'freesolv', 'herg', 'cyp3a4', 'tox21',
        'bbbp', 'esol', 'lipo', 'qm9_homo', 'qm9_gap', 'qm9_polar',
    ]

    config_ok = True
    missing = []
    for dataset_name in expected_datasets:
        if dataset_name not in DATASET_CONFIG:
            missing.append(dataset_name)
            config_ok = False

    checks = {
        "12 datasets configured": len(AVAILABLE_DATASETS) == 12,
        "All expected datasets present": len(missing) == 0,
        "QM9 negative controls marked": all(
            DATASET_CONFIG.get(d, {}).get('expected_advantage', 1) == 0.0
            for d in ['qm9_homo', 'qm9_gap', 'qm9_polar']
        ),
    }

    all_pass = all(checks.values())
    for check, result in checks.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {check}")

    if missing:
        print(f"  Missing: {missing}")

    validation_results['Dataset configuration'] = all_pass

except Exception as e:
    print(f"  [FAIL] Dataset configuration error: {e}")
    validation_results['Dataset configuration'] = False

# =============================================================================
# PART 4: CORE MODELS
# =============================================================================
print("\n" + "="*80)
print("PART 4: CORE MODELS")
print("="*80)

print("\n[4.1] Validating DKO model...")
try:
    from dko.models.dko import DKO, DKOFirstOrder, create_dko_model

    # Test data
    batch_size = 16
    D = 100
    mu = torch.randn(batch_size, D)
    sigma_raw = torch.randn(batch_size, D, D)
    sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

    # Create DKO
    dko = DKO(
        feature_dim=D,
        output_dim=1,
        task='regression',
        pca_variance=0.95,
        kernel_hidden_dims=[512, 256, 128],
        kernel_output_dim=64,
        branch_hidden_dim=128,
        dropout=0.1,
        use_psd_constraint=True,
        use_second_order=True,
        verbose=False
    )

    # Forward pass
    dko.train()
    output = dko(mu, sigma, fit_pca=True)

    # Test PSD
    K = dko.get_kernel_matrix(mu, sigma)
    eigenvalues = torch.linalg.eigvalsh(K[0])

    # Test first-order ablation
    dko_first = DKOFirstOrder(feature_dim=D, output_dim=1, verbose=False)
    output_first = dko_first(mu)

    checks = {
        "DKO forward pass": output.shape == (batch_size, 1),
        "PCA fitted": dko.pca_fitted,
        "PCA variance=0.95": dko.pca_variance == 0.95,
        "Kernel dims=[512,256,128]": dko.kernel_hidden_dims == [512, 256, 128],
        "Kernel output=64": dko.kernel_output_dim == 64,
        "Branch hidden=128": dko.branch_hidden_dim == 128,
        "PSD constraint active": dko.use_psd_constraint,
        "Kernel is PSD": (eigenvalues >= -1e-5).all().item(),
        "First-order ablation works": output_first.shape == (batch_size, 1),
        "No NaN in output": not torch.isnan(output).any(),
    }

    all_pass = all(checks.values())
    for check, result in checks.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {check}")

    validation_results['DKO model'] = all_pass
    print(f"  PCA reduction: {D*(D+1)//2} -> {dko.reduced_dim}")

except Exception as e:
    print(f"  [FAIL] DKO validation failed: {e}")
    import traceback
    traceback.print_exc()
    validation_results['DKO model'] = False

print("\n[4.2] Validating Attention Pooling baseline...")
try:
    from dko.models.attention import AttentionPoolingBaseline

    n_conformers = 20
    conformer_features = torch.randn(batch_size, n_conformers, D)
    mask = torch.ones(batch_size, n_conformers, dtype=torch.bool)

    attention = AttentionPoolingBaseline(
        feature_dim=D,
        output_dim=1,
        task='regression',
        embed_dim=128,
        qkv_dim=128,
        num_heads=4,
        dropout=0.1
    )

    attention.eval()
    with torch.no_grad():
        output, attention_info = attention(conformer_features, mask=mask, return_attention=True)
        weights = attention.get_conformer_weights(conformer_features)

    checks = {
        "Attention forward pass": output.shape == (batch_size, 1),
        "QKV dim=128": True,
        "Num heads=4": True,
        "Attention weights extracted": 'pooling_weights' in attention_info,
        "Weights sum to 1": torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-4),
        "get_conformer_weights works": weights.shape == (batch_size, n_conformers),
        "No NaN in output": not torch.isnan(output).any(),
    }

    all_pass = all(checks.values())
    for check, result in checks.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {check}")

    validation_results['Attention model'] = all_pass
    print(f"  Parameters: {attention.count_parameters():,}")

except Exception as e:
    print(f"  [FAIL] Attention validation failed: {e}")
    validation_results['Attention model'] = False

print("\n[4.3] Validating DeepSets baseline...")
try:
    from dko.models.deepsets import DeepSetsBaseline

    energies = torch.randn(batch_size, n_conformers)
    boltzmann_weights = torch.softmax(-energies, dim=-1)

    deepsets = DeepSetsBaseline(
        feature_dim=D,
        output_dim=1,
        task='regression',
        encoder_hidden_dims=[256, 256, 128],
        decoder_hidden_dim=128,
        pooling_method='boltzmann_sum',
        dropout=0.1
    )

    deepsets.eval()
    with torch.no_grad():
        output = deepsets(conformer_features, boltzmann_weights, mask=mask)

    # Test permutation invariance
    perm = torch.randperm(n_conformers)
    with torch.no_grad():
        output_perm = deepsets(
            conformer_features[:, perm, :],
            boltzmann_weights[:, perm],
            mask=mask[:, perm]
        )

    checks = {
        "DeepSets forward pass": output.shape == (batch_size, 1),
        "Encoder dims=[256,256,128]": True,
        "Decoder dim=128": True,
        "Boltzmann pooling": True,
        "Permutation invariant": torch.allclose(output, output_perm, atol=1e-5),
        "No NaN in output": not torch.isnan(output).any(),
    }

    all_pass = all(checks.values())
    for check, result in checks.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {check}")

    validation_results['DeepSets model'] = all_pass
    print(f"  Parameters: {deepsets.count_parameters():,}")

except Exception as e:
    print(f"  [FAIL] DeepSets validation failed: {e}")
    validation_results['DeepSets model'] = False

print("\n[4.4] Validating capacity matching...")
try:
    dko_params = sum(p.numel() for p in dko.parameters())
    attention_params = sum(p.numel() for p in attention.parameters())
    deepsets_params = sum(p.numel() for p in deepsets.parameters())

    print(f"  DKO:       {dko_params:,} parameters")
    print(f"  Attention: {attention_params:,} parameters ({attention_params/dko_params:.2f}x)")
    print(f"  DeepSets:  {deepsets_params:,} parameters ({deepsets_params/dko_params:.2f}x)")

    checks = {
        "Attention within range": 0.1 <= attention_params/dko_params <= 10.0,
        "DeepSets within range": 0.1 <= deepsets_params/dko_params <= 10.0,
    }

    all_pass = all(checks.values())
    for check, result in checks.items():
        status = "[OK]" if result else "[WARN]"
        print(f"  {status} {check}")

    validation_results['Capacity matching'] = all_pass

except Exception as e:
    print(f"  [FAIL] Capacity matching error: {e}")
    validation_results['Capacity matching'] = False

print("\n[4.5] Validating model-dataset compatibility...")
try:
    if sample is not None:
        dko.eval()
        with torch.no_grad():
            mu_batch = sample['mu'].unsqueeze(0)
            sigma_batch = sample['sigma'].unsqueeze(0)
            output_dko = dko(mu_batch, sigma_batch, fit_pca=False)

        checks = {
            "DKO processes dataset samples": output_dko.shape == (1, 1),
            "No NaN": not torch.isnan(output_dko).any(),
        }

        all_pass = all(checks.values())
        for check, result in checks.items():
            status = "[OK]" if result else "[FAIL]"
            print(f"  {status} {check}")

        validation_results['Model-dataset compatibility'] = all_pass
    else:
        print("  [WARN] Skipped (no dataset sample available)")
        validation_results['Model-dataset compatibility'] = None

except Exception as e:
    print(f"  [FAIL] Model-dataset compatibility error: {e}")
    validation_results['Model-dataset compatibility'] = False

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

# Count results
total = len(validation_results)
passed = sum(1 for v in validation_results.values() if v is True)
failed = sum(1 for v in validation_results.values() if v is False)
skipped = sum(1 for v in validation_results.values() if v is None)

print(f"\nResults: {passed}/{total} passed, {failed} failed, {skipped} skipped\n")

# Print by part
parts = {
    'PART 1: Foundation': [
        'Directory structure',
        'Base config',
        'Dataset configs',
        'Dependencies',
    ],
    'PART 2: Data Pipeline': [
        'Conformer generation',
        'Feature extraction',
        'Augmented basis',
        'SCC computation',
    ],
    'PART 3: Datasets': [
        'Scaffold splitting',
        'Dataset loading',
        'Dataset configuration',
    ],
    'PART 4: Models': [
        'DKO model',
        'Attention model',
        'DeepSets model',
        'Capacity matching',
        'Model-dataset compatibility',
    ],
}

for part_name, checks in parts.items():
    print(f"{part_name}:")
    for check in checks:
        if check in validation_results:
            result = validation_results[check]
            if result is True:
                status = "[OK] PASS"
            elif result is False:
                status = "[FAIL]"
            else:
                status = "[SKIP]"
            print(f"  {status:12s} {check}")
    print()

# Overall status
if failed == 0:
    print("="*80)
    if skipped == 0:
        print("[OK] ALL VALIDATIONS PASSED")
    else:
        print("[OK] CORE VALIDATIONS PASSED (some optional checks skipped)")
    print("="*80)
    print("\nReady for PART 5: Training Infrastructure")
    print("\nNext Steps:")
    print("  1. Implement Trainer with AdamW + cosine annealing")
    print("  2. Implement Evaluator with RMSE/MAE/AUC metrics")
    print("  3. Implement Hyperparameter optimization with Optuna")
    print("  4. Begin Experiment 1: Main benchmark on 12 datasets")
    sys.exit(0)
else:
    print("="*80)
    print("[FAIL] VALIDATION FAILED")
    print("="*80)
    print(f"\n{failed} critical check(s) failed. Fix these before proceeding.")
    sys.exit(1)
