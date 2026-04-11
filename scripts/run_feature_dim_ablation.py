#!/usr/bin/env python3
"""
Feature dimension ablation: sweep D = {256, 512, 1024, 2048}
on ESOL with hybrid FP+mu+sigma + XGBoost.
Addresses reviewer concern about D=1024 feature truncation justification.

Pipeline matches run_hybrid_fast.py / run_scaffold_splits.py exactly:
- mu truncated/padded to D dims (NO PCA)
- sigma: 5 scalar invariants
- XGBoost: n_estimators=100, lr=0.1, no early stopping
- Train on train+val combined, evaluate on test
"""

import json
import pickle
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).parent.parent))

DIMS = [256, 512, 1024, 2048]
SEEDS = [42, 123, 456]


def smiles_to_fingerprint(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def compute_conformer_stats(features, max_dim):
    """Compute mu (D-dim) and sigma (5 scalar invariants) — no PCA."""
    padded = []
    for conf_feat in features:
        cf = np.array(conf_feat, dtype=np.float64).flatten()
        if len(cf) < max_dim:
            cf = np.pad(cf, (0, max_dim - len(cf)))
        else:
            cf = cf[:max_dim]
        padded.append(cf)

    conformers = np.array(padded)
    if len(conformers) == 0:
        return np.zeros(max_dim), np.zeros(5)

    mu = np.mean(conformers, axis=0)  # (D,)

    if len(conformers) < 2:
        return mu, np.zeros(5)

    centered = conformers - mu
    variances = np.mean(centered ** 2, axis=0)

    total_var = np.sum(variances)
    max_var = np.max(variances)
    mean_var = np.mean(variances)
    sorted_var = np.sort(variances)[::-1]
    top5_var = np.sum(sorted_var[:5])
    effective_rank = float(np.sum(variances > 0.01 * total_var)) if total_var > 0 else 0

    sigma_stats = np.array([total_var, max_var, mean_var, top5_var, effective_rank])
    return mu, sigma_stats


def load_split(dataset_name, split, max_dim):
    """Load one split and compute FP + mu + sigma features."""
    data_path = Path(f"data/conformers/{dataset_name}/{split}.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    smiles_list = data["smiles"]
    labels = np.array([float(y) for y in data["labels"]])
    features_list = data["features"]

    n = len(smiles_list)
    fp_features = np.zeros((n, 2048), dtype=np.float32)
    mu_features = np.zeros((n, max_dim), dtype=np.float64)
    sigma_features = np.zeros((n, 5), dtype=np.float64)

    for i, (smi, mol_feats) in enumerate(zip(smiles_list, features_list)):
        fp_features[i] = smiles_to_fingerprint(smi)
        mu, sigma = compute_conformer_stats(mol_feats, max_dim)
        mu_features[i] = mu
        sigma_features[i] = sigma

    X = np.concatenate([fp_features, mu_features, sigma_features], axis=1)
    return X, labels


def run_ablation(dataset_name, dim, seed):
    """Run single ablation experiment matching run_hybrid_fast.py pipeline."""
    np.random.seed(seed)

    # Load splits
    X_train, y_train = load_split(dataset_name, "train", dim)
    X_val, y_val = load_split(dataset_name, "val", dim)
    X_test, y_test = load_split(dataset_name, "test", dim)

    # Combine train + val (same as run_hybrid_fast.py)
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    # XGBoost — exact settings from run_hybrid_fast.py / run_scaffold_splits.py
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=4,
        verbosity=0,
        tree_method='hist',
    )
    model.fit(X_full, y_full)

    preds = model.predict(X_test)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "n_features": X_full.shape[1],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="esol")
    parser.add_argument("--dims", nargs="+", type=int, default=DIMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output-dir", default="results/feature_dim_ablation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Feature Dimension Ablation (matching hybrid_fast.py pipeline)")
    print(f"Dataset: {args.dataset}")
    print(f"Dims: {args.dims}")
    print(f"Seeds: {args.seeds}")
    print("Pipeline: FP(2048) + mu(D) + sigma(5), XGB n=100 lr=0.1, train+val combined")
    print("=" * 60)

    results = []

    for dim in args.dims:
        print(f"\n=== D = {dim} ===")
        sys.stdout.flush()
        for seed in args.seeds:
            print(f"  Seed {seed}:", end=" ")
            sys.stdout.flush()
            try:
                metrics = run_ablation(args.dataset, dim, seed)
                results.append({
                    "dataset": args.dataset,
                    "max_feature_dim": dim,
                    "seed": seed,
                    **metrics,
                })
                print(f"RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")
                sys.stdout.flush()
                with open(output_dir / "feature_dim_results_partial.json", "w") as f:
                    json.dump({"results": results}, f, indent=2)
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for dim in args.dims:
        dim_results = [r for r in results if r["max_feature_dim"] == dim]
        if dim_results:
            rmses = [r["rmse"] for r in dim_results]
            print(f"D={dim:5d}: RMSE = {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")

    with open(output_dir / "feature_dim_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}/feature_dim_results.json")


if __name__ == "__main__":
    main()
