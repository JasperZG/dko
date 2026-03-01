#!/usr/bin/env python3
"""XGBoost hyperparameter tuning with Optuna.

Addresses: "Did you tune XGBoost too? The comparison to neural models is unfair."
Also compares default vs tuned XGBoost for both FP-only and hybrid features.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import mean_squared_error, r2_score

# ── Config ──
DATASETS = ["esol", "freesolv", "lipophilicity"]
N_TRIALS = 50
SEEDS = [42, 123, 456]
OUTPUT_DIR = Path("results/xgb_tuning")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048)
    return _FP_GEN.GetFingerprintAsNumPy(mol).astype(np.float64)


def compute_conformer_stats(features, max_dim=256):
    padded = []
    for conf_feat in features:
        cf = np.array(conf_feat).flatten()
        if len(cf) < max_dim:
            cf = np.pad(cf, (0, max_dim - len(cf)))
        else:
            cf = cf[:max_dim]
        padded.append(cf)
    conformers = np.array(padded)
    if len(conformers) == 0:
        return np.zeros(max_dim), np.zeros(5)
    mu = np.mean(conformers, axis=0)
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
    return mu, np.array([total_var, max_var, mean_var, top5_var, effective_rank])


def load_split(dataset_name, split):
    path = Path(f"data/conformers/{dataset_name}/{split}.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["smiles"], np.array([float(y) for y in data["labels"]]), data["features"]


def precompute_all(smiles_list, features_list):
    n = len(smiles_list)
    fp = np.zeros((n, 2048))
    mu = np.zeros((n, 256))
    sigma = np.zeros((n, 5))
    for i, (smi, feats) in enumerate(zip(smiles_list, features_list)):
        fp[i] = smiles_to_fingerprint(smi)
        mu[i], sigma[i] = compute_conformer_stats(feats, max_dim=256)
    return fp, mu, sigma


def tune_xgboost(train_X, train_y, val_X, val_y, n_trials=50):
    """Tune XGBoost with Optuna."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": 4,
            "verbosity": 0,
            "tree_method": "hist",
        }
        model = xgb.XGBRegressor(**params)
        model.fit(train_X, train_y)
        pred = model.predict(val_X)
        return float(np.sqrt(mean_squared_error(val_y, pred)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def run_xgb_with_params(train_X, train_y, test_X, test_y, params, seed):
    """Run XGBoost with specific parameters."""
    full_params = {
        "n_jobs": 4, "verbosity": 0, "tree_method": "hist",
        "random_state": seed,
    }
    full_params.update(params)
    model = xgb.XGBRegressor(**full_params)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    rmse = float(np.sqrt(mean_squared_error(test_y, pred)))
    r2 = float(r2_score(test_y, pred))
    return rmse, r2


def main():
    print("XGBoost Hyperparameter Tuning")
    print("=" * 60)

    DEFAULT_PARAMS = {
        "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8,
    }

    all_results = []

    for dataset in DATASETS:
        print(f"\n=== {dataset.upper()} ===")

        # Load data
        train_smi, train_y, train_feats = load_split(dataset, "train")
        val_smi, val_y, val_feats = load_split(dataset, "val")
        test_smi, test_y, test_feats = load_split(dataset, "test")

        # Precompute
        tr_fp, tr_mu, tr_sigma = precompute_all(train_smi, train_feats)
        va_fp, va_mu, va_sigma = precompute_all(val_smi, val_feats)
        te_fp, te_mu, te_sigma = precompute_all(test_smi, test_feats)

        feature_configs = {
            "FP-only": {
                "train": tr_fp, "val": va_fp, "test": te_fp,
                "trainval": np.vstack([tr_fp, va_fp]),
            },
            "FP+Mu+Sigma": {
                "train": np.hstack([tr_fp, tr_mu, tr_sigma]),
                "val": np.hstack([va_fp, va_mu, va_sigma]),
                "test": np.hstack([te_fp, te_mu, te_sigma]),
                "trainval": np.hstack([np.vstack([tr_fp, va_fp]),
                                       np.vstack([tr_mu, va_mu]),
                                       np.vstack([tr_sigma, va_sigma])]),
            },
        }
        trainval_y = np.concatenate([train_y, val_y])

        for config_name, feats in feature_configs.items():
            print(f"\n  {config_name}:")

            # Tune on train→val
            best_params, best_val_rmse = tune_xgboost(
                feats["train"], train_y, feats["val"], val_y, n_trials=N_TRIALS
            )
            print(f"    Best val RMSE: {best_val_rmse:.4f}")
            print(f"    Best params: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in best_params.items()})}")

            # Evaluate on test with tuned and default params, using trainval
            for seed in SEEDS:
                # Default
                rmse_def, r2_def = run_xgb_with_params(
                    feats["trainval"], trainval_y, feats["test"], test_y,
                    DEFAULT_PARAMS, seed
                )
                all_results.append({
                    "dataset": dataset, "features": config_name,
                    "tuning": "default", "seed": seed,
                    "rmse": rmse_def, "r2": r2_def,
                })

                # Tuned
                rmse_tuned, r2_tuned = run_xgb_with_params(
                    feats["trainval"], trainval_y, feats["test"], test_y,
                    best_params, seed
                )
                all_results.append({
                    "dataset": dataset, "features": config_name,
                    "tuning": "optuna_50", "seed": seed,
                    "rmse": rmse_tuned, "r2": r2_tuned,
                    "best_params": best_params,
                })

            # Summary
            default_rmses = [r["rmse"] for r in all_results
                             if r["dataset"] == dataset and r["features"] == config_name
                             and r["tuning"] == "default"]
            tuned_rmses = [r["rmse"] for r in all_results
                           if r["dataset"] == dataset and r["features"] == config_name
                           and r["tuning"] == "optuna_50"]
            imp = (np.mean(default_rmses) - np.mean(tuned_rmses)) / np.mean(default_rmses) * 100
            print(f"    Default test RMSE: {np.mean(default_rmses):.4f}±{np.std(default_rmses):.4f}")
            print(f"    Tuned test RMSE:   {np.mean(tuned_rmses):.4f}±{np.std(tuned_rmses):.4f}")
            print(f"    Improvement: {imp:+.1f}%")

    # Global summary
    print("\n\n" + "=" * 60)
    print("SUMMARY: Default vs Tuned XGBoost")
    print("=" * 60)
    for dataset in DATASETS:
        print(f"\n{dataset.upper()}:")
        for config_name in ["FP-only", "FP+Mu+Sigma"]:
            def_rmses = [r["rmse"] for r in all_results
                         if r["dataset"] == dataset and r["features"] == config_name
                         and r["tuning"] == "default"]
            tun_rmses = [r["rmse"] for r in all_results
                         if r["dataset"] == dataset and r["features"] == config_name
                         and r["tuning"] == "optuna_50"]
            if def_rmses and tun_rmses:
                imp = (np.mean(def_rmses) - np.mean(tun_rmses)) / np.mean(def_rmses) * 100
                print(f"  {config_name:15s}: Default={np.mean(def_rmses):.4f}, "
                      f"Tuned={np.mean(tun_rmses):.4f}, Δ={imp:+.1f}%")

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"datasets": DATASETS, "n_trials": N_TRIALS, "seeds": SEEDS},
        "raw_results": all_results,
    }
    with open(OUTPUT_DIR / "xgb_tuning_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
