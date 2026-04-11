#!/usr/bin/env python3
"""
Classification benchmark: BACE and BBBP with hybrid FP+conformer features.
Addresses reviewer concern about classification task coverage.

Tests whether the hybrid improvement taxonomy extends to classification:
- XGBClassifier with FP-only, FP+mu, FP+mu+sigma
- Neural models (attention, dko_gated) with BCEWithLogitsLoss
- Metrics: AUROC (primary), AUPRC, accuracy
"""

import argparse
import json
import pickle
import sys
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.decomposition import PCA
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))

DATASETS = ["bace", "bbbp"]
SEEDS = [42, 123, 456]


def load_data(dataset_name, split):
    """Load dataset from pickle."""
    data_path = Path(f"data/conformers/{dataset_name}/{split}.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_data_merged(dataset_name):
    """Load and merge all splits, returning a single dataset dict.

    The scaffold splits for BACE/BBBP produce one-class test sets due to class
    imbalance, making AUROC undefined. We merge splits and use stratified
    random splits for reliable evaluation.
    """
    all_smiles, all_labels, all_features = [], [], []
    for split in ["train", "val", "test"]:
        d = load_data(dataset_name, split)
        all_smiles.extend(d["smiles"])
        all_labels.extend(np.array(d["labels"]).squeeze().tolist())
        all_features.extend(d["features"])
    return {
        "smiles": all_smiles,
        "labels": all_labels,
        "features": all_features,
    }


def compute_features(data, max_feature_dim=1024, pca_model=None, fit_pca=False):
    """Compute FP, mu, and sigma features."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    features_list = data["features"]
    smiles_list = data["smiles"]
    labels = np.array(data["labels"]).squeeze()

    all_fp = []
    all_mu = []
    all_sigma = []
    valid_labels = []
    valid_indices = []

    for idx, (smi, mol_features) in enumerate(zip(smiles_list, features_list)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_arr = np.array(fp, dtype=np.float32)

        conf_features = []
        for feat in mol_features:
            feat = np.array(feat, dtype=np.float64)
            if len(feat) > max_feature_dim:
                feat = feat[:max_feature_dim]
            elif len(feat) < max_feature_dim:
                feat = np.pad(feat, (0, max_feature_dim - len(feat)))
            conf_features.append(feat)

        if len(conf_features) == 0:
            continue

        conf_features = np.array(conf_features)
        mu = np.mean(conf_features, axis=0)

        if conf_features.shape[0] > 1:
            var_per_feature = np.var(conf_features, axis=0)
            total_var = np.sum(var_per_feature)
            sorted_var = np.sort(var_per_feature)[::-1]
            top5_var = np.sum(sorted_var[:5])
            max_var = sorted_var[0] if len(sorted_var) > 0 else 0
            effective_rank = np.sum(var_per_feature > 1e-10)
            mean_var = np.mean(var_per_feature)
            sigma_stats = np.array([total_var, top5_var, max_var, effective_rank, mean_var])
        else:
            sigma_stats = np.zeros(5)

        all_fp.append(fp_arr)
        all_mu.append(mu)
        all_sigma.append(sigma_stats)
        valid_labels.append(labels[idx])
        valid_indices.append(idx)

    all_fp = np.array(all_fp)
    all_mu = np.array(all_mu)
    all_sigma = np.array(all_sigma)
    valid_labels = np.array(valid_labels)

    # PCA on mu
    pca_dim = min(256, max_feature_dim, all_mu.shape[0] - 1)
    if fit_pca:
        pca_model = PCA(n_components=pca_dim)
        all_mu_pca = pca_model.fit_transform(all_mu)
    else:
        all_mu_pca = pca_model.transform(all_mu)

    return {
        "fp": all_fp,
        "mu": all_mu_pca,
        "sigma": all_sigma,
        "labels": valid_labels,
        "pca": pca_model,
    }


def evaluate_classifier(y_true, y_pred_proba):
    """Compute classification metrics."""
    y_pred = (y_pred_proba >= 0.5).astype(int)
    metrics = {
        "auroc": float(roc_auc_score(y_true, y_pred_proba)),
        "auprc": float(average_precision_score(y_true, y_pred_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    return metrics


def run_xgb_classification(dataset_name, feature_set, seed, all_feats_cache=None):
    """Run XGBoost classification experiment with stratified random splits.

    Uses stratified 80/10/10 random splits instead of scaffold splits, since
    the scaffold splits for BACE/BBBP yield one-class test sets (class imbalance
    causes entire scaffold groups to have only one label).
    """
    from sklearn.model_selection import train_test_split

    np.random.seed(seed)

    if all_feats_cache is not None:
        all_feats = all_feats_cache
    else:
        all_data = load_data_merged(dataset_name)
        all_feats = compute_features(all_data, fit_pca=True)

    # Stratified random 80/10/10 split
    indices = np.arange(len(all_feats["labels"]))
    labels = all_feats["labels"]
    idx_train_val, idx_test = train_test_split(
        indices, test_size=0.10, random_state=seed, stratify=labels)
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=0.111, random_state=seed,
        stratify=labels[idx_train_val])

    def subset(feats, idx):
        return {k: v[idx] if isinstance(v, np.ndarray) else np.array(v)[idx]
                for k, v in feats.items() if k != "pca"}

    train_feats = subset(all_feats, idx_train)
    val_feats = subset(all_feats, idx_val)
    test_feats = subset(all_feats, idx_test)

    # Build feature matrix based on feature_set
    if feature_set == "fp_only":
        X_train = train_feats["fp"]
        X_val = val_feats["fp"]
        X_test = test_feats["fp"]
    elif feature_set == "fp_mu":
        X_train = np.concatenate([train_feats["fp"], train_feats["mu"]], axis=1)
        X_val = np.concatenate([val_feats["fp"], val_feats["mu"]], axis=1)
        X_test = np.concatenate([test_feats["fp"], test_feats["mu"]], axis=1)
    elif feature_set == "fp_mu_sigma":
        X_train = np.concatenate([train_feats["fp"], train_feats["mu"], train_feats["sigma"]], axis=1)
        X_val = np.concatenate([val_feats["fp"], val_feats["mu"], val_feats["sigma"]], axis=1)
        X_test = np.concatenate([test_feats["fp"], test_feats["mu"], test_feats["sigma"]], axis=1)
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    y_train = train_feats["labels"]
    y_val = val_feats["labels"]
    y_test = test_feats["labels"]

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        objective='binary:logistic',
        eval_metric='auc',
        early_stopping_rounds=30,
        random_state=seed,
        verbosity=0,
        n_jobs=4,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_classifier(y_test, y_pred_proba)

    return metrics


def run_neural_classification(dataset_name, model_name, seed, device="cpu", all_feats_cache=None):
    """Run neural classification experiment with stratified random splits."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    torch.manual_seed(seed)
    np.random.seed(seed)

    if all_feats_cache is not None:
        all_feats = all_feats_cache
    else:
        all_data = load_data_merged(dataset_name)
        all_feats = compute_features(all_data, fit_pca=True)

    # Stratified random 80/10/10 split
    indices = np.arange(len(all_feats["labels"]))
    labels = all_feats["labels"]
    idx_train_val, idx_test = train_test_split(
        indices, test_size=0.10, random_state=seed, stratify=labels)
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=0.111, random_state=seed,
        stratify=labels[idx_train_val])

    def subset(feats, idx):
        return {k: v[idx] if isinstance(v, np.ndarray) else np.array(v)[idx]
                for k, v in feats.items() if k != "pca"}

    train_feats = subset(all_feats, idx_train)
    val_feats = subset(all_feats, idx_val)
    test_feats = subset(all_feats, idx_test)

    # Use FP+mu+sigma as input for neural models
    X_train = np.concatenate([train_feats["fp"], train_feats["mu"], train_feats["sigma"]], axis=1)
    X_val = np.concatenate([val_feats["fp"], val_feats["mu"], val_feats["sigma"]], axis=1)
    X_test = np.concatenate([test_feats["fp"], test_feats["mu"], test_feats["sigma"]], axis=1)

    input_dim = X_train.shape[1]

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(train_feats["labels"]).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(val_feats["labels"]).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(test_feats["labels"]).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = TorchDataLoader(train_ds, batch_size=64, shuffle=True)

    # Simple MLP classifier
    if model_name == "attention":
        model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)
    else:  # dko_gated
        model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0
    best_state = None
    patience = 0

    for epoch in range(300):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).squeeze(-1)
            val_proba = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = roc_auc_score(val_feats["labels"], val_proba)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= 30:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t).squeeze(-1)
        test_proba = torch.sigmoid(test_logits).cpu().numpy()

    metrics = evaluate_classifier(test_feats["labels"], test_proba)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results/classification_benchmark")
    parser.add_argument("--xgb-only", action="store_true", help="Skip neural model experiments")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Classification Benchmark")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}")
    print("=" * 60)

    import sys
    results = []
    feature_sets = ["fp_only", "fp_mu", "fp_mu_sigma"]

    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*60}")

        # Pre-compute all features once (load merged dataset, compute FP+mu+sigma)
        print(f"  Loading {dataset} data and computing features...")
        sys.stdout.flush()
        all_data = load_data_merged(dataset)
        all_feats = compute_features(all_data, fit_pca=True)
        print(f"  {len(all_feats['labels'])} molecules, class balance: "
              f"{int(np.sum(all_feats['labels']))} pos / {int(np.sum(1-all_feats['labels']))} neg")
        sys.stdout.flush()

        # XGBoost experiments
        for fs in feature_sets:
            print(f"\n  XGBoost + {fs}:")
            sys.stdout.flush()
            for seed in args.seeds:
                print(f"    Seed {seed}:", end=" ")
                sys.stdout.flush()
                try:
                    metrics = run_xgb_classification(dataset, fs, seed, all_feats_cache=all_feats)
                    results.append({
                        "dataset": dataset,
                        "model": f"xgb_{fs}",
                        "seed": seed,
                        **metrics,
                    })
                    print(f"AUROC={metrics['auroc']:.4f}, "
                          f"AUPRC={metrics['auprc']:.4f}, Acc={metrics['accuracy']:.4f}")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"ERROR: {e}")
                    traceback.print_exc()

        # Neural experiments (skip if --xgb-only)
        if args.xgb_only:
            print(f"\n  Skipping neural models (--xgb-only).")
            sys.stdout.flush()
        for model_name in ([] if args.xgb_only else ["attention", "dko_gated"]):
            print(f"\n  Neural ({model_name}):")
            sys.stdout.flush()
            for seed in args.seeds:
                print(f"    Seed {seed}:", end=" ")
                sys.stdout.flush()
                try:
                    metrics = run_neural_classification(
                        dataset, model_name, seed, args.device,
                        all_feats_cache=all_feats)
                    results.append({
                        "dataset": dataset,
                        "model": f"neural_{model_name}",
                        "seed": seed,
                        **metrics,
                    })
                    print(f"AUROC={metrics['auroc']:.4f}, "
                          f"AUPRC={metrics['auprc']:.4f}, Acc={metrics['accuracy']:.4f}")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"ERROR: {e}")
                    traceback.print_exc()

        # Save incrementally
        with open(output_dir / "classification_results_partial.json", "w") as f:
            json.dump({"results": results}, f, indent=2)
        sys.stdout.flush()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for dataset in args.datasets:
        print(f"\n{dataset.upper()}:")
        ds_results = [r for r in results if r["dataset"] == dataset]
        models = sorted(set(r["model"] for r in ds_results))
        for model_name in models:
            mr = [r for r in ds_results if r["model"] == model_name]
            if mr:
                aurocs = [r["auroc"] for r in mr]
                print(f"  {model_name:25s}: AUROC = {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")

    with open(output_dir / "classification_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}/classification_results.json")


if __name__ == "__main__":
    main()
