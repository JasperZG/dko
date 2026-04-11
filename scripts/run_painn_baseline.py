#!/usr/bin/env python3
"""
PaiNN baseline for 3D molecular property prediction.
Addresses Reviewer tcvY Critique: additional SOTA 3D GNN baselines.

PaiNN (Polarizable Atom Interaction Neural Network) uses equivariant
message passing with scalar and vector features.

Reference: Schütt et al., "Equivariant message passing for the prediction
of tensorial properties and molecular spectra", ICML 2021.
"""

import argparse
import json
import math
import pickle
import sys
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from rdkit import Chem
from rdkit.Chem import AllChem

DATASETS = ["esol", "freesolv", "lipophilicity"]
SEEDS = [42, 123, 456]


# =============================================================================
# PaiNN Architecture (self-contained, no torch_scatter dependency)
# =============================================================================

def scatter_add(x, idx_i, dim_size, dim=0):
    """Pure PyTorch scatter_add replacement."""
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, idx_i, x)
    return y


class Dense(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, activation=None,
                 weight_init=xavier_uniform_, bias_init=zeros_):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super().__init__(in_features, out_features, bias)
        self.activation = activation if activation is not None else nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y


class GaussianRBF(nn.Module):
    def __init__(self, n_rbf, cutoff, start=0.0):
        super().__init__()
        self.n_rbf = n_rbf
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        self.register_buffer("widths", widths)
        self.register_buffer("offsets", offset)

    def forward(self, inputs):
        coeff = -0.5 / torch.pow(self.widths, 2)
        diff = inputs[..., None] - self.offsets
        return torch.exp(coeff * torch.pow(diff, 2))


class CosineCutoff(nn.Module):
    def __init__(self, cutoff):
        super().__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, input):
        input_cut = 0.5 * (torch.cos(input * math.pi / self.cutoff) + 1.0)
        input_cut *= (input < self.cutoff).float()
        return input_cut


class PaiNNInteraction(nn.Module):
    def __init__(self, n_atom_basis, activation):
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.interatomic_context_net = nn.Sequential(
            Dense(n_atom_basis, n_atom_basis, activation=activation),
            Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )

    def forward(self, q, mu, Wij, dir_ij, idx_i, idx_j, n_atoms):
        x = self.interatomic_context_net(q)
        xj = x[idx_j]
        muj = mu[idx_j]
        x = Wij * xj
        dq, dmuR, dmumu = torch.split(x, self.n_atom_basis, dim=-1)
        dq = scatter_add(dq, idx_i, dim_size=n_atoms)
        dmu = dmuR * dir_ij[..., None] + dmumu * muj
        dmu = scatter_add(dmu, idx_i, dim_size=n_atoms)
        q = q + dq
        mu = mu + dmu
        return q, mu


class PaiNNMixing(nn.Module):
    def __init__(self, n_atom_basis, activation, epsilon=1e-8):
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.intraatomic_context_net = nn.Sequential(
            Dense(2 * n_atom_basis, n_atom_basis, activation=activation),
            Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )
        self.mu_channel_mix = Dense(n_atom_basis, 2 * n_atom_basis, activation=None, bias=False)
        self.epsilon = epsilon

    def forward(self, q, mu):
        mu_mix = self.mu_channel_mix(mu)
        mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)
        mu_Vn = torch.sqrt(torch.sum(mu_V ** 2, dim=-2, keepdim=True) + self.epsilon)
        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intraatomic_context_net(ctx)
        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)
        dmu_intra = dmu_intra * mu_W
        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)
        q = q + dq_intra + dqmu_intra
        mu = mu + dmu_intra
        return q, mu


def radius_graph_pure(pos, batch, cutoff=5.0, max_num_neighbors=32):
    """Pure PyTorch radius graph computation."""
    device = pos.device
    n = pos.size(0)
    if n == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    dist = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)
    same_batch = batch.unsqueeze(0) == batch.unsqueeze(1)
    within_cutoff = dist < cutoff
    not_self = ~torch.eye(n, dtype=torch.bool, device=device)
    mask = same_batch & within_cutoff & not_self

    if max_num_neighbors < n:
        dist_masked = dist.clone()
        dist_masked[~mask] = float('inf')
        _, topk_idx = dist_masked.topk(min(max_num_neighbors, n - 1), dim=1, largest=False)
        topk_mask = torch.zeros_like(mask)
        topk_mask.scatter_(1, topk_idx, True)
        mask = mask & topk_mask

    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    return edge_index.flip(0)


class PaiNNModel(nn.Module):
    """Self-contained PaiNN model for molecular property prediction."""

    def __init__(self, hidden_dim=128, num_interactions=3, num_rbf=20,
                 cutoff=5.0, max_atomic_num=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_interactions = num_interactions
        self.cutoff = cutoff

        self.radial_basis = GaussianRBF(n_rbf=num_rbf, cutoff=cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.embedding = nn.Embedding(max_atomic_num, hidden_dim, padding_idx=0)

        self.filter_net = Dense(
            num_rbf, num_interactions * hidden_dim * 3, activation=None
        )

        self.interactions = nn.ModuleList([
            PaiNNInteraction(n_atom_basis=hidden_dim, activation=F.silu)
            for _ in range(num_interactions)
        ])
        self.mixing = nn.ModuleList([
            PaiNNMixing(n_atom_basis=hidden_dim, activation=F.silu)
            for _ in range(num_interactions)
        ])

        # Output head
        self.output_net = nn.Sequential(
            Dense(hidden_dim, hidden_dim, activation=F.silu),
            Dense(hidden_dim, 1, activation=None),
        )

    def forward(self, z, pos, batch):
        edge_index = radius_graph_pure(pos, batch, self.cutoff)

        if edge_index.size(1) == 0:
            n_graphs = batch.max().item() + 1
            return torch.zeros(n_graphs, device=pos.device)

        idx_i, idx_j = edge_index[0], edge_index[1]
        r_ij = pos[idx_i] - pos[idx_j]
        n_atoms = z.size(0)

        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / (d_ij + 1e-8)
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)

        filters = self.filter_net(phi_ij) * fcut[..., None]
        filter_list = torch.split(filters, 3 * self.hidden_dim, dim=-1)

        q = self.embedding(z)[:, None]
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)

        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)

        q = q.squeeze(1)

        # Per-atom output
        out = self.output_net(q).squeeze(-1)

        # Sum pooling per molecule
        n_graphs = batch.max().item() + 1
        result = torch.zeros(n_graphs, device=out.device)
        result.scatter_add_(0, batch, out)

        return result


# =============================================================================
# Data loading (reused from SchNet baseline)
# =============================================================================

def smiles_to_pyg_data(smiles: str, target: float, max_atoms: int = 100) -> Data:
    """Convert SMILES to PyG Data object with 3D coordinates."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result == -1:
                return None
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
    except Exception:
        return None

    if mol.GetNumConformers() == 0:
        return None

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    if n_atoms > max_atoms:
        return None

    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
    pos = torch.tensor([list(conf.GetAtomPosition(i)) for i in range(n_atoms)], dtype=torch.float)
    y = torch.tensor([target], dtype=torch.float)

    return Data(z=z, pos=pos, y=y)


def load_dataset_pyg(dataset_name: str, split: str = "train"):
    """Load dataset and convert to PyG format."""
    data_path = Path(f"data/conformers/{dataset_name}/{split}.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    smiles_list = data["smiles"]
    targets = np.array(data["labels"]).squeeze()

    pyg_data = []
    for smi, target in zip(smiles_list, targets):
        d = smiles_to_pyg_data(smi, target)
        if d is not None:
            pyg_data.append(d)

    return pyg_data


# =============================================================================
# Training and evaluation
# =============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.z, data.pos, data.batch)
        target = data.y.squeeze()
        if target.dim() == 0:
            target = target.unsqueeze(0)
        if out.dim() == 0:
            out = out.unsqueeze(0)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.z, data.pos, data.batch)
            out_np = out.cpu().numpy()
            y_np = data.y.squeeze(-1).cpu().numpy()
            if out_np.ndim == 0:
                out_np = out_np.reshape(1)
            if y_np.ndim == 0:
                y_np = y_np.reshape(1)
            preds.extend(out_np.tolist())
            targets.extend(y_np.tolist())

    preds = np.array(preds)
    targets = np.array(targets)

    return {
        "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
        "mae": float(mean_absolute_error(targets, preds)),
        "r2": float(r2_score(targets, preds)),
    }


def run_experiment(dataset_name: str, seed: int, device: str, epochs: int = 200):
    """Run single PaiNN experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"  Loading {dataset_name}...")
    train_data = load_dataset_pyg(dataset_name, "train")
    val_data = load_dataset_pyg(dataset_name, "val")
    test_data = load_dataset_pyg(dataset_name, "test")

    if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
        raise ValueError(f"Empty split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    model = PaiNNModel(
        hidden_dim=128,
        num_interactions=3,
        num_rbf=20,
        cutoff=5.0,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["rmse"])

        if val_metrics["rmse"] < best_val_loss:
            best_val_loss = val_metrics["rmse"]
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 50:
            print(f"  Early stopping at epoch {epoch}")
            break

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_rmse={val_metrics['rmse']:.4f}")

    model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_loader, device)

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output-dir", default="results/painn_baseline")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("PaiNN Baseline Experiments")
    print(f"Device: {device}")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}")
    print("=" * 60)

    results = []

    for dataset in args.datasets:
        print(f"\n=== {dataset.upper()} ===")
        for seed in args.seeds:
            print(f"\nSeed {seed}:")
            try:
                metrics = run_experiment(dataset, seed, device, args.epochs)
                results.append({
                    "dataset": dataset,
                    "seed": seed,
                    "model": "painn",
                    **metrics
                })
                print(f"  Test: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")

                with open(output_dir / "painn_results_partial.json", "w") as f:
                    json.dump({"results": results}, f, indent=2)

            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for dataset in args.datasets:
        ds_results = [r for r in results if r["dataset"] == dataset]
        if ds_results:
            rmses = [r["rmse"] for r in ds_results]
            print(f"{dataset}: RMSE = {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")

    with open(output_dir / "painn_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}/painn_results.json")


if __name__ == "__main__":
    main()
