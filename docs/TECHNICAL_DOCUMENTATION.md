# DKO (Distribution Kernel Operators) - Technical Documentation

**Version:** 1.1.0
**Last Updated:** January 5, 2026
**Status:** Complete with all models, baselines, and experiments

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Installation & Setup](#3-installation--setup)
4. [Core Concepts](#4-core-concepts)
5. [Data Pipeline](#5-data-pipeline)
6. [Models](#6-models)
7. [Training Infrastructure](#7-training-infrastructure)
8. [Evaluation](#8-evaluation)
9. [Hyperparameter Optimization](#9-hyperparameter-optimization)
10. [Configuration System](#10-configuration-system)
11. [API Reference](#11-api-reference)
12. [Testing](#12-testing)
13. [HPC Deployment](#13-hpc-deployment)

---

## 1. Project Overview

### 1.1 Purpose

DKO (Distribution Kernel Operators) is a research framework for molecular property prediction that leverages conformer ensemble information through a novel kernel-based approach. Unlike traditional methods that aggregate conformer features through simple pooling, DKO represents each molecule as a probability distribution over conformer space and learns kernel operators on these distributions.

### 1.2 Key Innovation

The core insight is representing molecular conformer ensembles using an **augmented basis representation**:

```
[μ, Σ] where:
  μ = (batch, D) - mean of conformer features
  Σ = (batch, D, D) - covariance matrix of conformer features
```

This captures both:
- **First-order statistics (μ)**: Average molecular properties
- **Second-order statistics (Σ)**: Conformational variability and correlations

### 1.3 Research Objectives

1. Demonstrate that second-order statistics improve molecular property prediction
2. Compare DKO against attention-based and set-based baselines
3. Validate across 12 benchmark datasets (regression and classification)
4. Analyze performance stratified by Structural Conformer Complexity (SCC)

### 1.4 Project Structure

```
dkoproject/
├── dko/                          # Main package
│   ├── __init__.py
│   ├── data/                     # Data pipeline
│   │   ├── __init__.py
│   │   ├── conformer_generator.py    # ETKDG conformer generation
│   │   ├── feature_extractor.py      # RDKit/Mordred descriptors
│   │   ├── augmented_basis.py        # [μ, Σ] computation
│   │   ├── scc_calculator.py         # Structural Conformer Complexity
│   │   ├── datasets.py               # Dataset classes
│   │   └── splitters.py              # Scaffold/random splitting
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   ├── dko.py                    # DKO model (full + ablations)
│   │   ├── attention.py              # Attention pooling baseline
│   │   ├── deepsets.py               # DeepSets baseline
│   │   ├── ensemble_baselines.py     # MFA, MIL, Mean/Boltzmann ensembles
│   │   └── gnn_baselines.py          # SchNet, DimeNet++, SphereNet (PyG)
│   ├── training/                 # Training infrastructure
│   │   ├── __init__.py
│   │   ├── trainer.py                # Basic trainer
│   │   ├── hpc_trainer.py            # HPC-grade trainer
│   │   ├── evaluator.py              # Metrics & evaluation
│   │   └── hyperopt.py               # Optuna optimization
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── config.py                 # Configuration management
│       └── registry.py               # Model/dataset registry
├── configs/                      # Configuration files
│   ├── base_config.yaml              # Default settings
│   ├── datasets/                     # Per-dataset configs
│   │   ├── esol.yaml
│   │   ├── lipophilicity.yaml
│   │   └── ... (12 datasets)
│   └── experiments/                  # Experiment templates
│       ├── dko_template.yaml
│       └── attention_template.yaml
├── scripts/                      # Utility scripts
│   ├── submit_hpc.sh                 # SLURM job template
│   ├── submit_all_experiments.py     # Batch experiment submission
│   ├── aggregate_results.py          # Results aggregation
│   └── test_training_integration.py  # Integration tests
├── tests/                        # Test suite
│   ├── test_trainer.py               # 40 trainer tests
│   ├── test_evaluator.py             # 36 evaluator tests
│   └── test_hyperopt.py              # 28 hyperopt tests
└── docs/                         # Documentation
    └── TECHNICAL_DOCUMENTATION.md
```

---

## 2. Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│  SMILES → Conformer Generation → Feature Extraction → [μ, Σ]        │
│           (ETKDG + MMFF)         (RDKit 2D)           (PCA)         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         MODEL LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────────────┐ │
│  │    DKO      │  │   Attention     │  │      DeepSets            │ │
│  │             │  │   Baseline      │  │      Baseline            │ │
│  │ μ + Σ → K   │  │ Q,K,V → attn    │  │ φ(x) → ρ(Σφ) → pred     │ │
│  └─────────────┘  └─────────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       TRAINING LAYER                                 │
├─────────────────────────────────────────────────────────────────────┤
│  AdamW + Cosine Annealing + Early Stopping + Mixed Precision        │
│  Checkpointing + Gradient Clipping + Experiment Logging             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      EVALUATION LAYER                                │
├─────────────────────────────────────────────────────────────────────┤
│  Regression: RMSE, MAE, R², Pearson, Spearman                       │
│  Classification: AUC-ROC, AUC-PR, Accuracy, F1                      │
│  Statistical: Bootstrap CI, Paired t-test, Wilcoxon                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
Raw SMILES
    │
    ▼
┌──────────────────────┐
│ ConformerGenerator   │  Generate 3D conformers using ETKDG
│ - num_conformers: 20 │  Optimize with MMFF force field
│ - energy_window: 10  │  Compute Boltzmann weights from energies
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│ FeatureExtractor     │  Extract molecular descriptors
│ - RDKit 2D (200 dim) │  Normalize features (z-score)
│ - Optional: Mordred  │  Handle missing values
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│ AugmentedBasis       │  Compute distribution statistics
│ - μ = Σ wᵢ xᵢ       │  Mean (Boltzmann-weighted)
│ - Σ = cov(X, W)     │  Covariance matrix (PSD guaranteed)
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│ DKO Model            │  Learn kernel on distributions
│ - PCA reduction      │  MLP on concatenated [μ, vec(Σ)]
│ - PSD enforcement    │  Output prediction
└──────────────────────┘
```

---

## 3. Installation & Setup

### 3.1 Requirements

```
Python >= 3.10
PyTorch >= 2.0
RDKit >= 2023.03
NumPy >= 1.24
SciPy >= 1.10
scikit-learn >= 1.3
pandas >= 2.0
PyYAML >= 6.0
tqdm >= 4.65
optuna >= 3.0 (for hyperparameter optimization)
matplotlib >= 3.7 (for visualization)
```

### 3.2 Installation

```bash
# Clone repository
git clone https://github.com/JasperZG/dko.git
cd dko

# Create conda environment
conda create -n dko python=3.11
conda activate dko

# Install RDKit
conda install -c conda-forge rdkit

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy scipy scikit-learn pandas pyyaml tqdm optuna matplotlib

# Install in development mode
pip install -e .
```

### 3.3 Verification

```bash
# Run validation script
python validate_parts1_5_complete.py

# Run test suite
pytest tests/ -v

# Run integration tests
python scripts/test_training_integration.py
```

---

## 4. Core Concepts

### 4.1 Conformer Ensembles

Molecules exist in multiple 3D conformations. DKO treats each molecule as a **probability distribution** over these conformations:

```python
# Each molecule has multiple conformers
conformers = [conf_1, conf_2, ..., conf_K]  # K conformers

# Each conformer has features
features = [[f_1^1, f_1^2, ..., f_1^D],     # Conformer 1: D features
            [f_2^1, f_2^2, ..., f_2^D],     # Conformer 2: D features
            ...
            [f_K^1, f_K^2, ..., f_K^D]]     # Conformer K: D features

# Boltzmann weights from conformer energies
weights = softmax(-energies / kT)  # Higher weight for lower energy
```

### 4.2 Augmented Basis Representation

The augmented basis captures the full distribution:

```python
# Mean (first-order): weighted average of conformer features
μ = Σᵢ wᵢ * xᵢ  # Shape: (D,)

# Covariance (second-order): spread and correlations
Σ = Σᵢ wᵢ * (xᵢ - μ)(xᵢ - μ)ᵀ  # Shape: (D, D)

# Augmented representation
[μ, Σ]  # Captures both mean and variability
```

### 4.3 Structural Conformer Complexity (SCC)

SCC quantifies how "conformationally diverse" a molecule is:

```python
# SCC = trace(Σ) / D = average variance across features
scc = np.trace(covariance_matrix) / feature_dim

# High SCC: Molecule has diverse conformers
# Low SCC: Conformers are similar (rigid molecule)
```

### 4.4 PSD Enforcement

Covariance matrices must be Positive Semi-Definite (PSD). DKO enforces this via:

```python
# Method 1: Eigenvalue clipping (default)
eigenvalues, eigenvectors = torch.linalg.eigh(Σ)
eigenvalues = torch.clamp(eigenvalues, min=1e-6)
Σ_psd = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T

# Method 2: Cholesky decomposition
L = lower_triangular_network(input)
Σ_psd = L @ L.T
```

---

## 5. Data Pipeline

### 5.1 ConformerGenerator

Generates 3D conformers using RDKit's ETKDG algorithm:

```python
from dko.data.conformer_generator import ConformerGenerator

generator = ConformerGenerator(
    num_conformers=20,      # Target number of conformers
    max_conformers=50,      # Maximum attempts
    energy_window=10.0,     # kcal/mol window for filtering
    optimize=True,          # MMFF optimization
    method='mmff',          # Force field (mmff or uff)
    random_seed=42,
)

# Generate conformers for a molecule
result = generator.generate("CCO")  # Ethanol
# Returns: {
#   'conformers': [...],      # RDKit conformer objects
#   'energies': [...],        # Energy values
#   'weights': [...],         # Boltzmann weights
#   'positions': [...],       # 3D coordinates
# }
```

### 5.2 FeatureExtractor

Extracts molecular descriptors:

```python
from dko.data.feature_extractor import FeatureExtractor

extractor = FeatureExtractor(
    descriptor_type='rdkit_2d',  # 'rdkit_2d', 'mordred', or 'both'
    normalize=True,              # Z-score normalization
    use_3d_features=True,        # Include 3D descriptors
)

# Extract features for conformers
features = extractor.extract(mol, conformers)
# Returns: np.ndarray of shape (num_conformers, feature_dim)
```

### 5.3 AugmentedBasisComputer

Computes the [μ, Σ] representation:

```python
from dko.data.augmented_basis import AugmentedBasisComputer

computer = AugmentedBasisComputer(
    pca_components=50,       # Reduce dimensionality first
    pca_variance=0.95,       # Or keep 95% variance
    regularization=1e-6,     # Covariance regularization
)

# Fit PCA on training data
computer.fit(all_training_features)

# Compute augmented basis for a molecule
mu, sigma = computer.compute(conformer_features, weights)
# mu: (D,) mean vector
# sigma: (D, D) covariance matrix
```

### 5.4 SCCCalculator

Computes Structural Conformer Complexity:

```python
from dko.data.scc_calculator import SCCCalculator

calculator = SCCCalculator()

# Compute SCC for a molecule
scc = calculator.compute(conformer_features, weights)
# Returns: float (higher = more conformational diversity)
```

### 5.5 Dataset Classes

```python
from dko.data.datasets import MolecularDataset, DKODataset

# Standard molecular dataset
dataset = MolecularDataset(
    smiles=['CCO', 'CC(=O)O', ...],
    labels=[0.5, 1.2, ...],
    task='regression',
)

# DKO-formatted dataset (pre-computed μ, Σ)
dko_dataset = DKODataset(
    mu=mu_tensor,           # (N, D)
    sigma=sigma_tensor,     # (N, D, D)
    labels=labels_tensor,   # (N, 1)
    smiles=smiles_list,     # Optional
    scc=scc_tensor,         # Optional
)
```

### 5.6 Scaffold Splitter

Splits data by molecular scaffolds (recommended for molecular ML):

```python
from dko.data.splitters import ScaffoldSplitter

splitter = ScaffoldSplitter(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42,
)

train_idx, val_idx, test_idx = splitter.split(smiles_list)
```

---

## 6. Models

### 6.1 DKO (Distribution Kernel Operator)

The main model that operates on augmented basis representations:

```python
from dko.models.dko import DKO

model = DKO(
    feature_dim=200,              # Input feature dimension
    output_dim=1,                 # Output dimension
    task='regression',            # 'regression' or 'classification'

    # PCA settings
    pca_components=50,            # Reduce to this many components
    pca_variance=None,            # Or keep this % variance

    # Architecture
    kernel_hidden_dims=[256, 128],    # Kernel network layers
    predictor_hidden_dims=[128, 64],  # Prediction network layers

    # Regularization
    dropout=0.1,
    use_batch_norm=True,

    # PSD enforcement
    psd_method='eigenvalue_clipping',  # or 'cholesky'

    verbose=True,
)

# Forward pass
# mu: (batch, D), sigma: (batch, D, D)
predictions = model(mu, sigma, fit_pca=False)
```

**Architecture:**
```
Input: [μ, Σ]
    │
    ▼
┌─────────────────────┐
│ PCA Reduction       │  D → pca_components
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Concatenate         │  [μ_pca, vec(Σ_pca)]
│                     │  Shape: pca_components + pca_components²
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Kernel Network      │  MLP: Linear → BN → ReLU → Dropout
│ [256, 128]          │  Repeated for each layer
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Predictor Network   │  MLP: Linear → BN → ReLU → Dropout
│ [128, 64]           │  Final: Linear → output_dim
└─────────────────────┘
    │
    ▼
Output: predictions
```

### 6.2 DKO Ablations

```python
from dko.models.dko import DKOFirstOrder, DKOSecondOrder, DKONoPSD

# First-order only (μ only, no covariance)
model_first = DKOFirstOrder(feature_dim=200, output_dim=1)

# Second-order only (Σ only, no mean)
model_second = DKOSecondOrder(feature_dim=200, output_dim=1)

# Full DKO without PSD enforcement
model_no_psd = DKONoPSD(feature_dim=200, output_dim=1)
```

### 6.3 AttentionPoolingBaseline

Learnable attention over conformers:

```python
from dko.models.attention import AttentionPoolingBaseline

model = AttentionPoolingBaseline(
    feature_dim=200,              # Input feature dimension
    output_dim=1,                 # Output dimension
    task='regression',

    # Attention settings
    embed_dim=128,                # Embedding dimension
    num_heads=4,                  # Number of attention heads
    num_attention_layers=2,       # Number of transformer layers
    qkv_dim=128,                  # Q, K, V dimension

    # Prediction head
    prediction_hidden_dim=128,

    # Regularization
    dropout=0.1,
    use_layer_norm=True,
)

# Forward pass
# x: (batch, num_conformers, feature_dim)
# mask: (batch, num_conformers) optional
predictions, attention_weights = model(x, mask=mask, return_weights=True)
```

**Architecture:**
```
Input: conformer features (B, K, D)
    │
    ▼
┌─────────────────────┐
│ Input Projection    │  Linear: D → embed_dim
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Multi-Head Attn     │  Q, K, V projections
│ (×num_layers)       │  Scaled dot-product attention
│                     │  + Residual + LayerNorm
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Attention Pooling   │  Weighted sum by attention scores
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Prediction Head     │  MLP: Linear → ReLU → Linear
└─────────────────────┘
    │
    ▼
Output: predictions
```

### 6.4 DeepSetsBaseline

Permutation-invariant set function:

```python
from dko.models.deepsets import DeepSetsBaseline

model = DeepSetsBaseline(
    feature_dim=200,
    output_dim=1,
    task='regression',

    # Encoder (per-element)
    encoder_hidden_dims=[256, 256, 128],

    # Decoder (after pooling)
    decoder_hidden_dim=128,

    # Pooling method
    pooling='boltzmann',          # 'boltzmann', 'mean', 'max', 'sum'

    # Regularization
    dropout=0.1,
)

# Forward pass
# x: (batch, num_conformers, feature_dim)
# weights: (batch, num_conformers) Boltzmann weights
predictions = model(x, weights)
```

**Architecture:**
```
Input: conformer features (B, K, D), weights (B, K)
    │
    ▼
┌─────────────────────┐
│ Encoder φ           │  Per-conformer MLP
│ (shared weights)    │  D → 256 → 256 → 128
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Boltzmann Pooling   │  output = Σᵢ wᵢ * φ(xᵢ)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Decoder ρ           │  MLP: 128 → 128 → output_dim
└─────────────────────┘
    │
    ▼
Output: predictions
```

### 6.5 MeanFeatureAggregation (MFA)

Averages conformer features BEFORE passing to the network:

```python
from dko.models.ensemble_baselines import MeanFeatureAggregation

model = MeanFeatureAggregation(
    feature_dim=200,
    output_dim=1,
    task='regression',

    # Architecture
    hidden_dims=[256, 128],
    prediction_hidden_dims=[64, 32],

    # Aggregation
    aggregation='mean',  # or 'boltzmann'
    temperature=300.0,

    # Regularization
    dropout=0.1,
)

# Forward pass
# x: (batch, num_conformers, feature_dim)
predictions = model(x, mask=mask, energies=energies)
```

**Key difference from MeanEnsemble:**
- MFA: mean(features) → network → prediction
- MeanEnsemble: features → network → mean(predictions)

### 6.6 MultiInstanceLearning (MIL)

Classic MIL with multiple pooling options:

```python
from dko.models.ensemble_baselines import MultiInstanceLearning

model = MultiInstanceLearning(
    feature_dim=200,
    output_dim=1,
    task='regression',

    # Pooling method
    pooling='attention',  # 'max', 'attention', 'mean', 'lse'
    attention_hidden_dim=64,

    # Architecture
    hidden_dims=[256, 128],
    prediction_hidden_dims=[64, 32],
)

# Forward pass with attention weights
predictions, attention_weights = model(x, mask=mask, return_attention=True)
```

### 6.7 GNN Baselines (PyTorch Geometric)

3D GNN models with conformer aggregation:

```python
from dko.models.gnn_baselines import get_gnn, SchNetPyG, DimeNetPPPyG

# Factory function (auto-selects PyG if available)
model = get_gnn('schnet', hidden_channels=128, num_outputs=1)
model = get_gnn('dimenet', hidden_channels=128, out_channels=1)
model = get_gnn('spherenet', hidden_channels=128, out_channels=1)

# Direct instantiation (with PyG)
from dko.models.gnn_baselines import HAS_PYG
if HAS_PYG:
    schnet = SchNetPyG(
        hidden_channels=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        conformer_aggregation='mean',  # 'mean', 'boltzmann', 'attention', 'max'
    )

    dimenet = DimeNetPPPyG(
        hidden_channels=128,
        num_blocks=4,
        conformer_aggregation='boltzmann',
    )
```

---

## 7. Training Infrastructure

### 7.1 Trainer

Basic trainer with all research plan specifications:

```python
from dko.training.trainer import Trainer, create_trainer

# Create trainer
trainer = Trainer(
    model=model,
    task='regression',              # or 'classification'

    # Optimizer (AdamW as specified)
    learning_rate=1e-4,
    weight_decay=1e-5,

    # Scheduler (Cosine annealing)
    scheduler_type='cosine_annealing',
    min_learning_rate=1e-6,

    # Training
    max_epochs=300,
    early_stopping_patience=30,
    gradient_clip_max_norm=1.0,

    # Mixed precision
    use_mixed_precision=True,

    # Checkpointing
    checkpoint_dir='./checkpoints',

    # Logging
    use_wandb=False,

    # Device
    device='cuda',
)

# Train
history = trainer.fit(train_loader, val_loader)
# Returns: {
#   'train_loss': [...],
#   'val_loss': [...],
#   'learning_rates': [...],
#   'best_epoch': int,
#   'best_val_loss': float,
# }

# Save/load checkpoints
trainer.save_checkpoint('checkpoint.pt')
trainer.load_checkpoint('checkpoint.pt')
```

### 7.2 Training Loop Details

```python
# Per-epoch training loop (internal)
def train_epoch(self, train_loader, fit_pca=False):
    self.model.train()

    for batch_idx, batch in enumerate(train_loader):
        # 1. Forward pass
        outputs = self._forward_pass(batch, fit_pca=(fit_pca and batch_idx == 0))
        loss = self.criterion(outputs, labels)

        # 2. Backward pass with mixed precision
        self.optimizer.zero_grad()
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_max_norm)
            self.optimizer.step()

    # 3. Update learning rate
    self.scheduler.step()

    # 4. Early stopping check
    if self.early_stopping(val_loss):
        return True  # Stop training
```

### 7.3 EarlyStopping

```python
from dko.training.trainer import EarlyStopping

early_stopping = EarlyStopping(
    patience=30,           # Epochs without improvement
    min_delta=1e-4,        # Minimum improvement threshold
    mode='min',            # 'min' for loss, 'max' for accuracy
)

# Usage
for epoch in range(max_epochs):
    val_loss = validate()
    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

### 7.4 HPC Trainer

Enhanced trainer for HPC environments with comprehensive logging:

```python
from dko.training.hpc_trainer import EnhancedTrainer, ExperimentLogger

# Create experiment logger
logger = ExperimentLogger(
    experiment_name='dko_esol',
    output_dir='./experiments',
    use_wandb=True,
    wandb_project='dko-experiments',
)

# Log environment info
logger.log_environment()  # SLURM, GPU, Git state

# Create enhanced trainer
trainer = EnhancedTrainer(
    model=model,
    task='regression',
    experiment_logger=logger,

    # Multi-checkpoint strategy
    save_best=True,
    save_last=True,
    save_every_n_epochs=10,
    keep_n_checkpoints=3,
)
```

---

## 8. Evaluation

### 8.1 Evaluator

Comprehensive evaluation with all metrics from research plan:

```python
from dko.training.evaluator import Evaluator

evaluator = Evaluator(
    task_type='regression',         # or 'classification'
    primary_metric='rmse',          # Primary metric for comparison
    device='cuda',
    bootstrap_n_samples=1000,       # For confidence intervals
    confidence_level=0.95,
)

# Basic evaluation
metrics = evaluator.evaluate(model, test_loader, verbose=True)
# Returns: {
#   'rmse': 0.45,
#   'mae': 0.35,
#   'r2': 0.89,
#   'pearson': 0.95,
#   'pearson_p': 1e-10,
#   'spearman': 0.93,
#   'spearman_p': 1e-9,
#   'n_samples': 100,
# }

# With predictions and confidence intervals
results = evaluator.evaluate(
    model, test_loader,
    return_predictions=True,
    compute_ci=True,
)
# Returns: {
#   'metrics': {...},
#   'predictions': np.ndarray,
#   'labels': np.ndarray,
#   'smiles': [...],  # if available
#   'scc': np.ndarray,  # if available
# }
```

### 8.2 Available Metrics

**Regression:**
| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| R² | Coefficient of Determination |
| Pearson | Pearson correlation coefficient |
| Spearman | Spearman rank correlation |

**Classification:**
| Metric | Description |
|--------|-------------|
| AUC-ROC | Area Under ROC Curve |
| AUC-PR | Area Under Precision-Recall Curve |
| Accuracy | Classification accuracy |
| Precision | Positive predictive value |
| Recall | Sensitivity / True positive rate |
| F1 | Harmonic mean of precision and recall |
| Specificity | True negative rate |

### 8.3 Stratified Evaluation

Evaluate performance by SCC quartiles:

```python
# Stratified evaluation by SCC
results = evaluator.stratified_evaluation(
    model, test_loader,
    stratify_by='scc',
    n_bins=4,  # Quartiles
)
# Returns: {
#   'overall': {...metrics...},
#   'stratified': {
#     'quartile_1': {...metrics..., 'bin_range': (0.1, 0.3), 'n_samples': 25},
#     'quartile_2': {...metrics..., 'bin_range': (0.3, 0.5), 'n_samples': 25},
#     'quartile_3': {...metrics..., 'bin_range': (0.5, 0.7), 'n_samples': 25},
#     'quartile_4': {...metrics..., 'bin_range': (0.7, 1.0), 'n_samples': 25},
#   },
# }
```

### 8.4 Statistical Significance

```python
# Paired t-test between models
stat, p_value = evaluator.significance_test(
    model1_rmse_values,  # List of RMSE from multiple seeds
    model2_rmse_values,
    test='paired_t',     # or 'wilcoxon'
)

# Compare multiple models against baseline
comparisons = evaluator.compare_models(
    results_dict={
        'baseline': {'rmse': {'values': [...]}},
        'dko': {'rmse': {'values': [...]}},
        'attention': {'rmse': {'values': [...]}},
    },
    baseline_name='baseline',
)
# Returns: {
#   'dko': {
#     'improvement_percent': 15.3,
#     'p_value': 0.002,
#     'significant': True,
#   },
#   ...
# }
```

### 8.5 Bootstrap Confidence Intervals

```python
from dko.training.evaluator import compute_confidence_intervals

# Compute CI from multiple seed results
mean, ci_lower, ci_upper = compute_confidence_intervals(
    values=[0.45, 0.43, 0.47, 0.44, 0.46],  # RMSE from 5 seeds
    confidence_level=0.95,
)
# mean: 0.45
# ci_lower: 0.43
# ci_upper: 0.47
```

---

## 9. Hyperparameter Optimization

### 9.1 HyperparameterOptimizer

Bayesian optimization using Optuna:

```python
from dko.training.hyperopt import HyperparameterOptimizer, run_hyperopt

optimizer = HyperparameterOptimizer(
    model_class=DKO,
    model_name='dko',
    task='regression',
    feature_dim=50,
    output_dim=1,
    train_loader=train_loader,
    val_loader=val_loader,

    # Optimization settings
    n_trials=50,                    # Number of trials
    max_epochs=100,                 # Max epochs per trial
    early_stopping_patience=15,     # Early stopping within trial

    # Custom search space (optional)
    search_space=None,              # Uses model-specific default

    verbose=True,
)

# Run optimization
results = optimizer.optimize()
# Returns: {
#   'best_params': {'learning_rate': 3e-4, 'dropout': 0.1, ...},
#   'best_value': 0.42,  # Best validation loss
#   'best_trial_number': 23,
#   'n_trials': 50,
#   'n_pruned': 12,  # Trials pruned by MedianPruner
# }

# Get parameter importance
importance = optimizer.get_importance()
# {'learning_rate': 0.45, 'dropout': 0.23, ...}

# Save results
optimizer.save_results('./hyperopt_results')
```

### 9.2 Model-Specific Search Spaces

```python
# DKO search space
DKO_SEARCH_SPACE = {
    'learning_rate': {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-3},
    'weight_decay': {'type': 'log_uniform', 'low': 1e-6, 'high': 1e-4},
    'dropout': {'type': 'categorical', 'choices': [0.0, 0.1, 0.2]},
    'pca_variance': {'type': 'categorical', 'choices': [0.90, 0.95, 0.99]},
    'kernel_output_dim': {'type': 'categorical', 'choices': [32, 64, 128]},
    'branch_hidden_dim': {'type': 'categorical', 'choices': [64, 128, 256]},
}

# Attention search space
ATTENTION_SEARCH_SPACE = {
    'learning_rate': {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-3},
    'weight_decay': {'type': 'log_uniform', 'low': 1e-6, 'high': 1e-4},
    'dropout': {'type': 'categorical', 'choices': [0.0, 0.1, 0.2]},
    'embed_dim': {'type': 'categorical', 'choices': [64, 128, 256]},
    'num_heads': {'type': 'categorical', 'choices': [2, 4, 8]},
    'num_attention_layers': {'type': 'int', 'low': 1, 'high': 3},
}

# DeepSets search space
DEEPSETS_SEARCH_SPACE = {
    'learning_rate': {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-3},
    'weight_decay': {'type': 'log_uniform', 'low': 1e-6, 'high': 1e-4},
    'dropout': {'type': 'categorical', 'choices': [0.0, 0.1, 0.2]},
    'encoder_hidden_dim': {'type': 'categorical', 'choices': [128, 256, 512]},
    'decoder_hidden_dim': {'type': 'categorical', 'choices': [64, 128, 256]},
}
```

### 9.3 Convenience Function

```python
from dko.training.hyperopt import run_hyperopt

# Quick hyperopt
results = run_hyperopt(
    model_class=DKO,
    model_name='dko',
    task='regression',
    feature_dim=50,
    output_dim=1,
    train_loader=train_loader,
    val_loader=val_loader,
    n_trials=50,
)
```

---

## 10. Configuration System

### 10.1 Base Configuration

`configs/base_config.yaml`:
```yaml
# Default settings for all experiments
conformer:
  num_conformers: 20
  max_conformers: 50
  energy_window: 10.0
  optimize: true
  method: mmff

features:
  descriptor_type: rdkit_2d
  normalize: true
  use_3d_features: true

training:
  optimizer: adamw
  base_learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  scheduler: cosine_annealing
  min_learning_rate: 1.0e-6
  max_epochs: 300
  batch_size: 32
  early_stopping_patience: 30
  gradient_clip_max_norm: 1.0
  mixed_precision: true
  seed: 42

evaluation:
  metrics:
    regression: [rmse, mae, r2, spearman]
    classification: [auroc, auprc, accuracy, f1]
  n_splits: 5

hardware:
  device: cuda
  num_workers: 4
  pin_memory: true
```

### 10.2 Dataset Configurations

`configs/datasets/esol.yaml`:
```yaml
dataset:
  name: esol
  task: regression
  target_column: measured log solubility in mols per litre
  smiles_column: smiles
  source: moleculenet
  n_samples: 1128

split:
  method: scaffold
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
```

### 10.3 Experiment Configuration

`configs/experiments/dko_esol.yaml`:
```yaml
experiment:
  name: dko_esol
  description: "DKO model on ESOL aqueous solubility dataset"
  tags: ["dko", "esol", "regression", "solubility"]

dataset:
  name: esol
  task: regression
  split_method: scaffold
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

model:
  type: dko
  dko:
    output_dim: 1
    pca_components: 50
    kernel_hidden_dims: [256, 128]
    predictor_hidden_dims: [128, 64]
    dropout: 0.1
    use_batch_norm: true
    psd_method: eigenvalue_clipping

training:
  optimizer: adamw
  base_learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  scheduler: cosine_annealing
  min_learning_rate: 1.0e-6
  max_epochs: 300
  batch_size: 32
  early_stopping_patience: 30

logging:
  use_wandb: false
  wandb_project: dko-esol
```

### 10.4 Loading Configurations

```python
from dko.utils.config import load_config, merge_configs

# Load single config
config = load_config('configs/experiments/dko_esol.yaml')

# Merge with base config
base = load_config('configs/base_config.yaml')
experiment = load_config('configs/experiments/dko_esol.yaml')
config = merge_configs(base, experiment)

# Access values
learning_rate = config['training']['base_learning_rate']
model_type = config['model']['type']
```

---

## 11. API Reference

### 11.1 dko.models.dko

```python
class DKO(nn.Module):
    """Distribution Kernel Operator model."""

    def __init__(
        self,
        feature_dim: int,
        output_dim: int = 1,
        task: str = 'regression',
        pca_components: Optional[int] = 50,
        pca_variance: Optional[float] = None,
        kernel_hidden_dims: List[int] = [256, 128],
        predictor_hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        psd_method: str = 'eigenvalue_clipping',
        verbose: bool = True,
    ):
        """
        Args:
            feature_dim: Input feature dimension
            output_dim: Output dimension (1 for regression/binary classification)
            task: 'regression' or 'classification'
            pca_components: Number of PCA components (None to skip PCA)
            pca_variance: Variance to retain (alternative to pca_components)
            kernel_hidden_dims: Hidden dimensions for kernel network
            predictor_hidden_dims: Hidden dimensions for predictor network
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            psd_method: 'eigenvalue_clipping' or 'cholesky'
            verbose: Whether to print model info
        """

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        fit_pca: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            mu: Mean vectors (batch, feature_dim)
            sigma: Covariance matrices (batch, feature_dim, feature_dim)
            fit_pca: Whether to fit PCA on this batch (first batch only)

        Returns:
            predictions: (batch, output_dim)
        """

    def count_parameters(self) -> int:
        """Count trainable parameters."""
```

### 11.2 dko.training.trainer

```python
class Trainer:
    """Training manager for molecular property prediction."""

    def __init__(
        self,
        model: nn.Module,
        task: str = 'regression',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        scheduler_type: str = 'cosine_annealing',
        min_learning_rate: float = 1e-6,
        max_epochs: int = 300,
        early_stopping_patience: int = 30,
        gradient_clip_max_norm: float = 1.0,
        use_mixed_precision: bool = True,
        checkpoint_dir: Optional[Path] = None,
        use_wandb: bool = False,
        device: Optional[str] = None,
    ):
        """Initialize trainer with research plan specifications."""

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Returns:
            history: Dictionary with training history
        """

    def train_epoch(
        self,
        train_loader: DataLoader,
        fit_pca: bool = False,
    ) -> Dict[str, float]:
        """Train for one epoch."""

    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Validate the model."""

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model checkpoint."""

    def load_checkpoint(self, path: Union[str, Path]) -> int:
        """Load model checkpoint. Returns epoch number."""
```

### 11.3 dko.training.evaluator

```python
class Evaluator:
    """Comprehensive evaluation for molecular property prediction."""

    def __init__(
        self,
        task_type: str = 'regression',
        primary_metric: Optional[str] = None,
        device: Optional[str] = None,
        bootstrap_n_samples: int = 1000,
        confidence_level: float = 0.95,
    ):
        """Initialize evaluator."""

    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        return_predictions: bool = False,
        compute_ci: bool = False,
        verbose: bool = True,
    ) -> Union[Dict[str, float], Dict[str, Any]]:
        """Evaluate model on dataset."""

    def stratified_evaluation(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        stratify_by: str = 'scc',
        n_bins: int = 4,
    ) -> Dict[str, Any]:
        """Evaluate stratified by a variable (e.g., SCC quartiles)."""

    def significance_test(
        self,
        model1_values: List[float],
        model2_values: List[float],
        test: str = 'paired_t',
    ) -> Tuple[float, float]:
        """Statistical significance test between two models."""

    def compare_models(
        self,
        results_dict: Dict[str, Dict],
        baseline_name: str = 'single_conformer',
    ) -> Dict[str, Dict]:
        """Compare multiple models against baseline."""
```

### 11.4 dko.training.hyperopt

```python
class HyperparameterOptimizer:
    """Bayesian hyperparameter optimization using Optuna."""

    def __init__(
        self,
        model_class: type,
        model_name: str,
        task: str,
        feature_dim: int,
        output_dim: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_trials: int = 50,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        device: Optional[str] = None,
        max_epochs: int = 100,
        early_stopping_patience: int = 15,
        search_space: Optional[Dict] = None,
        verbose: bool = True,
    ):
        """Initialize with TPE sampler and MedianPruner."""

    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization."""

    def get_importance(self) -> Dict[str, float]:
        """Get hyperparameter importance scores."""

    def save_results(self, output_dir: Union[str, Path]) -> None:
        """Save optimization results to YAML and JSON."""
```

---

## 12. Testing

### 12.1 Test Suite Overview

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `tests/test_trainer.py` | 40 | Trainer, EarlyStopping, checkpointing |
| `tests/test_evaluator.py` | 36 | All metrics, CI, stratified evaluation |
| `tests/test_hyperopt.py` | 28 | Optuna integration, search spaces |
| **Total** | **104** | **All passing** |

### 12.2 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_trainer.py -v

# Run with coverage
pytest tests/ --cov=dko --cov-report=html

# Run specific test class
pytest tests/test_evaluator.py::TestRegressionMetrics -v
```

### 12.3 Integration Tests

```bash
# Run comprehensive integration test
python scripts/test_training_integration.py

# Output:
# - test_results/integration_test_YYYYMMDD_HHMMSS/
#   - README.md
#   - test_report.json
#   - training_curves.png
#   - prediction_quality.png
#   - error_distributions.png
#   - model_comparison.png
```

### 12.4 Validation Scripts

```bash
# Validate Parts 1-5 complete
python validate_parts1_5_complete.py

# Expected output:
# [OK] ALL SYSTEMS GO - READY FOR EXPERIMENTS
```

---

## 13. HPC Deployment

### 13.1 SLURM Job Script

`scripts/submit_hpc.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=dko_experiment
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load modules
module load cuda/11.8
module load anaconda3

# Activate environment
source activate dko

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}"
export WANDB_MODE=offline  # For offline logging

# Run experiment
python scripts/run_experiment.py \
    --config configs/experiments/${CONFIG_FILE} \
    --seed ${SEED:-42} \
    --output_dir results/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
```

### 13.2 Batch Submission

```bash
# Submit multiple experiments
python scripts/submit_all_experiments.py \
    --datasets esol lipophilicity freesolv \
    --models dko_first_order dko_second_order \
    --seeds 42 123 456 \
    --submit
```

### 13.3 Results Aggregation

```bash
# Aggregate results from multiple runs
python scripts/aggregate_results.py \
    --results_dir results/ \
    --output results/aggregated_results.csv
```

---

## Appendix A: Benchmark Datasets

| Dataset | Task | Size | Metric | Description |
|---------|------|------|--------|-------------|
| ESOL | Regression | 1,128 | RMSE | Aqueous solubility |
| Lipophilicity | Regression | 4,200 | RMSE | Lipophilicity (logD) |
| FreeSolv | Regression | 642 | RMSE | Hydration free energy |
| BACE | Classification | 1,513 | AUC | BACE-1 inhibitors |
| BBBP | Classification | 2,039 | AUC | Blood-brain barrier |
| HIV | Classification | 41,127 | AUC | HIV replication inhibition |
| Tox21 | Multi-task | 7,831 | AUC | Toxicity (12 tasks) |
| ToxCast | Multi-task | 8,576 | AUC | Toxicity (617 tasks) |
| SIDER | Multi-task | 1,427 | AUC | Side effects (27 tasks) |
| ClinTox | Multi-task | 1,478 | AUC | Clinical trial toxicity |
| MUV | Multi-task | 93,087 | AUC | PubChem bioassays |
| QM7 | Regression | 7,165 | MAE | Atomization energies |

---

## Appendix B: Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size
trainer = Trainer(model, batch_size=16)  # Default is 32

# Or use gradient accumulation (not yet implemented)
```

**2. NaN Loss**
```python
# Check for NaN in inputs
assert not torch.isnan(mu).any(), "NaN in mean"
assert not torch.isnan(sigma).any(), "NaN in covariance"

# Reduce learning rate
trainer = Trainer(model, learning_rate=1e-5)
```

**3. PCA Not Fitted**
```python
# Ensure fit_pca=True on first batch
trainer.train_epoch(train_loader, fit_pca=True)
```

**4. Model Signature Mismatch**
```python
# The evaluator handles different model signatures automatically
# If issues persist, check model.forward() signature
```

---

## Appendix C: Citation

If you use this code, please cite:

```bibtex
@software{dko2026,
  title={DKO: Distribution Kernel Operators for Molecular Property Prediction},
  author={JasperZG},
  year={2026},
  url={https://github.com/JasperZG/dko}
}
```

---

## Appendix D: License

[Your License Here]

---

## Changelog

### v1.1.0 (2026-01-05)
- Add MeanFeatureAggregation (MFA) baseline
- Add MultiInstanceLearning (MIL) baseline
- Add full PyTorch Geometric GNN baselines (SchNet, DimeNet++, SphereNet)
- Add sample efficiency experiments (data fraction + conformer count)
- Add representation vs architecture study
- Add negative control experiments for SCC validation
- Add decision rule calibration with grid search and regret analysis
- Add sketching experiments for large ensembles
- Update documentation

### v1.0.0 (2026-01-03)
- Initial release
- Parts 1-5 complete
- 104 tests passing
- Ready for experiments
