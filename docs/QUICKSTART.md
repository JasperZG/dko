# DKO Quick Start Guide

## Installation

```bash
# Clone and setup
git clone https://github.com/JasperZG/dko.git
cd dko

# Create environment
conda create -n dko python=3.11
conda activate dko
conda install -c conda-forge rdkit
pip install torch numpy scipy scikit-learn pandas pyyaml tqdm optuna matplotlib

# Verify installation
python validate_parts1_5_complete.py
```

## Basic Usage

### 1. Train DKO Model

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from dko.models.dko import DKO
from dko.training.trainer import Trainer
from dko.training.evaluator import Evaluator

# Prepare data (mu, sigma, labels)
mu = torch.randn(100, 50)           # (N, D) mean vectors
sigma = torch.randn(100, 50, 50)    # (N, D, D) covariance matrices
sigma = torch.bmm(sigma, sigma.transpose(1, 2))  # Ensure PSD
labels = torch.randn(100, 1)        # (N, 1) targets

# Create data loader
def collate_fn(batch):
    mu, sigma, labels = zip(*batch)
    return {'mu': torch.stack(mu), 'sigma': torch.stack(sigma), 'label': torch.stack(labels)}

dataset = TensorDataset(mu, sigma, labels)
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# Create model
model = DKO(feature_dim=50, output_dim=1, verbose=True)

# Train
trainer = Trainer(model, task='regression', max_epochs=100)
history = trainer.fit(loader, loader)

# Evaluate
evaluator = Evaluator(task_type='regression')
metrics = evaluator.evaluate(model, loader)
print(f"RMSE: {metrics['rmse']:.4f}")
```

### 2. Train Baseline Models

```python
from dko.models.attention import AttentionPoolingBaseline
from dko.models.deepsets import DeepSetsBaseline

# Conformer-level data
features = torch.randn(100, 20, 50)  # (N, K, D) K conformers

def collate_fn_baseline(batch):
    features, labels = zip(*batch)
    return {'features': torch.stack(features), 'label': torch.stack(labels)}

dataset = TensorDataset(features, labels)
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn_baseline)

# Attention baseline
attention = AttentionPoolingBaseline(feature_dim=50, output_dim=1)
trainer = Trainer(attention, task='regression', max_epochs=100)
trainer.fit(loader, loader)

# DeepSets baseline
deepsets = DeepSetsBaseline(feature_dim=50, output_dim=1)
trainer = Trainer(deepsets, task='regression', max_epochs=100)
trainer.fit(loader, loader)
```

### 3. Hyperparameter Optimization

```python
from dko.training.hyperopt import run_hyperopt

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
print(f"Best params: {results['best_params']}")
```

### 4. Full Data Pipeline

```python
from dko.data.conformer_generator import ConformerGenerator
from dko.data.feature_extractor import FeatureExtractor
from dko.data.augmented_basis import AugmentedBasisComputer

# Generate conformers
generator = ConformerGenerator(num_conformers=20)
result = generator.generate("CCO")  # SMILES

# Extract features
extractor = FeatureExtractor(descriptor_type='rdkit_2d')
features = extractor.extract(result['mol'], result['conformers'])

# Compute augmented basis
computer = AugmentedBasisComputer(pca_components=50)
computer.fit(features)  # Fit on training data
mu, sigma = computer.compute(features, result['weights'])
```

## Configuration

```yaml
# configs/experiments/my_experiment.yaml
experiment:
  name: my_experiment

dataset:
  name: esol
  task: regression

model:
  type: dko
  dko:
    pca_components: 50
    kernel_hidden_dims: [256, 128]
    dropout: 0.1

training:
  learning_rate: 1.0e-4
  max_epochs: 300
  early_stopping_patience: 30
```

## Running Tests

```bash
# Unit tests
pytest tests/ -v

# Integration test
python scripts/test_training_integration.py

# Full validation
python validate_parts1_5_complete.py
```

## Key Files

| File | Purpose |
|------|---------|
| `dko/models/dko.py` | DKO model implementation |
| `dko/models/attention.py` | Attention baseline |
| `dko/models/deepsets.py` | DeepSets baseline |
| `dko/training/trainer.py` | Training infrastructure |
| `dko/training/evaluator.py` | Evaluation metrics |
| `dko/training/hyperopt.py` | Hyperparameter optimization |
| `configs/base_config.yaml` | Default configuration |

## Model Comparison

| Model | Input | Key Feature |
|-------|-------|-------------|
| DKO | [μ, Σ] | Uses both mean and covariance |
| DKO_FirstOrder | μ only | Ablation: mean only |
| Attention | (B, K, D) | Learnable attention weights |
| DeepSets | (B, K, D) | Permutation-invariant pooling |

## Metrics

**Regression:** RMSE, MAE, R², Pearson, Spearman

**Classification:** AUC-ROC, AUC-PR, Accuracy, F1

## Need Help?

- Full docs: `docs/TECHNICAL_DOCUMENTATION.md`
- Tests: `tests/`
- Examples: `scripts/test_training_integration.py`
