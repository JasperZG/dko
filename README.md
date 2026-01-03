# Distribution Kernel Operators (DKO) for Molecular Property Prediction

A comprehensive research framework for learning molecular properties from conformer ensemble distributions using Distribution Kernel Operators.

## Project Status

| Part | Description | Tests | Status |
|------|-------------|-------|--------|
| 1 | Project Foundation | - | Complete |
| 2 | Data Pipeline | - | Complete |
| 3 | Datasets (12 benchmarks) | - | Complete |
| 4 | Models (DKO, Attention, DeepSets) | 98 | Complete |
| 5 | Training Infrastructure | 104 | Complete |
| 6 | Experiments | - | Ready to Start |

**Total Tests: 104/104 passing** | **Validation: All systems go**

## Overview

This project implements DKO, a novel approach to molecular property prediction that explicitly models the distribution of molecular conformers using an **augmented basis representation**:

```
[mu, Sigma] where:
  mu = (batch, D) - mean of conformer features
  Sigma = (batch, D, D) - covariance matrix capturing conformational variability
```

### Key Features

- **Distribution Kernel Operators (DKO)**: Novel kernel-based method for aggregating conformer information
- **Augmented Basis Representation**: Captures both first-order (mean) and second-order (covariance) statistics
- **12 Benchmark Datasets**: Comprehensive evaluation across ESOL, Lipophilicity, BACE, BBBP, HIV, Tox21, etc.
- **Multiple Baselines**: Comparison with Attention Pooling and DeepSets
- **Rigorous Evaluation**: Scaffold splitting, multiple seeds, bootstrap CI, and statistical significance testing
- **Stratified Analysis**: Performance analysis by Structural Conformer Complexity (SCC) quartiles

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- RDKit 2023.03+
- CUDA 11.8+ (optional, for GPU)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/dkoproject.git
cd dkoproject

# Create environment
conda create -n dko python=3.11
conda activate dko

# Install RDKit
conda install -c conda-forge rdkit

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy scipy scikit-learn pandas pyyaml tqdm optuna matplotlib

# Verify installation
python validate_parts1_5_complete.py
```

## Quick Start

### 1. Train DKO Model

```python
from dko.models.dko import DKO
from dko.training.trainer import Trainer
from dko.training.evaluator import Evaluator

# Create model
model = DKO(feature_dim=50, output_dim=1, verbose=True)

# Train with research plan specifications
trainer = Trainer(
    model=model,
    task='regression',
    learning_rate=1e-4,        # AdamW
    weight_decay=1e-5,
    max_epochs=300,
    early_stopping_patience=30,
)
history = trainer.fit(train_loader, val_loader)

# Evaluate with full metrics
evaluator = Evaluator(task_type='regression')
metrics = evaluator.evaluate(model, test_loader)
print(f"RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")
```

### 2. Run Integration Tests

```bash
# Comprehensive integration test with visualizations
python scripts/test_training_integration.py

# Outputs:
# - training_curves.png
# - prediction_quality.png
# - error_distributions.png
# - model_comparison.png
# - test_report.json
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
    n_trials=50,  # TPE sampler with MedianPruner
)
```

## Project Structure

```
dkoproject/
├── dko/
│   ├── data/               # Data pipeline
│   │   ├── conformer_generator.py   # ETKDG conformer generation
│   │   ├── feature_extractor.py     # RDKit/Mordred descriptors
│   │   ├── augmented_basis.py       # [mu, Sigma] computation
│   │   ├── scc_calculator.py        # Structural Conformer Complexity
│   │   ├── datasets.py              # Dataset classes
│   │   └── splitters.py             # Scaffold/random splitting
│   ├── models/             # Model implementations
│   │   ├── dko.py          # DKO (full + ablations)
│   │   ├── attention.py    # Attention pooling baseline
│   │   └── deepsets.py     # DeepSets baseline
│   ├── training/           # Training infrastructure
│   │   ├── trainer.py      # Basic trainer (AdamW, cosine, early stop)
│   │   ├── hpc_trainer.py  # HPC-grade trainer with logging
│   │   ├── evaluator.py    # All metrics + bootstrap CI
│   │   └── hyperopt.py     # Optuna TPE optimization
│   └── utils/              # Utilities
├── configs/
│   ├── base_config.yaml    # Default settings
│   ├── datasets/           # Per-dataset configs (12 datasets)
│   └── experiments/        # Experiment templates
├── scripts/
│   ├── submit_hpc.sh       # SLURM job template
│   ├── submit_batch.py     # Batch experiment submission
│   └── test_training_integration.py  # Integration tests
├── tests/                  # 104 tests
│   ├── test_trainer.py     # 40 tests
│   ├── test_evaluator.py   # 36 tests
│   └── test_hyperopt.py    # 28 tests
└── docs/
    ├── TECHNICAL_DOCUMENTATION.md
    └── QUICKSTART.md
```

## Models

| Model | Input | Description | Parameters (D=100) |
|-------|-------|-------------|-------------------|
| **DKO** | [mu, Sigma] | Full distribution kernel operator | ~8,500 |
| **DKO_FirstOrder** | mu only | Ablation: mean only | ~729,000 |
| **AttentionPooling** | (B, K, D) | Learnable attention over conformers | ~222,000 |
| **DeepSets** | (B, K, D) | Permutation-invariant set function | ~130,000 |

## Training Specifications (from Research Plan)

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 to 1e-6 (cosine) |
| Weight Decay | 1e-5 |
| Early Stopping | Patience 30 |
| Gradient Clipping | max_norm=1.0 |
| Mixed Precision | FP16 |
| Batch Size | 32 |
| Max Epochs | 300 |

## Evaluation Metrics

**Regression:** RMSE, MAE, R2, Pearson, Spearman (with 95% bootstrap CI)

**Classification:** AUC-ROC, AUC-PR, Accuracy, Precision, Recall, F1

**Statistical Tests:** Paired t-test, Wilcoxon signed-rank

## Datasets (12 Benchmarks)

| Dataset | Task | Size | Metric |
|---------|------|------|--------|
| ESOL | Regression | 1,128 | RMSE |
| Lipophilicity | Regression | 4,200 | RMSE |
| FreeSolv | Regression | 642 | RMSE |
| BACE | Classification | 1,513 | AUC |
| BBBP | Classification | 2,039 | AUC |
| HIV | Classification | 41,127 | AUC |
| Tox21 | Multi-task | 7,831 | AUC |
| ClinTox | Multi-task | 1,478 | AUC |
| SIDER | Multi-task | 1,427 | AUC |
| ToxCast | Multi-task | 8,576 | AUC |
| MUV | Multi-task | 93,087 | AUC |
| QM7 | Regression | 7,165 | MAE |

## Testing

```bash
# Run all tests (104 tests)
pytest tests/ -v

# Run specific test suite
pytest tests/test_trainer.py -v      # 40 tests
pytest tests/test_evaluator.py -v    # 36 tests
pytest tests/test_hyperopt.py -v     # 28 tests

# Full validation
python validate_parts1_5_complete.py
```

## Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get started in 5 minutes
- **[Technical Documentation](docs/TECHNICAL_DOCUMENTATION.md)** - Full API reference

## HPC Deployment

```bash
# Submit SLURM job
sbatch scripts/submit_hpc.sh

# Batch submission for multiple experiments
python scripts/submit_batch.py \
    --datasets esol lipophilicity freesolv \
    --models dko attention deepsets \
    --seeds 42 123 456
```

## Citation

```bibtex
@software{dko2026,
  title={DKO: Distribution Kernel Operators for Molecular Property Prediction},
  author={Your Name},
  year={2026},
  url={https://github.com/your-org/dkoproject}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- RDKit team for molecular informatics tools
- MoleculeNet for benchmark datasets
- Optuna team for hyperparameter optimization
