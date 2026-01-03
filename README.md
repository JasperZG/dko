# Distribution Kernel Operators (DKO) for Molecular Property Prediction

A comprehensive research framework for learning molecular properties from conformer ensemble distributions using Distribution Kernel Operators.

## Overview

This project implements DKO, a novel approach to molecular property prediction that explicitly models the distribution of molecular conformers rather than relying on single conformer representations. The key insight is that many molecular properties depend on the ensemble of accessible 3D structures, not just the lowest-energy conformation.

### Key Features

- **Distribution Kernel Operators (DKO)**: Novel kernel-based method for aggregating conformer information
- **12 Benchmark Datasets**: Comprehensive evaluation across binding affinity, ADMET, and quantum mechanical properties
- **Multiple Baselines**: Comparison with DeepSets, Attention, SchNet, DimeNet++, and other state-of-the-art methods
- **Rigorous Evaluation**: Scaffold splitting, multiple seeds, and statistical significance testing
- **Extensive Analysis**: Sample efficiency, attention visualization, and statistical consistency checks

## Installation

### Requirements

- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- ~50GB disk space for datasets

### Setup

```bash
# Clone the repository
git clone https://github.com/JasperZG/dko.git
cd dko

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Optional: RDKit Installation

If RDKit installation fails via pip, use conda:

```bash
conda install -c conda-forge rdkit
```

## Quick Start

### 1. Prepare Datasets

```bash
# Download and preprocess all datasets
python scripts/prepare_datasets.py --all

# Or prepare specific datasets
python scripts/prepare_datasets.py --datasets bace pdbbind freesolv
```

### 2. Run Experiments

```bash
# Run main benchmark on all datasets
python scripts/run_experiment.py --experiment main_benchmark

# Run on specific dataset with specific model
python scripts/run_experiment.py --dataset bace --model dko

# Run with hyperparameter optimization
python scripts/run_experiment.py --dataset bace --model dko --hyperopt
```

### 3. Analyze Results

```bash
# Generate analysis and visualizations
python scripts/analyze_results.py --experiment main_benchmark
```

## Project Structure

```
dko-research/
├── dko/
│   ├── models/          # Model implementations
│   │   ├── dko.py       # Distribution Kernel Operator
│   │   ├── attention.py # Attention-based aggregation
│   │   ├── deepsets.py  # DeepSets baseline
│   │   └── gnn_baselines.py  # SchNet, DimeNet++, etc.
│   ├── data/            # Data loading and processing
│   │   ├── datasets.py  # Dataset classes
│   │   ├── conformers.py # Conformer generation
│   │   ├── features.py  # Feature extraction
│   │   └── splits.py    # Data splitting utilities
│   ├── training/        # Training infrastructure
│   │   ├── trainer.py   # Training loop
│   │   ├── evaluator.py # Evaluation metrics
│   │   └── hyperopt.py  # Hyperparameter optimization
│   ├── experiments/     # Experiment scripts
│   │   ├── main_benchmark.py
│   │   ├── decomposition.py
│   │   ├── sample_efficiency.py
│   │   └── attention_analysis.py
│   ├── analysis/        # Analysis utilities
│   │   ├── scc.py       # Statistical consistency checks
│   │   ├── statistics.py
│   │   └── visualization.py
│   └── utils/           # Utility functions
│       ├── config.py    # Configuration system
│       └── logging_utils.py
├── configs/             # Configuration files
│   ├── base_config.yaml
│   ├── datasets/        # Per-dataset configs
│   └── models/          # Per-model configs
├── scripts/             # Entry point scripts
├── tests/               # Unit tests
└── notebooks/           # Analysis notebooks
```

## Datasets

| Dataset | Task | Type | Molecules | Expected DKO Advantage |
|---------|------|------|-----------|------------------------|
| BACE | Binding Affinity | Regression | 1,513 | 5-8% |
| PDBBind | Binding Affinity | Regression | 11,908 | 8-12% |
| FreeSolv | Solvation | Regression | 642 | 4-7% |
| hERG | Toxicity | Classification | 7,889 | 2-4% |
| CYP3A4 | Metabolism | Classification | 12,328 | 2-3% |
| Tox21 | Toxicity | Classification | 7,831 | 1-3% |
| BBBP | Permeability | Classification | 2,039 | 1-2% |
| ESOL | Solubility | Regression | 1,128 | 1-3% |
| Lipophilicity | Lipophilicity | Regression | 4,200 | 1-3% |
| QM9 HOMO | Electronic | Regression | 133,885 | 4-6% |
| QM9 Gap | Electronic | Regression | 133,885 | 4-6% |
| QM9 Polar | Electronic | Regression | 133,885 | 5-7% |

## Configuration

Configuration uses a hierarchical YAML system:

```yaml
# configs/base_config.yaml - Base settings
# configs/datasets/bace.yaml - Dataset-specific
# configs/models/dko.yaml - Model-specific
```

Override settings via environment variables:

```bash
export DKO_TRAINING_BATCH_SIZE=64
export DKO_TRAINING_BASE_LEARNING_RATE=0.0001
```

## Experiment Tracking

Experiments are tracked with Weights & Biases:

```bash
# Set W&B credentials
export WANDB_ENTITY=your-entity
export WANDB_API_KEY=your-key

# Run with tracking
python scripts/run_experiment.py --dataset bace --model dko
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=dko --cov-report=html
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dko2026,
  title={Distribution Kernel Operators for Molecular Property Prediction from Conformer Ensembles},
  author={JasperZG},
  year={2026},
  url={https://github.com/JasperZG/dko}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- RDKit team for molecular informatics tools
- PyTorch Geometric team for GNN implementations
- MoleculeNet for benchmark datasets
