# Distribution Kernel Operators (DKO) for Molecular Property Prediction

**Bryan Cheng, Austin Jin, Jasper Zhang**

> Accepted at **ACM-BCB 2026** (17th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics).

A comprehensive research framework for learning molecular properties from conformer ensemble distributions using Distribution Kernel Operators.

## Overview

This project implements DKO, a novel approach to molecular property prediction that explicitly models the distribution of molecular conformers rather than relying on single conformer representations. The key insight is that many molecular properties depend on the ensemble of accessible 3D structures, not just the lowest-energy conformation.

### Key Features

- **Distribution Kernel Operators (DKO)**: Novel kernel-based method for aggregating conformer information using first-order (mean) and second-order (covariance) statistics
- **12 Benchmark Datasets**: Comprehensive evaluation across binding affinity, ADMET, and quantum mechanical properties
- **Multiple Baselines**: Comparison with DeepSets, Attention, MFA, MIL, SchNet, DimeNet++, SphereNet, and other state-of-the-art methods
- **Full PyTorch Geometric Support**: Native PyG implementations of SchNet and DimeNet++ with conformer aggregation
- **Rigorous Evaluation**: Scaffold splitting, multiple seeds, and statistical significance testing
- **Extensive Analysis**: Sample efficiency, attention visualization, SCC validation, and 80/20 decomposition studies
- **SCC-based Decision Rules**: Automatic method selection based on Structural Conformational Complexity

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
python scripts/run_experiment.py --experiment benchmark

# Run on specific dataset with specific model
python scripts/run_experiment.py --dataset bace --model dko

# Run with hyperparameter optimization
python scripts/run_experiment.py --dataset bace --model dko --hyperopt
```

### 3. Analyze Results

```bash
# Generate analysis and visualizations
python scripts/analyze_results.py --experiment benchmark
```

### Reproducing the paper

Results in the paper map to the following commands (precomputed conformers
are regenerated from `scripts/prepare_datasets.py --all` on first run):

| Paper artifact | Command |
| --- | --- |
| Main benchmark (Tables 1, 2) | `python scripts/run_experiment.py --experiment benchmark` |
| 80/20 first/second-order decomposition (Section 5) | `python scripts/run_experiment.py --experiment decomposition --dataset freesolv` |
| Representation vs. architecture ablation | `python scripts/run_experiment.py --experiment rep_vs_arch` |
| 10-seed statistical validation | `python scripts/run_10seed_validation.py` |
| Feature attribution / mutual information | `python scripts/run_feature_attribution.py` and `python scripts/run_mi_analysis.py` |
| Scaffold-split robustness | `python scripts/run_scaffold_splits.py` |
| Fingerprint + XGBoost baseline | `python scripts/run_fingerprint_baseline.py` |
| SchNet / PaiNN 3D GNN baselines | `python scripts/run_schnet_baseline.py`, `python scripts/run_painn_baseline.py` |

## Project Structure

```
dko-research/
├── dko/
│   ├── models/              # Model implementations
│   │   ├── dko.py           # Distribution Kernel Operator
│   │   ├── attention.py     # Attention-based aggregation
│   │   ├── deepsets.py      # DeepSets baseline
│   │   ├── ensemble_baselines.py  # MFA, MIL, Mean/Boltzmann ensembles
│   │   └── gnn_baselines.py # SchNet, DimeNet++, SphereNet (PyG & simplified)
│   ├── data/                # Data loading and processing
│   │   ├── datasets.py      # Dataset classes
│   │   ├── conformers.py    # Conformer generation
│   │   ├── features.py      # Feature extraction
│   │   └── splits.py        # Data splitting utilities
│   ├── training/            # Training infrastructure
│   │   ├── trainer.py       # Training loop
│   │   ├── evaluator.py     # Evaluation metrics
│   │   └── hyperopt.py      # Hyperparameter optimization
│   ├── experiments/         # Experiment scripts
│   │   ├── main_benchmark.py
│   │   ├── decomposition.py          # 80/20 decomposition study
│   │   ├── sample_efficiency.py      # Data fraction & conformer count
│   │   ├── attention_analysis.py     # Attention visualization & scaling
│   │   ├── representation_vs_architecture.py  # Rep vs arch study
│   │   ├── negative_controls.py      # SCC-based negative controls
│   │   ├── scc_validation.py         # SCC metric validation
│   │   ├── decision_rule.py          # SCC decision rule calibration
│   │   └── sketching.py              # Large ensemble sketching
│   ├── analysis/            # Analysis utilities
│   │   ├── scc.py           # Structural Conformational Complexity
│   │   ├── statistics.py
│   │   └── visualization.py
│   └── utils/               # Utility functions
│       ├── config.py        # Configuration system
│       └── logging_utils.py
├── configs/                 # Configuration files
│   ├── base_config.yaml
│   ├── datasets/            # Per-dataset configs
│   └── models/              # Per-model configs
├── scripts/                 # Entry point scripts
├── tests/                   # Unit tests
└── scripts/                 # Entry point scripts (continued)
```

## Models

### DKO Variants
- **DKO**: Full Distribution Kernel Operator with first and second-order statistics
- **DKOFirstOrder**: First-order only (mean statistics)
- **DKOKernel**: Kernel-based variant
- **DKONoPSD**: Without positive semi-definite constraint

### Aggregation Baselines
- **DeepSets**: Permutation-invariant set function
- **DeepSetsAugmented**: DeepSets with explicit second-order features (for Rep vs Arch study)
- **AttentionPooling**: Multi-head attention aggregation
- **AttentionAugmented**: Attention with explicit second-order features (outer products)
- **MeanFeatureAggregation (MFA)**: Averages features before network
- **MultiInstanceLearning (MIL)**: Instance-level encoding with max/attention pooling
- **MeanEnsemble**: Simple prediction averaging
- **BoltzmannEnsemble**: Energy-weighted prediction averaging
- **SingleConformer**: Lowest-energy conformer only

### GNN Baselines (with conformer aggregation)
- **SchNet/SchNetPyG**: Continuous-filter convolutional network
- **DimeNet++/DimeNetPPPyG**: Directional message passing with spherical harmonics
- **SphereNet**: Spherical message passing
- **3D-Infomax**: Contrastive learning for 3D molecular representations
- **GEM**: Geometry-Enhanced Molecular representation learning

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

If you use this code in your research, please cite the ACM-BCB 2026 paper:

```bibtex
@inproceedings{cheng2026dko,
  title     = {When Does Conformer Geometry Help? Complementarity of 3D Ensemble Statistics
               and 2D Fingerprints for Molecular Property Prediction},
  author    = {Cheng, Bryan and Jin, Austin and Zhang, Jasper},
  booktitle = {Proceedings of the 17th ACM Conference on Bioinformatics,
               Computational Biology, and Health Informatics (ACM-BCB '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {Calabria, Italy},
  url       = {https://github.com/JasperZG/dko}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- RDKit team for molecular informatics tools
- PyTorch Geometric team for GNN implementations
- MoleculeNet for benchmark datasets
