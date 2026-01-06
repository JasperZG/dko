"""
DKO Data Module

Contains utilities for:
- Dataset loading and processing
- Conformer generation
- Feature extraction
- Data splitting
"""

from dko.data.datasets import (
    MoleculeDataset,
    ConformerDataset,
    get_dataset,
    AVAILABLE_DATASETS,
)
from dko.data.conformers import (
    ConformerGenerator,
    generate_conformers,
    compute_boltzmann_weights,
)
from dko.data.features import (
    FeatureExtractor,
    compute_pairwise_distances,
    compute_bond_angles,
    compute_torsion_angles,
)
from dko.data.splits import (
    scaffold_split,
    random_split,
    stratified_split,
    get_split,
)

__all__ = [
    # Datasets
    "MoleculeDataset",
    "ConformerDataset",
    "get_dataset",
    "AVAILABLE_DATASETS",
    # Conformers
    "ConformerGenerator",
    "generate_conformers",
    "compute_boltzmann_weights",
    # Features
    "FeatureExtractor",
    "compute_pairwise_distances",
    "compute_bond_angles",
    "compute_torsion_angles",
    # Splits
    "scaffold_split",
    "random_split",
    "stratified_split",
    "get_split",
]
