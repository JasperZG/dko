"""
DKO Models Module

Contains implementations of:
- DKO (Distribution Kernel Operators)
- Attention-based aggregation
- DeepSets
- GNN baselines (SchNet, DimeNet++, SphereNet)
- Ensemble baselines (Mean, Boltzmann, MFA, MIL)
"""

from dko.models.dko import DKO, DKOKernel, DKOFirstOrder, DKONoPSD
from dko.models.attention import (
    AttentionAggregation,
    MultiHeadAttention,
    AttentionPoolingBaseline,
)
from dko.models.deepsets import DeepSets, DeepSetsBaseline
from dko.models.gnn_baselines import (
    SchNet,
    DimeNetPP,
    SphereNet,
    get_gnn,
    GNNWithConformerAggregation,
    ConformerAggregation,
    HAS_PYG,
)

# Import PyG versions if available
if HAS_PYG:
    from dko.models.gnn_baselines import SchNetPyG, DimeNetPPPyG, GNNEnsembleWrapper
from dko.models.ensemble_baselines import (
    SingleConformer,
    SingleConformerBaseline,
    MeanFeatureAggregation,
    MultiInstanceLearning,
    MILBaseline,
    MeanEnsemble,
    BoltzmannEnsemble,
    LearnedWeightEnsemble,
)

__all__ = [
    # DKO
    "DKO",
    "DKOKernel",
    "DKOFirstOrder",
    "DKONoPSD",
    # Attention
    "AttentionAggregation",
    "MultiHeadAttention",
    "AttentionPoolingBaseline",
    # DeepSets
    "DeepSets",
    "DeepSetsBaseline",
    # GNN baselines
    "SchNet",
    "DimeNetPP",
    "SphereNet",
    # Ensemble baselines
    "SingleConformer",
    "SingleConformerBaseline",
    "MeanFeatureAggregation",
    "MultiInstanceLearning",
    "MILBaseline",
    "MeanEnsemble",
    "BoltzmannEnsemble",
    "LearnedWeightEnsemble",
]
