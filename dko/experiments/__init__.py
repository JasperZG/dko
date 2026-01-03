"""
DKO Experiments Module

Contains experiment scripts for:
- Main benchmark evaluation
- Ablation studies
- Sample efficiency analysis
- Attention visualization
- Sketching experiments
- Statistical consistency checks
"""

from dko.experiments.main_benchmark import run_main_benchmark
from dko.experiments.decomposition import run_decomposition_study
from dko.experiments.sample_efficiency import run_sample_efficiency_experiment
from dko.experiments.attention_analysis import run_attention_analysis
from dko.experiments.sketching import run_sketching_experiment
from dko.experiments.scc_validation import run_scc_validation
from dko.experiments.decision_rule import run_decision_rule_experiment

__all__ = [
    "run_main_benchmark",
    "run_decomposition_study",
    "run_sample_efficiency_experiment",
    "run_attention_analysis",
    "run_sketching_experiment",
    "run_scc_validation",
    "run_decision_rule_experiment",
]
