"""
DKO Analysis Module

Contains utilities for:
- Statistical Consistency Checks (SCC)
- Statistical analysis and multiple comparison corrections
- Visualization
"""

from dko.analysis.scc import (
    StatisticalConsistencyChecker,
    compute_scc_scores,
    validate_scc,
)
from dko.analysis.statistics import (
    compute_confidence_intervals,
    perform_significance_tests,
    bootstrap_statistics,
    compute_effect_size,
    bonferroni_correction,
    holm_bonferroni_correction,
    benjamini_hochberg_correction,
    multiple_comparison_summary,
    paired_comparisons_with_correction,
)
from dko.analysis.visualization import (
    plot_learning_curves,
    plot_attention_weights,
    plot_conformer_distributions,
    plot_performance_comparison,
    create_summary_table,
)

__all__ = [
    # SCC
    "StatisticalConsistencyChecker",
    "compute_scc_scores",
    "validate_scc",
    # Statistics
    "compute_confidence_intervals",
    "perform_significance_tests",
    "bootstrap_statistics",
    "compute_effect_size",
    # Multiple comparison corrections
    "bonferroni_correction",
    "holm_bonferroni_correction",
    "benjamini_hochberg_correction",
    "multiple_comparison_summary",
    "paired_comparisons_with_correction",
    # Visualization
    "plot_learning_curves",
    "plot_attention_weights",
    "plot_conformer_distributions",
    "plot_performance_comparison",
    "create_summary_table",
]
