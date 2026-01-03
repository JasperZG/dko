"""
Decision rule experiment for DKO.

Develops a decision rule for when DKO is likely to outperform baselines.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dko.utils.logging_utils import get_logger

logger = get_logger("decision_rule")


def run_decision_rule_experiment(
    all_results: Dict,
    dataset_properties: Dict,
    output_dir: str = "results/decision_rule",
) -> Dict:
    """
    Analyze results to develop decision rules.

    Correlates dataset properties with DKO improvement.

    Args:
        all_results: Results for all datasets
        dataset_properties: Properties of each dataset (conformational dependence, etc.)
        output_dir: Output directory

    Returns:
        Dictionary with decision rule analysis
    """
    logger.info("Analyzing decision rules")

    # Collect improvements and properties
    improvements = []
    properties = []

    for dataset, results in all_results.items():
        if dataset not in dataset_properties:
            continue

        dko = results.get("dko", {})
        baseline = results.get("single_conformer", {})

        if not dko or not baseline:
            continue

        # Compute improvement
        dko_metric = dko.get("rmse", {}).get("mean", 0)
        baseline_metric = baseline.get("rmse", {}).get("mean", 0)

        if baseline_metric > 0:
            improvement = (baseline_metric - dko_metric) / baseline_metric * 100
            improvements.append(improvement)
            properties.append(dataset_properties[dataset])

    # Analyze correlations
    if improvements:
        improvements = np.array(improvements)

        # Group by property type
        conf_dependent = [imp for imp, prop in zip(improvements, properties)
                         if prop.get("conformationally_dependent", False)]
        not_conf_dependent = [imp for imp, prop in zip(improvements, properties)
                             if not prop.get("conformationally_dependent", False)]

        analysis = {
            "mean_improvement_conf_dependent": np.mean(conf_dependent) if conf_dependent else 0,
            "mean_improvement_not_conf_dependent": np.mean(not_conf_dependent) if not_conf_dependent else 0,
            "n_conf_dependent": len(conf_dependent),
            "n_not_conf_dependent": len(not_conf_dependent),
        }

        # Decision rule
        threshold = 3.0  # 3% improvement threshold
        analysis["decision_rule"] = {
            "rule": "Use DKO when property is conformationally dependent",
            "threshold_improvement": threshold,
            "expected_gain_conf_dependent": analysis["mean_improvement_conf_dependent"],
        }

    else:
        analysis = {"error": "Insufficient data"}

    return analysis
