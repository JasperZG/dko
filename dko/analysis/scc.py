"""
Statistical Consistency Check (SCC) utilities.

This module provides tools for validating that experimental results
are statistically meaningful and consistent.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats


class StatisticalConsistencyChecker:
    """
    Statistical Consistency Checker for model comparison.

    Provides methods for:
    - Testing significance of improvements
    - Computing effect sizes
    - Validating result consistency across seeds
    """

    def __init__(
        self,
        alpha: float = 0.05,
        min_seeds: int = 3,
        effect_size_threshold: float = 0.2,
    ):
        """
        Initialize SCC.

        Args:
            alpha: Significance level
            min_seeds: Minimum number of seeds for valid comparison
            effect_size_threshold: Minimum effect size for practical significance
        """
        self.alpha = alpha
        self.min_seeds = min_seeds
        self.effect_size_threshold = effect_size_threshold

    def check_significance(
        self,
        model1_values: List[float],
        model2_values: List[float],
        paired: bool = True,
    ) -> Dict:
        """
        Check statistical significance of difference.

        Args:
            model1_values: Values from model 1 across seeds
            model2_values: Values from model 2 across seeds
            paired: Whether to use paired test

        Returns:
            Dictionary with test results
        """
        model1_values = np.array(model1_values)
        model2_values = np.array(model2_values)

        n = len(model1_values)

        if n < self.min_seeds:
            return {
                "valid": False,
                "reason": f"Insufficient seeds ({n} < {self.min_seeds})",
            }

        # Compute means and difference
        mean1 = np.mean(model1_values)
        mean2 = np.mean(model2_values)
        diff = model1_values - model2_values

        # Statistical test
        if paired:
            t_stat, p_value = stats.ttest_rel(model1_values, model2_values)
        else:
            t_stat, p_value = stats.ttest_ind(model1_values, model2_values)

        # Effect size (Cohen's d for paired)
        if paired:
            cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
        else:
            pooled_std = np.sqrt(
                (np.var(model1_values) + np.var(model2_values)) / 2
            )
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

        # Confidence interval
        ci_margin = stats.t.ppf(1 - self.alpha / 2, df=n - 1) * np.std(diff, ddof=1) / np.sqrt(n)

        return {
            "valid": True,
            "mean_model1": mean1,
            "mean_model2": mean2,
            "mean_difference": np.mean(diff),
            "std_difference": np.std(diff, ddof=1),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "cohens_d": cohens_d,
            "effect_size": self._interpret_effect_size(cohens_d),
            "practically_significant": abs(cohens_d) >= self.effect_size_threshold,
            "ci_lower": np.mean(diff) - ci_margin,
            "ci_upper": np.mean(diff) + ci_margin,
            "n_seeds": n,
        }

    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def check_consistency(
        self,
        values: List[float],
        expected_cv: float = 0.1,
    ) -> Dict:
        """
        Check if results are consistent across seeds.

        Args:
            values: Metric values across seeds
            expected_cv: Expected coefficient of variation

        Returns:
            Consistency check results
        """
        values = np.array(values)
        n = len(values)

        if n < 2:
            return {"valid": False, "reason": "Need at least 2 values"}

        mean = np.mean(values)
        std = np.std(values, ddof=1)
        cv = std / abs(mean) if mean != 0 else float("inf")

        # Check for outliers using IQR
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        outliers = np.sum((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr))

        return {
            "valid": True,
            "mean": mean,
            "std": std,
            "cv": cv,
            "consistent": cv <= expected_cv,
            "n_outliers": int(outliers),
            "range": np.max(values) - np.min(values),
        }

    def validate_improvement(
        self,
        baseline_values: List[float],
        model_values: List[float],
        expected_improvement: float,
        lower_is_better: bool = True,
    ) -> Dict:
        """
        Validate that improvement meets expectations.

        Args:
            baseline_values: Baseline metric values
            model_values: Model metric values
            expected_improvement: Expected improvement percentage
            lower_is_better: Whether lower metric values are better

        Returns:
            Validation results
        """
        baseline_values = np.array(baseline_values)
        model_values = np.array(model_values)

        baseline_mean = np.mean(baseline_values)
        model_mean = np.mean(model_values)

        # Compute actual improvement
        if lower_is_better:
            actual_improvement = (baseline_mean - model_mean) / baseline_mean * 100
        else:
            actual_improvement = (model_mean - baseline_mean) / baseline_mean * 100

        # Check significance
        sig_result = self.check_significance(
            baseline_values if lower_is_better else model_values,
            model_values if lower_is_better else baseline_values,
        )

        # Check if improvement meets expectation (with some margin)
        margin = 0.2  # 20% margin
        meets_expectation = actual_improvement >= expected_improvement * (1 - margin)

        return {
            "expected_improvement": expected_improvement,
            "actual_improvement": actual_improvement,
            "meets_expectation": meets_expectation,
            "improvement_ratio": actual_improvement / expected_improvement if expected_improvement > 0 else float("inf"),
            **sig_result,
        }


def compute_scc_scores(
    results: Dict[str, Dict[str, List[float]]],
    baseline_key: str = "single_conformer",
    metric_key: str = "rmse",
) -> Dict[str, Dict]:
    """
    Compute SCC scores for all model comparisons.

    Args:
        results: Dictionary of model name -> metric name -> values
        baseline_key: Key for baseline model
        metric_key: Metric to compare

    Returns:
        Dictionary of model name -> SCC results
    """
    scc = StatisticalConsistencyChecker()
    scores = {}

    baseline_values = results.get(baseline_key, {}).get(metric_key, [])

    if not baseline_values:
        return {"error": "Baseline values not found"}

    for model_name, metrics in results.items():
        if model_name == baseline_key:
            continue

        model_values = metrics.get(metric_key, [])
        if model_values:
            scores[model_name] = scc.check_significance(baseline_values, model_values)

    return scores


def validate_scc(
    results: Dict,
    expected_improvements: Dict[str, float],
    metric_key: str = "rmse",
) -> Dict:
    """
    Validate results against expected improvements.

    Args:
        results: Experimental results
        expected_improvements: Expected improvement per dataset
        metric_key: Metric to validate

    Returns:
        Validation summary
    """
    scc = StatisticalConsistencyChecker()
    validation = {}

    for dataset, expected_imp in expected_improvements.items():
        dataset_results = results.get(dataset, {})

        baseline = dataset_results.get("single_conformer", {}).get(metric_key, [])
        model = dataset_results.get("dko", {}).get(metric_key, [])

        if baseline and model:
            validation[dataset] = scc.validate_improvement(
                baseline, model, expected_imp, lower_is_better=True
            )
        else:
            validation[dataset] = {"valid": False, "reason": "Missing data"}

    return validation
