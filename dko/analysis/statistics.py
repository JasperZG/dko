"""
Statistical analysis utilities.

This module provides functions for computing confidence intervals,
significance tests, and bootstrap statistics.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats


def compute_confidence_intervals(
    values: Union[List[float], np.ndarray],
    confidence_level: float = 0.95,
    method: str = "t",
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for a set of values.

    Args:
        values: Sample values
        confidence_level: Confidence level (default 0.95)
        method: 't' for t-distribution, 'bootstrap' for bootstrap

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)

    if n < 2:
        return mean, mean, mean

    if method == "t":
        std = np.std(values, ddof=1)
        alpha = 1 - confidence_level
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin = t_crit * std / np.sqrt(n)
        return mean, mean - margin, mean + margin

    elif method == "bootstrap":
        lower, upper = bootstrap_ci(values, confidence_level=confidence_level)
        return mean, lower, upper

    else:
        raise ValueError(f"Unknown method: {method}")


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        values: Sample values
        statistic: Statistic function (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    values = np.array(values)
    n = len(values)

    # Generate bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Compute percentiles
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return lower, upper


def bootstrap_statistics(
    values: Union[List[float], np.ndarray],
    statistics: List[str] = ["mean", "std", "median"],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
) -> Dict[str, Dict]:
    """
    Compute multiple bootstrap statistics.

    Args:
        values: Sample values
        statistics: List of statistics to compute
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level

    Returns:
        Dictionary mapping statistic name to (value, ci_lower, ci_upper)
    """
    values = np.array(values)

    stat_funcs = {
        "mean": np.mean,
        "std": lambda x: np.std(x, ddof=1),
        "median": np.median,
        "min": np.min,
        "max": np.max,
    }

    results = {}

    for stat_name in statistics:
        if stat_name not in stat_funcs:
            continue

        func = stat_funcs[stat_name]
        value = func(values)
        lower, upper = bootstrap_ci(values, func, n_bootstrap, confidence_level)

        results[stat_name] = {
            "value": value,
            "ci_lower": lower,
            "ci_upper": upper,
        }

    return results


def perform_significance_tests(
    model1_values: Union[List[float], np.ndarray],
    model2_values: Union[List[float], np.ndarray],
    tests: List[str] = ["paired_t", "wilcoxon"],
) -> Dict[str, Dict]:
    """
    Perform multiple significance tests.

    Args:
        model1_values: Values from model 1
        model2_values: Values from model 2
        tests: List of test names

    Returns:
        Dictionary mapping test name to results
    """
    model1_values = np.array(model1_values)
    model2_values = np.array(model2_values)

    results = {}

    for test_name in tests:
        if test_name == "paired_t":
            stat, p_value = stats.ttest_rel(model1_values, model2_values)
            results[test_name] = {
                "statistic": stat,
                "p_value": p_value,
                "significant_05": p_value < 0.05,
                "significant_01": p_value < 0.01,
            }

        elif test_name == "wilcoxon":
            try:
                stat, p_value = stats.wilcoxon(model1_values, model2_values)
                results[test_name] = {
                    "statistic": stat,
                    "p_value": p_value,
                    "significant_05": p_value < 0.05,
                    "significant_01": p_value < 0.01,
                }
            except ValueError as e:
                results[test_name] = {"error": str(e)}

        elif test_name == "mann_whitney":
            stat, p_value = stats.mannwhitneyu(model1_values, model2_values)
            results[test_name] = {
                "statistic": stat,
                "p_value": p_value,
                "significant_05": p_value < 0.05,
            }

        elif test_name == "bootstrap":
            stat, p_value = _bootstrap_test(model1_values, model2_values)
            results[test_name] = {
                "statistic": stat,
                "p_value": p_value,
                "significant_05": p_value < 0.05,
            }

    return results


def _bootstrap_test(
    values1: np.ndarray,
    values2: np.ndarray,
    n_bootstrap: int = 10000,
) -> Tuple[float, float]:
    """Permutation-based bootstrap test."""
    observed_diff = np.mean(values1) - np.mean(values2)

    combined = np.concatenate([values1, values2])
    n1 = len(values1)

    count = 0
    for _ in range(n_bootstrap):
        np.random.shuffle(combined)
        boot_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        if abs(boot_diff) >= abs(observed_diff):
            count += 1

    p_value = (count + 1) / (n_bootstrap + 1)
    return observed_diff, p_value


def compute_effect_size(
    model1_values: Union[List[float], np.ndarray],
    model2_values: Union[List[float], np.ndarray],
    paired: bool = True,
) -> Dict:
    """
    Compute various effect size measures.

    Args:
        model1_values: Values from model 1
        model2_values: Values from model 2
        paired: Whether the comparison is paired

    Returns:
        Dictionary with effect size measures
    """
    model1_values = np.array(model1_values)
    model2_values = np.array(model2_values)

    mean1 = np.mean(model1_values)
    mean2 = np.mean(model2_values)
    diff = model1_values - model2_values

    results = {
        "mean_difference": np.mean(diff),
        "relative_difference": (mean1 - mean2) / mean2 if mean2 != 0 else float("inf"),
    }

    # Cohen's d
    if paired:
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
    else:
        pooled_std = np.sqrt(
            (np.var(model1_values, ddof=1) + np.var(model2_values, ddof=1)) / 2
        )
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

    results["cohens_d"] = cohens_d
    results["cohens_d_interpretation"] = _interpret_cohens_d(cohens_d)

    # Hedge's g (corrected for small samples)
    n = len(model1_values)
    correction = 1 - 3 / (4 * (2 * n - 2) - 1)
    results["hedges_g"] = cohens_d * correction

    return results


def _interpret_cohens_d(d: float) -> str:
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
