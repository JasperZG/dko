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


class StructuralConformationalComplexity:
    """
    Structural Conformational Complexity (SCC) metric.

    SCC measures conformational variability as the total variance of geometric
    features across the conformer ensemble. This metric is used to predict
    a priori whether ensemble methods will provide benefit.

    From the research plan:
    - SCC = sum of variance of each geometric feature across conformers
    - Low SCC → single-conformer methods suffice
    - High SCC → ensemble methods may help

    Theorem 1 (SCC Upper Bound):
    For any Lipschitz-continuous property function, the ensemble advantage
    is bounded by: L * sqrt(SCC), where L is the Lipschitz constant.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        use_boltzmann_weights: bool = True,
    ):
        """
        Initialize SCC calculator.

        Args:
            threshold: SCC threshold for decision rule (calibrated on validation)
            use_boltzmann_weights: Whether to use Boltzmann weights for variance calc
        """
        self.threshold = threshold
        self.use_boltzmann_weights = use_boltzmann_weights
        self._calibrated = False

    def compute(
        self,
        features_list: List[np.ndarray],
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute SCC as total variance of geometric features.

        SCC = Σ_i Var(f_i) = Σ_i E[(f_i - μ_i)²]

        where f_i is the i-th geometric feature and expectation is over conformers.

        Args:
            features_list: List of feature vectors, one per conformer
            weights: Optional Boltzmann weights (uniform if None)

        Returns:
            SCC score (sum of feature variances)
        """
        if len(features_list) < 2:
            return 0.0

        # Handle variable-length feature vectors by padding
        max_len = max(len(f) for f in features_list)
        padded_features = []
        for f in features_list:
            if len(f) < max_len:
                padded = np.pad(f, (0, max_len - len(f)), mode='constant')
                padded_features.append(padded)
            else:
                padded_features.append(f)

        features = np.stack(padded_features, axis=0)  # (n_conformers, feature_dim)
        n_conformers, feature_dim = features.shape

        # Handle weights
        if weights is None or not self.use_boltzmann_weights:
            weights = np.ones(n_conformers) / n_conformers
        else:
            weights = np.asarray(weights)
            weights = weights / weights.sum()

        # Weighted mean: μ_i = Σ_j w_j * f_ji
        mean = np.sum(weights[:, np.newaxis] * features, axis=0)

        # Weighted variance for each feature: Var(f_i) = Σ_j w_j * (f_ji - μ_i)²
        centered = features - mean[np.newaxis, :]
        variances = np.sum(weights[:, np.newaxis] * centered ** 2, axis=0)

        # SCC = total variance = sum of all feature variances
        scc = float(np.sum(variances))

        return scc

    def compute_from_ensemble(
        self,
        mol,
        conformer_ids: Optional[List[int]] = None,
        weights: Optional[np.ndarray] = None,
        extractor=None,
    ) -> float:
        """
        Compute SCC directly from RDKit molecule with conformers.

        Args:
            mol: RDKit Mol object with conformers
            conformer_ids: Conformer IDs to use (None for all)
            weights: Boltzmann weights
            extractor: GeometricFeatureExtractor instance

        Returns:
            SCC score
        """
        from dko.data.features import GeometricFeatureExtractor

        if extractor is None:
            extractor = GeometricFeatureExtractor()

        if conformer_ids is None:
            conformer_ids = list(range(mol.GetNumConformers()))

        if len(conformer_ids) < 2:
            return 0.0

        # Extract features for all conformers
        features_list = []
        for conf_id in conformer_ids:
            geo_features = extractor.extract(mol, conf_id)
            features_list.append(geo_features.to_flat_vector())

        return self.compute(features_list, weights)

    def predict_ensemble_benefit(self, scc: float) -> bool:
        """
        Decision rule: should we use ensemble methods?

        Args:
            scc: SCC score for a molecule/dataset

        Returns:
            True if ensemble methods recommended, False otherwise
        """
        return scc > self.threshold

    def calibrate_threshold(
        self,
        scc_values: List[float],
        ensemble_advantages: List[float],
        target_accuracy: float = 0.85,
    ) -> float:
        """
        Calibrate the SCC threshold to maximize decision accuracy.

        Args:
            scc_values: SCC scores for validation molecules
            ensemble_advantages: Actual ensemble vs single-conformer improvement
            target_accuracy: Target decision accuracy

        Returns:
            Calibrated threshold
        """
        scc_values = np.array(scc_values)
        advantages = np.array(ensemble_advantages)

        # True labels: ensemble helps if advantage > 0
        true_labels = advantages > 0

        # Search for optimal threshold
        best_threshold = 0.0
        best_accuracy = 0.0

        # Try percentiles as candidate thresholds
        for percentile in range(5, 96, 5):
            threshold = np.percentile(scc_values, percentile)
            predictions = scc_values > threshold
            accuracy = np.mean(predictions == true_labels)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        self.threshold = best_threshold
        self._calibrated = True

        return best_threshold

    def compute_regret(
        self,
        scc_values: List[float],
        ensemble_advantages: List[float],
    ) -> Dict:
        """
        Compute regret from following SCC decision rule vs oracle.

        Regret = performance loss from following SCC rule instead of
        always choosing the better method (oracle).

        Args:
            scc_values: SCC scores
            ensemble_advantages: Actual improvements from ensemble

        Returns:
            Dict with regret statistics
        """
        scc_values = np.array(scc_values)
        advantages = np.array(ensemble_advantages)

        # Predictions from SCC rule
        predictions = scc_values > self.threshold

        # Oracle always chooses correctly
        oracle_choices = advantages > 0

        # Compute regret
        # When we recommend single but ensemble was better: we lose the advantage
        # When we recommend ensemble but single was better: we lose |advantage|
        regret = np.zeros_like(advantages)

        # False negatives (missed ensemble benefit)
        fn_mask = (~predictions) & oracle_choices
        regret[fn_mask] = advantages[fn_mask]

        # False positives (unnecessary ensemble)
        fp_mask = predictions & (~oracle_choices)
        regret[fp_mask] = -advantages[fp_mask]  # advantage is negative here

        return {
            "mean_regret": float(np.mean(regret)),
            "max_regret": float(np.max(regret)),
            "total_regret": float(np.sum(regret)),
            "accuracy": float(np.mean(predictions == oracle_choices)),
            "false_positive_rate": float(np.mean(fp_mask)),
            "false_negative_rate": float(np.mean(fn_mask)),
            "threshold": self.threshold,
        }


def compute_dataset_scc(
    smiles_list: List[str],
    n_conformers: int = 50,
    show_progress: bool = True,
) -> Dict:
    """
    Compute SCC statistics for a dataset.

    Args:
        smiles_list: List of SMILES strings
        n_conformers: Number of conformers per molecule
        show_progress: Whether to show progress bar

    Returns:
        Dict with SCC statistics
    """
    from dko.data.conformers import ConformerGenerator
    from dko.data.features import GeometricFeatureExtractor

    generator = ConformerGenerator(max_conformers=n_conformers)
    extractor = GeometricFeatureExtractor()
    scc_calc = StructuralConformationalComplexity()

    scc_values = []

    iterator = smiles_list
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(smiles_list, desc="Computing SCC")
        except ImportError:
            pass

    for smiles in iterator:
        try:
            ensemble = generator.generate_from_smiles(smiles)
            if ensemble.n_conformers >= 2:
                scc = scc_calc.compute_from_ensemble(
                    ensemble.mol,
                    ensemble.conformer_ids,
                    ensemble.boltzmann_weights,
                    extractor,
                )
                scc_values.append(scc)
        except Exception:
            continue

    scc_array = np.array(scc_values)

    return {
        "mean": float(np.mean(scc_array)) if len(scc_array) > 0 else 0.0,
        "std": float(np.std(scc_array)) if len(scc_array) > 0 else 0.0,
        "median": float(np.median(scc_array)) if len(scc_array) > 0 else 0.0,
        "min": float(np.min(scc_array)) if len(scc_array) > 0 else 0.0,
        "max": float(np.max(scc_array)) if len(scc_array) > 0 else 0.0,
        "quartiles": {
            "q1": float(np.percentile(scc_array, 25)) if len(scc_array) > 0 else 0.0,
            "q2": float(np.percentile(scc_array, 50)) if len(scc_array) > 0 else 0.0,
            "q3": float(np.percentile(scc_array, 75)) if len(scc_array) > 0 else 0.0,
        },
        "n_molecules": len(scc_values),
        "values": scc_values,
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
