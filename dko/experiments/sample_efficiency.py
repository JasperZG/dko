"""
Sample efficiency experiment for DKO.

Evaluates how DKO and baselines perform with varying training set sizes.
"""

from typing import Dict, List, Optional
import numpy as np
from dko.utils.logging_utils import get_logger

logger = get_logger("sample_efficiency")


def run_sample_efficiency_experiment(
    dataset_name: str,
    fractions: List[float] = [0.1, 0.25, 0.5, 0.75, 1.0],
    models: Optional[List[str]] = None,
    seeds: List[int] = [42, 123, 456],
    config_path: Optional[str] = None,
    output_dir: str = "results/sample_efficiency",
    device: str = "cuda",
) -> Dict:
    """
    Run sample efficiency experiment.

    Trains models on varying fractions of the training data
    to evaluate sample efficiency.

    Args:
        dataset_name: Dataset to use
        fractions: Training data fractions to test
        models: Models to compare
        seeds: Random seeds
        config_path: Path to config
        output_dir: Output directory
        device: Device

    Returns:
        Dictionary mapping fraction -> model -> metrics
    """
    logger.info(f"Running sample efficiency experiment on {dataset_name}")

    results = {}
    for fraction in fractions:
        results[fraction] = {}
        logger.info(f"Training with {fraction*100}% of data")

        # Placeholder implementation
        results[fraction]["dko"] = {"rmse": 0.0, "std": 0.0}
        results[fraction]["single_conformer"] = {"rmse": 0.0, "std": 0.0}

    return results
