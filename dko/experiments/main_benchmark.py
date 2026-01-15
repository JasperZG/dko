"""
Main benchmark experiment for DKO.

This script runs the main comparative benchmark across all datasets
and models, evaluating DKO against baselines.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np
import torch

from dko.utils.config import Config, create_experiment_config
from dko.utils.logging_utils import ExperimentTracker, get_logger
from dko.data.datasets import create_dataloaders, AVAILABLE_DATASETS
from dko.models import (
    DKO,
    DKOFirstOrder,
    AttentionAggregation,
    AttentionAugmented,
    DeepSets,
    DeepSetsAugmented,
    SingleConformer,
    MeanEnsemble,
    BoltzmannEnsemble,
    MeanFeatureAggregation,
    MultiInstanceLearning,
    SchNet,
    DimeNetPP,
    SphereNet,
    ThreeDInfomax,
    GEM,
)
from dko.training.trainer import train_model
from dko.training.evaluator import Evaluator


logger = get_logger("benchmark")


# Model registry
MODEL_REGISTRY = {
    # DKO variants
    "dko": DKO,
    "dko_first_order": DKOFirstOrder,
    # Attention-based
    "attention": AttentionAggregation,
    "attention_augmented": AttentionAugmented,
    # DeepSets-based
    "deepsets": DeepSets,
    "deepsets_augmented": DeepSetsAugmented,
    # Ensemble baselines
    "single_conformer": SingleConformer,
    "mean_ensemble": MeanEnsemble,
    "boltzmann_ensemble": BoltzmannEnsemble,
    "mfa": MeanFeatureAggregation,
    "mil": MultiInstanceLearning,
    # GNN baselines
    "schnet": SchNet,
    "dimenet++": DimeNetPP,
    "dimenetpp": DimeNetPP,  # Alias without special chars
    "spherenet": SphereNet,
    "3d_infomax": ThreeDInfomax,
    "3dinfomax": ThreeDInfomax,  # Alias without underscore
    "gem": GEM,
}


def run_single_experiment(
    dataset_name: str,
    model_name: str,
    config: Config,
    seed: int = 42,
    device: str = "cuda",
) -> Dict:
    """
    Run a single experiment (one dataset, one model, one seed).

    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        config: Configuration
        seed: Random seed
        device: Device to use

    Returns:
        Dictionary of results
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    logger.info(f"Loading dataset: {dataset_name}")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name,
        batch_size=config.get("training.batch_size", 32),
        num_workers=config.get("project.num_workers", 4),
    )

    # Get feature dimension from data
    sample = next(iter(train_loader))
    feature_dim = sample["features"].shape[-1]

    # Create model
    logger.info(f"Creating model: {model_name}")
    model_class = MODEL_REGISTRY[model_name]
    model_config = config.get("model", {})
    model_config["feature_dim"] = feature_dim

    # Set number of outputs
    if dataset_name in ["tox21"]:
        model_config["num_outputs"] = 12  # Multi-task
    else:
        model_config["num_outputs"] = 1

    model = model_class(**model_config)

    # Determine task type
    task_type = "classification" if dataset_name in ["herg", "cyp3a4", "tox21", "bbbp"] else "regression"

    # Training config
    training_config = {
        "optimizer": config.get("training.optimizer", "AdamW"),
        "base_learning_rate": config.get("training.base_learning_rate", 1e-4),
        "weight_decay": config.get("training.weight_decay", 1e-5),
        "max_epochs": config.get("training.max_epochs", 300),
        "early_stopping_patience": config.get("training.early_stopping_patience", 30),
        "gradient_clip_max_norm": config.get("training.gradient_clip_max_norm", 1.0),
        "mixed_precision": config.get("training.mixed_precision", True),
        "task_type": task_type,
        "scheduler": config.get("training.scheduler", {}),
    }

    # Train
    experiment_name = f"{dataset_name}_{model_name}_seed{seed}"
    model, train_results = train_model(
        model,
        train_loader,
        val_loader,
        training_config,
        device=device,
        experiment_name=experiment_name,
    )

    # Evaluate on test set
    evaluator = Evaluator(task_type=task_type)
    test_metrics = evaluator.evaluate(model, test_loader, device)

    logger.info(f"Test metrics: {test_metrics}")

    return {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "test_metrics": test_metrics,
        "train_results": train_results,
    }


def run_main_benchmark(
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    seeds: List[int] = [42, 123, 456],
    config_path: Optional[str] = None,
    output_dir: str = "results/benchmark",
    device: str = "cuda",
) -> Dict:
    """
    Run the main benchmark across datasets and models.

    Args:
        datasets: List of datasets (default: all)
        models: List of models (default: all)
        seeds: Random seeds for multiple runs
        config_path: Path to config file
        output_dir: Output directory for results
        device: Device to use

    Returns:
        Dictionary of all results
    """
    datasets = datasets or AVAILABLE_DATASETS
    models = models or list(MODEL_REGISTRY.keys())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if config_path:
        config = Config(base_config_path=config_path)
    else:
        config = Config()

    # Run experiments
    all_results = {}

    for dataset in datasets:
        logger.info(f"=== Dataset: {dataset} ===")
        all_results[dataset] = {}

        for model_name in models:
            logger.info(f"--- Model: {model_name} ---")
            all_results[dataset][model_name] = []

            for seed in seeds:
                try:
                    result = run_single_experiment(
                        dataset, model_name, config, seed, device
                    )
                    all_results[dataset][model_name].append(result)
                except Exception as e:
                    logger.error(f"Error in {dataset}/{model_name}/seed{seed}: {e}")
                    all_results[dataset][model_name].append({
                        "error": str(e),
                        "seed": seed,
                    })

    # Aggregate results
    summary = aggregate_results(all_results)

    # Save results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    summary_path = output_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")

    return {"results": all_results, "summary": summary}


def aggregate_results(all_results: Dict) -> Dict:
    """
    Aggregate results across seeds.

    Args:
        all_results: Raw results dictionary

    Returns:
        Summary statistics
    """
    summary = {}

    for dataset, model_results in all_results.items():
        summary[dataset] = {}

        for model_name, seed_results in model_results.items():
            # Extract metric values
            metrics_per_seed = {}

            for result in seed_results:
                if "error" in result:
                    continue

                for metric, value in result.get("test_metrics", {}).items():
                    if metric not in metrics_per_seed:
                        metrics_per_seed[metric] = []
                    metrics_per_seed[metric].append(value)

            # Compute statistics
            summary[dataset][model_name] = {}
            for metric, values in metrics_per_seed.items():
                if values:
                    summary[dataset][model_name][metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "n": len(values),
                    }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DKO benchmark")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to evaluate")
    parser.add_argument("--models", nargs="+", default=None, help="Models to evaluate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--output-dir", type=str, default="results/benchmark")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    run_main_benchmark(
        datasets=args.datasets,
        models=args.models,
        seeds=args.seeds,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
    )
