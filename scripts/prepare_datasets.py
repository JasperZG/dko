#!/usr/bin/env python
"""
Dataset preparation script for DKO experiments.

This script downloads, preprocesses, and generates conformers
for all molecular datasets.
"""

import argparse
import os
from pathlib import Path
import pickle
from typing import List, Optional

from tqdm import tqdm

from dko.data.datasets import AVAILABLE_DATASETS
from dko.data.conformers import ConformerGenerator
from dko.data.features import FeatureExtractor
from dko.data.splits import get_split
from dko.utils.config import Config
from dko.utils.logging_utils import get_logger, setup_logging


logger = get_logger("prepare_datasets")


def download_dataset(name: str, data_dir: Path) -> Path:
    """
    Download a dataset if not already present.

    Args:
        name: Dataset name
        data_dir: Data directory

    Returns:
        Path to raw data file
    """
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{name}.csv"

    if raw_path.exists():
        logger.info(f"Dataset {name} already exists at {raw_path}")
        return raw_path

    # Dataset download URLs (placeholder - actual URLs would be here)
    urls = {
        "bace": "https://moleculenet.org/datasets/bace",
        "freesolv": "https://moleculenet.org/datasets/freesolv",
        "esol": "https://moleculenet.org/datasets/esol",
        "lipo": "https://moleculenet.org/datasets/lipo",
        "bbbp": "https://moleculenet.org/datasets/bbbp",
        "tox21": "https://moleculenet.org/datasets/tox21",
        "herg": "https://github.com/cheminfo/herg",
        "cyp3a4": "https://pubchem.ncbi.nlm.nih.gov/bioassay/1851",
        "pdbbind": "http://www.pdbbind.org.cn/",
        "qm9_homo": "https://figshare.com/collections/qm9",
        "qm9_gap": "https://figshare.com/collections/qm9",
        "qm9_polar": "https://figshare.com/collections/qm9",
    }

    logger.warning(
        f"Dataset {name} not found. Please download from: {urls.get(name, 'Unknown')}"
    )
    logger.info(f"Place the CSV file at: {raw_path}")

    # Create placeholder file for demonstration
    # In production, implement actual download logic
    logger.info(f"Creating placeholder for {name}")

    return raw_path


def generate_conformers_for_dataset(
    name: str,
    data_dir: Path,
    config: Config,
    splits: List[str] = ["train", "val", "test"],
) -> None:
    """
    Generate conformers for a dataset.

    Args:
        name: Dataset name
        data_dir: Data directory
        config: Configuration
        splits: Splits to process
    """
    import pandas as pd

    raw_path = data_dir / "raw" / f"{name}.csv"

    if not raw_path.exists():
        logger.error(f"Raw data not found: {raw_path}")
        return

    # Load data
    df = pd.read_csv(raw_path)
    smiles_col = "smiles"

    if smiles_col not in df.columns:
        logger.error(f"SMILES column not found in {name}")
        return

    smiles_list = df[smiles_col].tolist()

    # Get labels for stratification
    label_col = None
    for col in ["pIC50", "expt", "activity", "label", "exp", "homo", "gap", "alpha"]:
        if col in df.columns:
            label_col = col
            break

    labels = df[label_col].values if label_col else None

    # Get split indices
    train_idx, val_idx, test_idx = get_split(
        smiles_list,
        labels,
        method=config.get("data.splitting.method", "scaffold"),
        seed=config.get("data.splitting.seed", 42),
    )

    split_indices = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    # Initialize generators
    conf_config = config.get("data.conformer_generation", {})
    conformer_generator = ConformerGenerator(**conf_config)
    feature_extractor = FeatureExtractor()

    # Process each split
    conformers_dir = data_dir / "conformers" / name
    conformers_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        if split not in split_indices:
            continue

        split_path = conformers_dir / f"{split}.pkl"

        if split_path.exists():
            logger.info(f"Conformers for {name}/{split} already exist")
            continue

        indices = split_indices[split]
        split_smiles = [smiles_list[i] for i in indices]

        logger.info(f"Generating conformers for {name}/{split} ({len(split_smiles)} molecules)")

        all_features = []
        all_energies = []
        failed = 0

        for smiles in tqdm(split_smiles, desc=f"{name}/{split}"):
            conformers, energies = conformer_generator.generate(smiles)

            if conformers:
                features = feature_extractor.extract(conformers)
                all_features.append(features)
                all_energies.append(energies if energies else [0.0] * len(conformers))
            else:
                # Fallback for failed molecules
                all_features.append(None)
                all_energies.append(None)
                failed += 1

        # Save
        data = {
            "features": all_features,
            "energies": all_energies,
            "smiles": split_smiles,
            "indices": indices,
        }

        with open(split_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(
            f"Saved {name}/{split}: {len(split_smiles) - failed} molecules, "
            f"{failed} failed"
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Prepare datasets for DKO experiments")

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Datasets to prepare. Available: {AVAILABLE_DATASETS}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Prepare all available datasets",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration file path",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download, don't generate conformers",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(Path(args.data_dir) / "logs", name="prepare_datasets")

    # Load config
    if args.config:
        config = Config(base_config_path=args.config)
    else:
        config = Config()

    # Determine datasets to process
    if args.all:
        datasets = AVAILABLE_DATASETS
    elif args.datasets:
        datasets = args.datasets
    else:
        parser.error("Please specify --datasets or --all")

    data_dir = Path(args.data_dir)

    logger.info(f"Preparing datasets: {datasets}")

    for name in datasets:
        logger.info(f"=== Processing {name} ===")

        # Download
        download_dataset(name, data_dir)

        # Generate conformers
        if not args.download_only:
            generate_conformers_for_dataset(name, data_dir, config, args.splits)

    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main()
