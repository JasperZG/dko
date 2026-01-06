#!/usr/bin/env python
"""
Dataset preparation script for DKO experiments.

This script downloads, preprocesses, and generates conformers
for all molecular datasets.

Supported datasets:
- MoleculeNet: ESOL, FreeSolv, Lipophilicity, BACE, BBBP, Tox21, HIV, SIDER
- QM9: Electronic properties (HOMO, LUMO, Gap, Polarizability)
- hERG, CYP3A4: ADMET endpoints
"""

import argparse
import os
from pathlib import Path
import pickle
import urllib.request
import gzip
import shutil
from typing import List, Optional, Tuple
import ssl
import certifi

from tqdm import tqdm
import pandas as pd

from dko.data.conformers import ConformerGenerator
from dko.data.features import FeatureExtractor
from dko.data.splits import get_split
from dko.utils.config import Config
from dko.utils.logging_utils import get_logger, setup_logging


logger = get_logger("prepare_datasets")


# =============================================================================
# Dataset Download URLs and Configurations
# =============================================================================

DATASET_CONFIGS = {
    # MoleculeNet datasets (from DeepChem/MoleculeNet)
    "esol": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "filename": "esol.csv",
        "smiles_col": "smiles",
        "target_col": "measured log solubility in mols per litre",
        "task": "regression",
    },
    "freesolv": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
        "filename": "freesolv.csv",
        "smiles_col": "smiles",
        "target_col": "expt",
        "task": "regression",
    },
    "lipophilicity": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        "filename": "lipophilicity.csv",
        "smiles_col": "smiles",
        "target_col": "exp",
        "task": "regression",
    },
    "lipo": {  # Alias
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        "filename": "lipo.csv",
        "smiles_col": "smiles",
        "target_col": "exp",
        "task": "regression",
    },
    "bace": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        "filename": "bace.csv",
        "smiles_col": "mol",
        "target_col": "Class",
        "task": "classification",
    },
    "bbbp": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "filename": "bbbp.csv",
        "smiles_col": "smiles",
        "target_col": "p_np",
        "task": "classification",
    },
    "tox21": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "filename": "tox21.csv",
        "smiles_col": "smiles",
        "target_col": "NR-AR",  # Primary target; has 12 total
        "task": "classification",
        "compressed": True,
    },
    "hiv": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        "filename": "hiv.csv",
        "smiles_col": "smiles",
        "target_col": "HIV_active",
        "task": "classification",
    },
    "sider": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
        "filename": "sider.csv",
        "smiles_col": "smiles",
        "target_col": "Hepatobiliary disorders",
        "task": "classification",
        "compressed": True,
    },
    "clintox": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        "filename": "clintox.csv",
        "smiles_col": "smiles",
        "target_col": "CT_TOX",
        "task": "classification",
        "compressed": True,
    },
    # QM9 datasets
    "qm9": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",
        "filename": "qm9.csv",
        "smiles_col": "smiles",
        "target_col": "homo",
        "task": "regression",
    },
    "qm9_homo": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",
        "filename": "qm9.csv",
        "smiles_col": "smiles",
        "target_col": "homo",
        "task": "regression",
        "subset_size": 10000,
    },
    "qm9_lumo": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",
        "filename": "qm9.csv",
        "smiles_col": "smiles",
        "target_col": "lumo",
        "task": "regression",
        "subset_size": 10000,
    },
    "qm9_gap": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",
        "filename": "qm9.csv",
        "smiles_col": "smiles",
        "target_col": "gap",
        "task": "regression",
        "subset_size": 10000,
    },
    "qm9_polar": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",
        "filename": "qm9.csv",
        "smiles_col": "smiles",
        "target_col": "alpha",
        "task": "regression",
        "subset_size": 10000,
    },
    # hERG - from TDC (Therapeutics Data Commons)
    "herg": {
        "url": "https://dataverse.harvard.edu/api/access/datafile/6180620",
        "filename": "herg.csv",
        "smiles_col": "Drug",
        "target_col": "Y",
        "task": "classification",
        "post_process": "herg",
    },
    # CYP3A4 - from TDC
    "cyp3a4": {
        "url": "https://dataverse.harvard.edu/api/access/datafile/6180617",
        "filename": "cyp3a4.csv",
        "smiles_col": "Drug",
        "target_col": "Y",
        "task": "classification",
        "post_process": "cyp",
    },
}


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, desc: str = None) -> bool:
    """
    Download a file with progress bar.

    Args:
        url: URL to download
        output_path: Output file path
        desc: Description for progress bar

    Returns:
        True if successful
    """
    try:
        # Create SSL context with certifi certificates
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(
                url,
                filename=output_path,
                reporthook=t.update_to,
            )
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        # Try without SSL verification as fallback
        try:
            logger.info("Retrying without SSL verification...")
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ssl_context)
            )
            urllib.request.install_opener(opener)

            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
                urllib.request.urlretrieve(
                    url,
                    filename=output_path,
                    reporthook=t.update_to,
                )
            return True
        except Exception as e2:
            logger.error(f"Download failed (fallback): {e2}")
            return False


def download_dataset(name: str, data_dir: Path) -> Optional[Path]:
    """
    Download a dataset if not already present.

    Args:
        name: Dataset name
        data_dir: Data directory

    Returns:
        Path to raw data file, or None if download failed
    """
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Get config
    if name not in DATASET_CONFIGS:
        logger.error(f"Unknown dataset: {name}")
        logger.info(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        return None

    config = DATASET_CONFIGS[name]
    raw_path = raw_dir / config["filename"]

    # Check if already exists
    if raw_path.exists():
        logger.info(f"Dataset {name} already exists at {raw_path}")
        return raw_path

    # Download
    logger.info(f"Downloading {name} from {config['url']}")

    # Handle compressed files
    if config.get("compressed", False):
        compressed_path = raw_path.with_suffix(raw_path.suffix + ".gz")
        success = download_file(config["url"], compressed_path, desc=name)

        if success:
            # Decompress
            logger.info(f"Decompressing {compressed_path}")
            with gzip.open(compressed_path, 'rb') as f_in:
                with open(raw_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            compressed_path.unlink()  # Remove compressed file
    else:
        success = download_file(config["url"], raw_path, desc=name)

    if not success:
        logger.error(f"Failed to download {name}")
        return None

    # Post-process if needed
    post_process = config.get("post_process")
    if post_process:
        raw_path = post_process_dataset(name, raw_path, config)

    logger.info(f"Successfully downloaded {name} to {raw_path}")
    return raw_path


def post_process_dataset(name: str, raw_path: Path, config: dict) -> Path:
    """
    Post-process downloaded dataset to standardize format.

    Args:
        name: Dataset name
        raw_path: Path to raw file
        config: Dataset config

    Returns:
        Path to processed file
    """
    post_process_type = config.get("post_process")

    if post_process_type in ["herg", "cyp"]:
        # TDC datasets need column renaming
        try:
            df = pd.read_csv(raw_path)

            # Rename columns to standard format
            rename_map = {
                config["smiles_col"]: "smiles",
                config["target_col"]: "label",
            }
            df = df.rename(columns=rename_map)

            # Save back
            df.to_csv(raw_path, index=False)
            logger.info(f"Post-processed {name}: renamed columns")

            # Update config for downstream use
            config["smiles_col"] = "smiles"
            config["target_col"] = "label"

        except Exception as e:
            logger.warning(f"Post-processing failed for {name}: {e}")

    return raw_path


def load_dataset_csv(
    name: str,
    data_dir: Path,
    subset_size: Optional[int] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Load dataset CSV and return dataframe with column info.

    Args:
        name: Dataset name
        data_dir: Data directory
        subset_size: Optional size limit for large datasets
        seed: Random seed for subsetting

    Returns:
        Tuple of (dataframe, smiles_column, target_column)
    """
    config = DATASET_CONFIGS.get(name, {})
    raw_path = data_dir / "raw" / config.get("filename", f"{name}.csv")

    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset not found: {raw_path}")

    df = pd.read_csv(raw_path)

    # Get column names
    smiles_col = config.get("smiles_col", "smiles")
    target_col = config.get("target_col", "label")

    # Handle column name variations
    if smiles_col not in df.columns:
        for alt in ["smiles", "SMILES", "mol", "Smiles", "canonical_smiles"]:
            if alt in df.columns:
                smiles_col = alt
                break

    if target_col not in df.columns:
        # Try to find a target column
        exclude = {smiles_col, "mol_id", "name", "Name", "ID"}
        candidates = [c for c in df.columns if c not in exclude]
        if candidates:
            target_col = candidates[0]

    # Subset if needed
    subset_size = subset_size or config.get("subset_size")
    if subset_size and len(df) > subset_size:
        logger.info(f"Subsetting {name} from {len(df)} to {subset_size} molecules")
        df = df.sample(n=subset_size, random_state=seed).reset_index(drop=True)

    # Remove rows with missing SMILES
    df = df.dropna(subset=[smiles_col])

    logger.info(f"Loaded {name}: {len(df)} molecules, target={target_col}")

    return df, smiles_col, target_col


def generate_conformers_for_dataset(
    name: str,
    data_dir: Path,
    config: Config,
    splits: List[str] = ["train", "val", "test"],
    force: bool = False,
) -> None:
    """
    Generate conformers for a dataset.

    Args:
        name: Dataset name
        data_dir: Data directory
        config: Configuration
        splits: Splits to process
        force: Force regeneration even if files exist
    """
    # Load dataset
    try:
        df, smiles_col, target_col = load_dataset_csv(name, data_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    smiles_list = df[smiles_col].tolist()
    labels = df[target_col].values if target_col in df.columns else None

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
    if not conf_config:
        conf_config = {
            "max_conformers": 50,
            "energy_window": 15.0,
            "rmsd_threshold": 0.5,
        }

    conformer_generator = ConformerGenerator(**conf_config)
    feature_extractor = FeatureExtractor()

    # Process each split
    conformers_dir = data_dir / "conformers" / name
    conformers_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        if split not in split_indices:
            continue

        split_path = conformers_dir / f"{split}.pkl"

        if split_path.exists() and not force:
            logger.info(f"Conformers for {name}/{split} already exist (use --force to regenerate)")
            continue

        indices = split_indices[split]
        split_smiles = [smiles_list[i] for i in indices]
        split_labels = [labels[i] for i in indices] if labels is not None else None

        logger.info(f"Generating conformers for {name}/{split} ({len(split_smiles)} molecules)")

        all_features = []
        all_energies = []
        all_weights = []
        valid_smiles = []
        valid_labels = []
        failed = 0

        for i, smiles in enumerate(tqdm(split_smiles, desc=f"{name}/{split}")):
            try:
                ensemble = conformer_generator.generate_from_smiles(smiles)

                if ensemble.n_conformers > 0:
                    # Extract features for each conformer
                    mol_features = []
                    for conf_id in ensemble.conformer_ids:
                        features = feature_extractor.extract(ensemble.mol, conf_id)
                        mol_features.append(features)

                    all_features.append(mol_features)
                    all_energies.append(ensemble.energies)
                    all_weights.append(ensemble.boltzmann_weights)
                    valid_smiles.append(smiles)
                    if split_labels is not None:
                        valid_labels.append(split_labels[i])
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                continue

        # Save
        data = {
            "features": all_features,
            "energies": all_energies,
            "boltzmann_weights": all_weights,
            "smiles": valid_smiles,
            "labels": valid_labels if valid_labels else None,
            "indices": indices,
            "dataset_config": DATASET_CONFIGS.get(name, {}),
        }

        with open(split_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(
            f"Saved {name}/{split}: {len(valid_smiles)} molecules, "
            f"{failed} failed ({failed/(len(split_smiles))*100:.1f}%)"
        )


def download_all_datasets(data_dir: Path, datasets: Optional[List[str]] = None) -> dict:
    """
    Download all specified datasets.

    Args:
        data_dir: Data directory
        datasets: List of datasets to download (None for all)

    Returns:
        Dict mapping dataset name to success status
    """
    if datasets is None:
        datasets = list(DATASET_CONFIGS.keys())

    # Remove duplicates (e.g., qm9_homo, qm9_lumo share same file)
    unique_downloads = {}
    for name in datasets:
        config = DATASET_CONFIGS.get(name, {})
        filename = config.get("filename", f"{name}.csv")
        if filename not in unique_downloads:
            unique_downloads[filename] = name

    results = {}
    for filename, name in unique_downloads.items():
        path = download_dataset(name, data_dir)
        results[name] = path is not None

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare datasets for DKO experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and prepare all datasets
  python prepare_datasets.py --all

  # Download only (skip conformer generation)
  python prepare_datasets.py --all --download-only

  # Prepare specific datasets
  python prepare_datasets.py --datasets esol bace freesolv

  # Force regeneration of conformers
  python prepare_datasets.py --datasets esol --force

Available datasets:
  MoleculeNet: esol, freesolv, lipophilicity, bace, bbbp, tox21, hiv, sider, clintox
  QM9 (10k subsets): qm9_homo, qm9_lumo, qm9_gap, qm9_polar
  ADMET: herg, cyp3a4
        """,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to prepare",
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
        help="Data directory (default: data)",
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
        "--force",
        action="store_true",
        help="Force regeneration of conformers",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process (default: train val test)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit",
    )

    args = parser.parse_args()

    # List datasets
    if args.list:
        print("\nAvailable datasets:\n")
        print("MoleculeNet:")
        for name in ["esol", "freesolv", "lipophilicity", "bace", "bbbp", "tox21", "hiv", "sider", "clintox"]:
            config = DATASET_CONFIGS.get(name, {})
            print(f"  {name:15} - {config.get('task', 'unknown')} task")
        print("\nQM9 (10k subsets):")
        for name in ["qm9_homo", "qm9_lumo", "qm9_gap", "qm9_polar"]:
            print(f"  {name:15} - regression task")
        print("\nADMET:")
        for name in ["herg", "cyp3a4"]:
            print(f"  {name:15} - classification task")
        return

    # Setup logging
    data_dir = Path(args.data_dir)
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, name="prepare_datasets")

    # Load config
    if args.config:
        config = Config(base_config_path=args.config)
    else:
        config = Config()

    # Determine datasets to process
    if args.all:
        # Core benchmark datasets
        datasets = [
            "esol", "freesolv", "lipophilicity",
            "bace", "bbbp",
            "qm9_homo", "qm9_lumo", "qm9_gap",
        ]
    elif args.datasets:
        datasets = args.datasets
    else:
        parser.error("Please specify --datasets or --all (use --list to see options)")

    logger.info(f"Preparing datasets: {datasets}")
    logger.info(f"Data directory: {data_dir}")

    # Download datasets
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Downloading datasets")
    logger.info("=" * 60)

    download_results = download_all_datasets(data_dir, datasets)

    successful = [name for name, success in download_results.items() if success]
    failed = [name for name, success in download_results.items() if not success]

    if failed:
        logger.warning(f"Failed to download: {failed}")

    # Generate conformers
    if not args.download_only:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Generating conformers")
        logger.info("=" * 60)

        for name in datasets:
            if name in failed:
                logger.warning(f"Skipping {name} (download failed)")
                continue

            logger.info(f"\n=== Processing {name} ===")
            generate_conformers_for_dataset(
                name,
                data_dir,
                config,
                args.splits,
                force=args.force,
            )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Downloaded: {len(successful)}/{len(datasets)} datasets")
    if not args.download_only:
        logger.info(f"Conformers generated in: {data_dir / 'conformers'}")
    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main()
