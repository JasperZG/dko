# DKO Cluster Deployment Guide

Complete reference for deploying and running DKO experiments on HPC clusters.

## Quick Start

### 1. Initial Setup

```bash
# Clone repository
git clone https://github.com/JasperZG/dko.git
cd dko

# Run setup (creates venv, installs deps, validates)
bash scripts/setup_cluster.sh

# Validate deployment
python scripts/validate_cluster_ready.py
```

### 2. Validate Before Running

```bash
# Quick validation
python scripts/validate_hpc_setup.py

# Comprehensive 20-test validation
python scripts/validate_cluster_ready.py
```

---

## Submitting Jobs

### Single Experiment

```bash
# Basic submission
sbatch scripts/submit_hpc.sh configs/experiments/dko_esol.yaml

# With experiment name
sbatch scripts/submit_hpc.sh configs/experiments/dko_esol.yaml my_experiment
```

### Batch Experiments

```bash
# Submit all default experiments (dry run first)
python scripts/submit_all_experiments.py --dry-run

# Actually submit
python scripts/submit_all_experiments.py --submit

# Custom datasets and models
python scripts/submit_all_experiments.py \
    --datasets esol freesolv lipophilicity \
    --models dko_first_order dko_second_order \
    --seeds 42 123 456 \
    --submit
```

### Array Jobs (Efficient Batch)

```bash
# Submit as SLURM array job (more efficient)
python scripts/submit_all_experiments.py --array --max-concurrent 5 --submit

# Or directly with sbatch
sbatch --array=0-9%3 scripts/slurm_submit_batch.sh configs/experiments/
```

### Hyperparameter Optimization

```bash
# Submit hyperopt job
sbatch scripts/slurm_hyperopt.sh esol 50 tpe

# Or run directly
python scripts/run_hyperopt.py \
    --dataset esol \
    --n-trials 100 \
    --sampler tpe \
    --pruner hyperband
```

---

## Monitoring Jobs

### Real-time Dashboard

```bash
# Monitor all your jobs
python scripts/monitor_jobs.py --watch

# Faster refresh
python scripts/monitor_jobs.py --watch --interval 30

# Check status once
python scripts/monitor_jobs.py --status
```

### SLURM Commands

```bash
# Your running jobs
squeue -u $USER

# All jobs with details
squeue -u $USER -l

# Specific job info
scontrol show job <job_id>

# Job history
sacct -u $USER --format=JobID,JobName,State,Elapsed,MaxRSS
```

---

## Handling Failures

### Check for Failed Jobs

```bash
# Scan for failed experiments
python scripts/recover_failed_jobs.py --scan

# View details
python scripts/recover_failed_jobs.py --check-failed
```

### Recover Failed Jobs

```bash
# Preview recovery actions
python scripts/recover_failed_jobs.py --recover --dry-run

# Actually recover
python scripts/recover_failed_jobs.py --recover

# Generate recovery script for manual review
python scripts/recover_failed_jobs.py --generate-script
```

### Clean Up Failed Jobs

```bash
# Clean logs but keep checkpoints
python scripts/recover_failed_jobs.py --cleanup --dry-run
python scripts/recover_failed_jobs.py --cleanup

# Remove everything (careful!)
python scripts/recover_failed_jobs.py --cleanup --remove-all
```

---

## Results & Analysis

### Aggregate Results

```bash
# Basic aggregation
python scripts/aggregate_results.py

# Filter by dataset
python scripts/aggregate_results.py --filter-dataset esol

# Output formats
python scripts/aggregate_results.py --output results/summary.csv
python scripts/aggregate_results.py --format latex --output results/table.tex
python scripts/aggregate_results.py --format markdown --output results/table.md
```

### View Results

```bash
# Quick summary
python scripts/aggregate_results.py --verbose

# Check specific experiment
cat experiments/<exp_name>/results.json
```

---

## Directory Structure

```
dko-project/
├── configs/
│   ├── base_config.yaml        # Default settings
│   └── experiments/            # Experiment-specific configs
├── data/                       # Downloaded datasets
├── experiments/                # Experiment outputs
│   └── <exp_name>/
│       ├── config.json         # Experiment config
│       ├── results.json        # Final results
│       ├── training.log        # Training log
│       ├── checkpoints/        # Model checkpoints
│       └── metrics/            # Training metrics
├── logs/                       # SLURM logs
├── scripts/                    # All scripts
└── analysis/                   # Aggregated results
```

---

## Common Issues

### Out of Memory (OOM)

```bash
# Solutions:
# 1. Reduce batch size in config
# 2. Enable mixed precision (add to config)
#    training:
#      use_amp: true
# 3. Request more GPU memory
#    #SBATCH --gres=gpu:a100:1
```

### Job Timeout

```bash
# Solutions:
# 1. Increase time limit
#    #SBATCH --time=48:00:00
# 2. Enable checkpointing (automatic resume)
# 3. Use early stopping
```

### Job Won't Start

```bash
# Check queue
squeue -u $USER

# Check partition availability
sinfo

# Check your limits
sacctmgr show user $USER

# Try different partition
sbatch --partition=gpu-low scripts/submit_hpc.sh configs/experiments/dko_esol.yaml
```

### CUDA Not Found

```bash
# Load CUDA module
module load cuda/11.8

# Verify
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Useful SLURM Commands

```bash
# Cancel a job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Cancel jobs by name
scancel --name=dko_train

# Job efficiency report
seff <job_id>

# Detailed job info
sstat -j <job_id>

# Node information
scontrol show node <node_name>

# Partition info
sinfo -p gpu

# Fair share priority
sprio -u $USER
```

---

## Configuration Reference

### Base Config Structure

```yaml
dataset:
  name: esol                    # Dataset name
  n_conformers: 20              # Conformers per molecule
  split_seed: 42                # Data split seed

model:
  type: dko_first_order         # Model type
  kernel_hidden_dims: [128, 64] # Kernel network
  kernel_output_dim: 32         # Kernel output
  dropout: 0.1                  # Dropout rate

training:
  max_epochs: 200               # Training epochs
  batch_size: 32                # Batch size
  learning_rate: 0.001          # Learning rate
  weight_decay: 0.0001          # L2 regularization
  patience: 20                  # Early stopping
  use_amp: false                # Mixed precision
  seed: 42                      # Random seed
```

### Environment Variables

```bash
# Set in SLURM script or bashrc
export CUDA_VISIBLE_DEVICES=0       # GPU to use
export OMP_NUM_THREADS=8            # CPU threads
export PYTHONUNBUFFERED=1           # Real-time logging
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory
```

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `setup_cluster.sh` | Initial cluster setup |
| `validate_hpc_setup.py` | Quick HPC validation |
| `validate_cluster_ready.py` | Comprehensive 20-test validation |
| `submit_hpc.sh` | Submit single experiment |
| `slurm_submit_batch.sh` | Submit array jobs |
| `slurm_hyperopt.sh` | Submit hyperopt job |
| `submit_all_experiments.py` | Bulk experiment submission |
| `train_single_experiment.py` | Training wrapper |
| `run_hyperopt.py` | Hyperparameter optimization |
| `monitor_jobs.py` | Job monitoring |
| `recover_failed_jobs.py` | Failure recovery |
| `aggregate_results.py` | Results aggregation |

---

## Support

For issues:
1. Check logs: `cat logs/slurm_<job_id>.err`
2. Run validation: `python scripts/validate_cluster_ready.py`
3. Check common issues above
4. Contact cluster support for resource issues
