# DKO Full Experiment Guide

Complete guide for running the full DKO benchmark experiments on an HPC cluster.

---

## Overview

The full experiment pipeline consists of:
1. **Setup** - Install dependencies, validate environment
2. **Hyperparameter Optimization** - Find best settings for each model/dataset
3. **Main Experiments** - Run all models on all datasets with multiple seeds
4. **Results Aggregation** - Combine results and generate analysis

**Estimated time:** 2-4 days depending on cluster resources

---

## Phase 1: Setup (30 minutes)

### 1.1 Transfer Code to Cluster

```bash
# From your local machine
rsync -avz --exclude='data/' --exclude='__pycache__/' --exclude='.git/' \
    dko/ user@cluster:~/dko/
```

### 1.2 SSH to Cluster and Setup

```bash
ssh user@cluster
cd ~/dko

# Run automated setup
bash scripts/setup_cluster.sh
```

This creates a virtual environment and installs all dependencies.

### 1.3 Validate Setup

```bash
# Quick validation (< 1 minute)
python scripts/validate_hpc_setup.py

# If issues, run comprehensive validation
python scripts/validate_cluster_ready.py
```

**Expected output:** All tests should pass. If not, check the error messages.

---

## Phase 2: Hyperparameter Optimization (4-8 hours)

Find optimal hyperparameters for each model before running main experiments.

### 2.1 Run Hyperopt for Each Model

```bash
# DKO model on a representative dataset
sbatch scripts/slurm_hyperopt.sh esol 50 tpe

# Check job status
squeue -u $USER
```

**What this does:**
- Runs 50 trials with different hyperparameter combinations
- Uses TPE (Tree-structured Parzen Estimator) for smart search
- Saves best parameters to `results/hyperopt/`

### 2.2 Hyperopt for All Models (Optional)

```bash
# Run hyperopt for multiple models
for model in dko attention deepsets; do
    sbatch scripts/slurm_hyperopt.sh esol 50 tpe --model $model
done
```

### 2.3 Check Hyperopt Results

```bash
# View best parameters found
cat results/hyperopt/esol_dko_best_params.json
```

---

## Phase 3: Main Experiments (1-3 days)

### 3.1 Experiment Configuration

The experiments are defined in `configs/experiments/`. Each config specifies:
- Dataset (esol, bace, freesolv, etc.)
- Model (dko, attention, deepsets, etc.)
- Training settings

### 3.2 Option A: Run All Experiments (Recommended)

```bash
# Preview what will run (dry run)
python scripts/submit_all_experiments.py --dry-run

# Submit all experiments
python scripts/submit_all_experiments.py --submit
```

This runs:
- 12 datasets
- 5 models (DKO, DKO-FirstOrder, Attention, DeepSets, MeanPooling)
- 3 random seeds each
- = **180 total experiments**

### 3.3 Option B: Run as Array Job (More Efficient)

```bash
# Submit as SLURM array job
python scripts/submit_all_experiments.py --array --max-concurrent 10 --submit
```

Array jobs are more efficient because:
- Single job submission for all experiments
- SLURM manages scheduling automatically
- Easier to monitor and cancel

### 3.4 Option C: Run Specific Experiments

```bash
# Single experiment
sbatch scripts/submit_hpc.sh configs/experiments/dko_esol.yaml

# Specific datasets and models
python scripts/submit_all_experiments.py \
    --datasets esol bace freesolv \
    --models dko attention \
    --seeds 42 123 456 \
    --submit
```

### 3.5 Monitor Progress

```bash
# Check job status
squeue -u $USER

# Detailed monitoring
python scripts/monitor_jobs.py

# View logs for a specific job
cat logs/slurm_<job_id>.out
cat logs/slurm_<job_id>.err
```

### 3.6 Handle Failures

```bash
# Check for failed jobs
python scripts/monitor_jobs.py --failed

# Automatically resubmit failed jobs
python scripts/recover_failed_jobs.py --resubmit
```

---

## Phase 4: Results Aggregation (10 minutes)

After all experiments complete:

### 4.1 Aggregate Results

```bash
python scripts/aggregate_results.py \
    --results-dir results/ \
    --output results/aggregated/
```

### 4.2 View Summary

```bash
# Print summary table
python scripts/aggregate_results.py --summary

# Output:
# Dataset     | DKO    | Attention | DeepSets | Improvement
# ------------|--------|-----------|----------|------------
# ESOL        | 0.523  | 0.567     | 0.589    | +7.8%
# BACE        | 0.842  | 0.801     | 0.789    | +5.1%
# ...
```

### 4.3 Generate Plots

```bash
python scripts/analyze_results.py \
    --results results/aggregated/ \
    --output figures/
```

This generates:
- Performance comparison bar charts
- Learning curves
- Statistical significance tables
- Attention weight visualizations

---

## Quick Reference

### Job Commands

| Command | Purpose |
|---------|---------|
| `squeue -u $USER` | Check your jobs |
| `scancel <job_id>` | Cancel a job |
| `scancel -u $USER` | Cancel all your jobs |
| `sinfo` | Check cluster status |

### Key Files

| File | Purpose |
|------|---------|
| `logs/slurm_*.out` | Job stdout |
| `logs/slurm_*.err` | Job stderr |
| `results/<exp>/metrics.json` | Experiment results |
| `checkpoints/<exp>/best.pt` | Best model weights |

### Troubleshooting

**Job pending too long?**
```bash
squeue -u $USER  # Check reason column
sinfo            # Check if partition is available
```

**Out of memory?**
Edit `scripts/submit_hpc.sh` and increase `--mem`:
```bash
#SBATCH --mem=32G  # Increase from 16G
```

**Job failed?**
```bash
cat logs/slurm_<job_id>.err  # Check error message
```

---

## Experiment Checklist

```
[ ] Phase 1: Setup
    [ ] Code transferred to cluster
    [ ] setup_cluster.sh completed
    [ ] validate_hpc_setup.py passed

[ ] Phase 2: Hyperopt
    [ ] Hyperopt jobs submitted
    [ ] Best parameters saved

[ ] Phase 3: Main Experiments
    [ ] All experiments submitted
    [ ] Jobs completing without errors
    [ ] Failed jobs resubmitted

[ ] Phase 4: Results
    [ ] Results aggregated
    [ ] Summary generated
    [ ] Figures created
```

---

## Expected Results

After completing all experiments, you should see:

| Dataset | DKO Improvement vs Best Baseline |
|---------|----------------------------------|
| BACE | 5-8% |
| PDBBind | 8-12% |
| FreeSolv | 4-7% |
| hERG | 2-4% |
| ESOL | 1-3% |
| Lipophilicity | 1-3% |
| QM9 | 4-7% |

DKO performs best on datasets where conformational flexibility matters (binding affinity, solvation).
