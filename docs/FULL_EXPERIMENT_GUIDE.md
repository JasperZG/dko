# DKO Full Experiment Guide

Complete guide for running the full DKO benchmark experiments on an HPC cluster.

---

## Overview

The full experiment pipeline consists of:
1. **Setup** - Install dependencies, validate environment
2. **Hyperparameter Optimization** - Find best settings for each model/dataset
3. **Main Experiments** - Run all models on all datasets with multiple seeds
3b. **Analysis Experiments** - Sample efficiency, decomposition, negative controls (optional)
4. **Results Aggregation** - Combine results and generate analysis

**Estimated time:** 2-4 days depending on cluster resources

---

## Available Models

### DKO Variants
- **DKO** - Full Distribution Kernel Operator with first and second-order statistics
- **DKOFirstOrder** - First-order only (mean statistics)
- **DKOKernel** - Kernel-based variant
- **DKONoPSD** - Without positive semi-definite constraint

### Ensemble Baselines
- **MeanFeatureAggregation (MFA)** - Averages features before network
- **MultiInstanceLearning (MIL)** - Instance-level encoding with pooling
- **MeanEnsemble** - Simple prediction averaging
- **BoltzmannEnsemble** - Energy-weighted prediction averaging
- **SingleConformer** - Lowest-energy conformer only

### Aggregation Methods
- **AttentionPooling** - Multi-head attention aggregation
- **DeepSets** - Permutation-invariant set function

### GNN Baselines (with conformer aggregation)
- **SchNet/SchNetPyG** - Continuous-filter convolutional network
- **DimeNet++/DimeNetPPPyG** - Directional message passing
- **SphereNet** - Spherical message passing
- **3D-Infomax** - Contrastive learning for 3D representations
- **GEM** - Geometry-Enhanced Molecular representation

### Augmented Baselines (for Rep vs Arch study)
- **AttentionAugmented** - Attention with explicit second-order features (outer products)
- **DeepSetsAugmented** - DeepSets with explicit second-order features

---

## Step 1: Setup (30 minutes)

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

## Step 2: Hyperparameter Optimization (4-8 hours)

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

## Step 3: Main Experiments (1-3 days)

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
- 10 models (DKO, DKO-FirstOrder, Attention, DeepSets, MFA, MIL, MeanEnsemble, BoltzmannEnsemble, SchNet, DimeNet++)
- 3 random seeds each
- = **360 total experiments**

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

## Step 3b: Analysis Experiments (Optional, 4-8 hours)

After main benchmark, run additional analysis experiments to understand DKO's behavior.

### Sample Efficiency Study

```bash
# Data fraction experiment (how much training data is needed?)
python -c "from dko.experiments import run_sample_efficiency_experiment; run_sample_efficiency_experiment('esol')"

# Conformer count experiment (how many conformers are needed?)
python -c "from dko.experiments import run_conformer_count_experiment; run_conformer_count_experiment('esol')"

# Full sample efficiency study
python -c "from dko.experiments import run_full_sample_efficiency_study; run_full_sample_efficiency_study()"
```

### 80/20 Decomposition Study

Decomposes DKO's advantage into mean estimation vs covariance contributions:

```bash
python -c "from dko.experiments import run_full_decomposition_study; run_full_decomposition_study()"
```

### Representation vs Architecture Study

Tests whether representation (geometric features) or architecture (DKO) matters more:

```bash
python -c "from dko.experiments import run_full_rep_vs_arch_study; run_full_rep_vs_arch_study()"
```

### Negative Control Experiments

Validates that DKO advantage correlates with SCC (conformational complexity):

```bash
# SCC-based negative controls
python -c "from dko.experiments import run_negative_control_experiment; run_negative_control_experiment()"

# SCC-advantage correlation
python -c "from dko.experiments import run_scc_advantage_correlation; run_scc_advantage_correlation()"
```

### Attention Scaling Analysis

Analyzes attention weight entropy and scaling behavior:

```bash
python -c "from dko.experiments import run_attention_scaling_experiment; run_attention_scaling_experiment('esol')"
```

### Decision Rule Calibration

Calibrates SCC threshold for automatic method selection:

```bash
python -c "from dko.experiments import run_decision_rule_experiment; run_decision_rule_experiment()"
```

### Sketching Experiments (for large ensembles)

Tests random sketching for scaling to large conformer ensembles:

```bash
python -c "from dko.experiments import run_sketching_experiment; run_sketching_experiment('esol')"
```

---

## Step 4: Results Aggregation (10 minutes)

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
[ ] Step 1: Setup
    [ ] Code transferred to cluster
    [ ] setup_cluster.sh completed
    [ ] validate_hpc_setup.py passed

[ ] Step 2: Hyperopt
    [ ] Hyperopt jobs submitted
    [ ] Best parameters saved

[ ] Step 3: Main Experiments
    [ ] All experiments submitted (DKO, baselines, GNNs)
    [ ] Jobs completing without errors
    [ ] Failed jobs resubmitted

[ ] Step 3b: Analysis Experiments (Optional)
    [ ] Sample efficiency study
    [ ] 80/20 decomposition study
    [ ] Representation vs architecture study
    [ ] Negative control experiments
    [ ] Attention scaling analysis
    [ ] Decision rule calibration
    [ ] Sketching experiments

[ ] Step 4: Results
    [ ] Results aggregated
    [ ] Summary generated
    [ ] Figures created
```

---

## Expected Results

After completing all experiments, you should see:

### Main Benchmark

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

### Analysis Experiments

| Experiment | Expected Finding |
|------------|------------------|
| Sample Efficiency | DKO advantage grows with more data |
| Conformer Count | Diminishing returns after ~30 conformers |
| 80/20 Decomposition | ~80% from mean, ~20% from covariance |
| Rep vs Arch | Representation matters more than architecture |
| Negative Controls | DKO advantage correlates with SCC |
| Decision Rule | SCC threshold ~0.3-0.5 for method selection |
| Sketching | 50% sketch retains 95%+ performance |
