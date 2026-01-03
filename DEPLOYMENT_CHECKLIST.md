# DKO Cluster Deployment - Final Checklist

**Date:** ___________
**Cluster:** ___________
**User:** ___________

---

## Pre-Deployment Checklist

### [ ] 1. Repository Setup

- [ ] Code is committed to git
- [ ] No uncommitted changes (or documented why)
- [ ] Git hash recorded: `________________`
- [ ] All required files present:
  - [ ] `dko/` directory with all modules
  - [ ] `configs/` directory with configs
  - [ ] `scripts/` directory with all scripts
  - [ ] `requirements.txt` or `environment.yml`

### [ ] 2. Local Validation

Run on your local machine/login node:

```bash
python scripts/validate_cluster_ready.py
```

**Result:** `[ ] PASS` / `[ ] FAIL`

If FAIL, do not proceed. Fix issues first.

### [ ] 3. Cluster Access

- [ ] Can SSH to cluster: `ssh <cluster>`
- [ ] Can access compute nodes
- [ ] SLURM commands work:
  - [ ] `squeue` works
  - [ ] `sbatch --version` works
  - [ ] `sinfo` shows GPU partition
- [ ] Your account has GPU allocation
- [ ] Check allocation balance: `____________ hours remaining`

### [ ] 4. File Transfer

Transfer code to cluster:

```bash
# From local machine
rsync -avz --exclude='data/' --exclude='*.pyc' \
    dko-project/ <user>@<cluster>:~/dko-project/
```

- [ ] Files transferred successfully
- [ ] Code directory on cluster: `~/dko-project/`

### [ ] 5. Environment Setup

On cluster:

```bash
cd ~/dko-project
bash scripts/setup_cluster.sh
```

- [ ] Virtual environment created at `~/venv/dko/`
- [ ] All packages installed
- [ ] No installation errors

### [ ] 6. Module Loading

Test module loading:

```bash
module purge
module load python/3.9
module load cuda/11.8
module load cudnn/8.6
```

- [ ] Python version: `___________`
- [ ] CUDA version: `___________`
- [ ] cuDNN version: `___________`

**Document exact module names for your cluster:**
- Python: `module load ___________`
- CUDA: `module load ___________`
- cuDNN: `module load ___________`

### [ ] 7. Validation on Cluster

```bash
python scripts/validate_cluster_ready.py
```

**Results:**
- [ ] System info: PASS
- [ ] PyTorch/CUDA: PASS
- [ ] Required packages: PASS
- [ ] Conformer generation: PASS
- [ ] Feature extraction: PASS
- [ ] Dataset loading: PASS (or documented if data not downloaded)
- [ ] DKO model: PASS
- [ ] Attention model: PASS
- [ ] DeepSets model: PASS

**Critical failures:** (list any)
```
___________________________________________
___________________________________________
```

### [ ] 8. Cluster-Specific Testing

```bash
python scripts/test_cluster_deployment.py --quick
```

**Results:**
- [ ] SLURM availability: PASS
- [ ] GPU nodes: PASS
- [ ] Python environment: PASS
- [ ] Data loading: PASS
- [ ] Checkpoint operations: PASS
- [ ] Monitoring scripts: PASS

**Total:** `___ / ___ tests passed`

If < 90% pass rate, investigate failures before proceeding.

---

## Test Job Submission

### [ ] 9. Dry Run Test

Submit a minimal test job:

```bash
sbatch scripts/slurm_submit_single.sh DKO esol configs/base_config.yaml test_output
```

- [ ] Job submitted successfully
- [ ] Job ID: `___________`

Monitor the job:

```bash
# Check status
squeue -j <job_id>

# Check output (wait for completion)
tail -f test_output/training.log
```

**Job completed:** `[ ] YES` / `[ ] NO`

**If YES:**
- [ ] `test_output/results.json` exists
- [ ] No errors in `logs/slurm_*.err`
- [ ] Training completed successfully
- [ ] GPU was used (check logs)

**Test metrics:**
- RMSE: `___________`
- Training time: `___________`

**If NO, failure reason:**
```
___________________________________________
```

### [ ] 10. Monitor Test

Start monitoring dashboard:

```bash
python scripts/monitor_jobs.py --status
```

- [ ] Dashboard shows test job
- [ ] Progress updates correctly
- [ ] No errors in dashboard

---

## Data Preparation

### [ ] 11. Dataset Download

Download datasets:

```bash
# Test with one dataset first
python -c "
from dko.data.datasets import get_dataset
from pathlib import Path

datasets = ['esol']  # Start with one
for ds in datasets:
    print(f'Downloading {ds}...')
    get_dataset(ds, root=Path('./data'), split='train')
    print(f'Done: {ds}')
"
```

- [ ] ESOL downloaded
- [ ] Data directory size: `___________ GB`

**For full experiments, download all datasets as needed.**

- [ ] All required datasets downloaded
- [ ] Total data size: `___________ GB`
- [ ] Disk quota check: `___________ GB available`

---

## Configuration Review

### [ ] 12. Experiment Configurations

Review `configs/base_config.yaml`:

**Training settings:**
- Max epochs: `___________` (default: 200)
- Batch size: `___________` (default: 32)
- Learning rate: `___________` (default: 1e-3)
- Early stopping patience: `___________` (default: 20)

**Data settings:**
- Max conformers: `___________` (default: 20)

**Computational:**
- Use mixed precision: `___________` (default: False)
- Gradient clip: `___________` (default: 1.0)

- [ ] Settings reviewed and acceptable
- [ ] Modified settings documented

### [ ] 13. SLURM Configuration

Review `scripts/slurm_submit_single.sh`:

**Resource requests:**
- Time limit: `___________` (default: 24:00:00)
- Nodes: `___________` (default: 1)
- CPUs per task: `___________` (default: 8)
- GPUs: `___________` (default: 1)
- Memory: `___________` (default: 64G)
- Partition: `___________` (default: gpu)

**Are these appropriate for your cluster?**
- [ ] YES - Using defaults
- [ ] MODIFIED - Changes documented below

**Modifications:**
```
___________________________________________
___________________________________________
```

---

## Batch Submission Plan

### [ ] 14. Experiment Scope

**Models to test:** (check all that apply)
- [ ] DKO (full second-order)
- [ ] DKO_FirstOrder (ablation)
- [ ] Attention (baseline)
- [ ] DeepSets (baseline)

**Datasets to use:** (check all that apply)
- [ ] esol
- [ ] freesolv
- [ ] lipophilicity
- [ ] (add others as needed)

**Total experiments:** `___ models x ___ datasets x ___ seeds = ___ jobs`

**Estimated resources:**
- Jobs: `___________`
- GPU hours per job (estimate): `___________`
- Total GPU hours: `___________`
- Your allocation: `___________`

**Sufficient allocation?** `[ ] YES` / `[ ] NO`

If NO, reduce scope or request more hours.

### [ ] 15. Submission Strategy

**Choose one:**

**Option A: Submit all at once**
```bash
python scripts/submit_all_experiments.py \
    --models dko_first_order dko_second_order \
    --datasets esol freesolv \
    --submit
```

**Option B: Submit in batches**
```bash
# Batch 1: Small datasets
python scripts/submit_all_experiments.py \
    --models dko_first_order \
    --datasets esol freesolv \
    --submit

# Wait for completion, then submit Batch 2
```

**Option C: Array job**
```bash
sbatch --array=0-9%3 scripts/slurm_submit_batch.sh configs/experiments/
```

**Selected strategy:** `___________`

**Reason:**
```
___________________________________________
```

---

## Monitoring Plan

### [ ] 16. Monitoring Setup

**Dashboard monitoring:**
```bash
# In a tmux/screen session
tmux new -s dko_monitor
python scripts/monitor_jobs.py --watch
# Detach: Ctrl+B, D
```

- [ ] Monitoring session created
- [ ] Session name: `___________`

**Log checking plan:**
- [ ] Will check logs daily at: `___________`
- [ ] Error log location: `logs/`
- [ ] Experiment logs: `experiments/*/training.log`

### [ ] 17. Error Recovery Plan

**In case of failures:**

1. **Detect failures:**
```bash
python scripts/recover_failed_jobs.py --scan
```

2. **Review failure report**

3. **Recover if appropriate:**
```bash
python scripts/recover_failed_jobs.py --recover
```

- [ ] Recovery scripts tested
- [ ] Know how to use recovery tools

---

## Pre-Launch Checklist

### [ ] 18. Final Verification

**Before submitting production jobs:**

- [ ] Test job completed successfully
- [ ] All validation tests passed
- [ ] Data downloaded and verified
- [ ] Configurations reviewed
- [ ] Resource allocation sufficient
- [ ] Monitoring plan in place
- [ ] Backup/snapshot of code taken
- [ ] Collaborators notified (if applicable)

**Backup code:**
```bash
# Create snapshot
git tag -a cluster-deploy-$(date +%Y%m%d) -m "Cluster deployment"
git push --tags

# Or create tarball
tar -czf dko-backup-$(date +%Y%m%d).tar.gz dko-project/
```

- [ ] Backup created at: `___________`

### [ ] 19. Go/No-Go Decision

**Review all sections above. Any critical failures?**

- [ ] **GO** - All critical items passed, ready to launch
- [ ] **NO-GO** - Issues remain, need to resolve:

**If NO-GO, issues to resolve:**
```
1. ___________________________________________
2. ___________________________________________
3. ___________________________________________
```

---

## Launch

### [ ] 20. Production Submission

**Time:** `___________`
**Date:** `___________`

```bash
# Submit production jobs
python scripts/submit_all_experiments.py \
    --models dko_first_order dko_second_order \
    --datasets <your dataset list> \
    --submit
```

**Jobs submitted:**
- Job IDs: `___________`
- Total jobs: `___________`

**Monitor:**
```bash
# Attach to monitoring session
tmux attach -t dko_monitor

# Or check queue
squeue -u $USER
```

### [ ] 21. Initial Monitoring (First 2 Hours)

**After 30 minutes:**
- [ ] At least one job started
- [ ] No immediate errors in logs
- [ ] GPU utilization looks normal

**After 1 hour:**
- [ ] Jobs progressing (check epoch numbers)
- [ ] Loss values are reasonable (not NaN)
- [ ] No repeated failures

**After 2 hours:**
- [ ] Multiple jobs running or completed
- [ ] System stable
- [ ] No resource issues

**Issues found:**
```
___________________________________________
___________________________________________
```

---

## Post-Launch

### [ ] 22. Daily Monitoring

**Check daily at:** `___________`

**Daily checklist:**
- [ ] Check job status: `squeue -u $USER`
- [ ] Review error logs: `ls -lrt logs/*.err`
- [ ] Monitor dashboard
- [ ] Check for failed jobs
- [ ] Verify disk space: `df -h`

**Log date and status:**

| Date | Jobs Running | Jobs Completed | Jobs Failed | Notes |
|------|--------------|----------------|-------------|-------|
|      |              |                |             |       |
|      |              |                |             |       |
|      |              |                |             |       |

### [ ] 23. Results Collection

**When jobs complete:**

```bash
# Aggregate results
python scripts/aggregate_results.py

# View summary
python scripts/aggregate_results.py --verbose
```

- [ ] All jobs completed
- [ ] Results aggregated
- [ ] Analysis generated

**Final job count:**
- Completed successfully: `___________`
- Failed: `___________`
- Success rate: `___________`%

---

## Sign-off

**Deployment completed by:** `___________`
**Date:** `___________`
**Total duration:** `___________`
**Total GPU hours used:** `___________`

**Overall success:** `[ ] YES` / `[ ] PARTIAL` / `[ ] NO`

**Key findings:**
```
___________________________________________
___________________________________________
___________________________________________
```

**Next steps:**
```
___________________________________________
___________________________________________
___________________________________________
```

---

## Troubleshooting Reference

### Common Issues

**Jobs not starting:**
```bash
# Check why
squeue -j <job_id> -o "%.18i %.9P %.8T %.10M %.9l %.6D %R"
```

**OOM errors:**
- Reduce batch size in config
- Enable mixed precision
- Use gradient accumulation

**Slow progress:**
- Check GPU utilization: `nvidia-smi`
- Check if data loading is bottleneck
- Verify number of workers

**Data loading errors:**
- Check data directory permissions
- Verify dataset files exist
- Check disk quota

### Emergency Commands

**Cancel all jobs:**
```bash
scancel -u $USER
```

**Cancel specific job:**
```bash
scancel <job_id>
```

**Check job details:**
```bash
scontrol show job <job_id>
```

**Check node status:**
```bash
sinfo -N -l
```

---

**END OF CHECKLIST**

Save this file with timestamps and notes for future reference.
