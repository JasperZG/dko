# Harsh Critique: What's Missing for Nature

**Status:** Working through critiques systematically

---

## Summary Table

| # | Critique | Current Status | Priority | Gap Level |
|---|----------|----------------|----------|-----------|
| 1 | No Mechanistic Insight | Not addressed | HIGH | Critical |
| 2 | 3D Features Never Fixed | Not addressed | HIGH | Critical |
| 3 | No SOTA 3D GNN Baselines | Not addressed | HIGH | Critical |
| 4 | Hybrid Result Undersold | Partial (3 seeds) | MEDIUM | Medium |
| 5 | BDE Failure Unexplained | Not addressed | MEDIUM | Medium |
| 6 | Kraken Contradicts Narrative | Partial | MEDIUM | Medium |
| 7 | No Prospective Validation | Not addressed | HIGH | High |
| 8 | Scale Too Small | Not addressed | MEDIUM | Medium |
| 9 | No Theoretical Contribution | Not addressed | HIGH | High |
| 10 | Reproducibility Concerns | Partial | LOW | Low |
| 11 | "FP Wins" Not Novel | Not addressed | MEDIUM | Medium |
| 12 | Statistical Rigor Incomplete | Partial | MEDIUM | Medium |
| 13 | Missing Ablations | Not addressed | MEDIUM | Medium |
| 14 | No Drug Discovery Connection | Not addressed | HIGH | High |

---

## Critique 1: No Mechanistic Insight

**Problem:** Nature publishes "why," not "what." Paper documents that fingerprints win but never explains why.

**Questions to answer:**
- What information do Morgan fingerprints encode that conformer statistics don't?
- Is it substructure patterns? Topological motifs? Ring systems?

**Required work:**
- [ ] Information-theoretic analysis: mutual information between features and targets
- [ ] Feature attribution: which fingerprint bits matter? Which conformer features?
- [ ] Controlled experiments isolating the information gap
- [ ] SHAP/integrated gradients analysis
- [ ] Synthetic molecules where ground truth is known

**Status:** NOT STARTED

---

## Critique 2: 3D Features Need Enhancement

**INVESTIGATION COMPLETE (2026-02-21)**

**Current State (Better Than Expected):**
The codebase already uses **coordinate-based geometric features**:
- Pairwise distances (within 4.0Å cutoff) - **CONFORMER-VARYING**
- Bond angles - **CONFORMER-VARYING**
- Torsion angles (cos/sin encoded) - **CONFORMER-VARYING**
- Atom features (19-dim per atom) - **CONFORMER-INVARIANT**

**Actual Problem:**
While geometric features DO vary, they may not capture property-relevant 3D information. Missing descriptors that capture molecular shape/surface:

**Required work:**
- [ ] Implement additional 3D conformer features:
  - PMI ratios (principal moments of inertia) - shape descriptors
  - SASA (solvent accessible surface area) - critical for solvation
  - 3D pharmacophores - interaction patterns
  - USR (ultrafast shape recognition) - fast shape fingerprints
  - Molecular volume / surface area
  - Radius of gyration
  - Asphericity, eccentricity
- [ ] Implement learned 3D representations:
  - SchNet embeddings
  - PaiNN embeddings
- [ ] Validate new features variance spectrum
- [ ] Compare feature-target correlations before/after

**Status:** INVESTIGATION COMPLETE, IMPLEMENTATION NEEDED

---

## Critique 3: No SOTA 3D GNN Baselines

**Problem:** Current baselines (attention, DeepSets, mean ensemble) are weak. Field has much stronger methods.

**Required baselines:**
- [ ] SchNet (continuous-filter convolutions)
- [ ] DimeNet++ (directional message passing)
- [ ] PaiNN (equivariant message passing)
- [ ] GemNet (geometric message passing)
- [ ] Uni-Mol (3D pre-trained)
- [ ] 3D-Infomax (contrastive 3D)
- [ ] GeomGCL (geometric contrastive learning)
- [ ] MARCEL paper baselines

**Additional requirements:**
- [ ] Computational cost comparison (FLOPs, training time)
- [ ] Memory usage comparison
- [ ] Inference time comparison

**Status:** NOT STARTED

---

## Critique 4: Hybrid Result Undersold and Undervalidated

**Problem:** Strongest positive finding (FP + conformer hybrid improves 10% on ESOL) is buried and only validated on 3 seeds.

**Required work:**
- [ ] 10+ seeds on ALL datasets for hybrid experiments
- [ ] Detailed ablation: which conformer features contribute?
- [ ] External validation on held-out dataset
- [ ] Prospective validation if possible
- [ ] Understanding of WHY hybrid works
- [ ] Demonstration on novel task where this matters
- [ ] Make hybrid features the centerpiece of the paper

**Status:** PARTIAL (3 seeds, basic ablation done)

---

## Critique 5: BDE Failure is a Red Flag

**INVESTIGATION COMPLETE (2026-02-21)**

**Root Causes Identified:**

1. **Feature-Target Mismatch (PRIMARY):**
   - Mean |correlation| between features and BDE: **0.0436** (nearly random!)
   - Only 5% of features have |r| > 0.1
   - BDE is a quantum/electronic property; geometric features don't capture it

2. **Hardcoded Dummy Energies (BUG):**
   - `prepare_bde.py` line 161: `energies = np.ones(...)`
   - All conformer energies set to 1.0 instead of actual MMFF94 values
   - Prevents proper Boltzmann weighting

3. **Variable Feature Dimensions:**
   - Features range 187-1854 dimensions across molecules
   - 60% need padding, 38% are truncated
   - Inconsistent feature semantics

4. **Task-Model Mismatch:**
   - BDE requires understanding electronic structure
   - Would need quantum descriptors or learned representations

**Fix Plan:**
- [x] Diagnose WHY neural models fail - DONE
- [ ] Fix conformer energy extraction in prepare_bde.py
- [ ] Add electronic/quantum descriptors for BDE
- [ ] Or: acknowledge limitation and remove BDE from neural comparison
- [ ] Keep FP+XGBoost result (works well: R²=0.958)

**Recommendation:** Document BDE as a **negative control** showing that geometric conformer features are insufficient for electronic properties. FP+XGBoost success shows 2D structure encodes bond information better.

**Status:** DIAGNOSIS COMPLETE, DECISION NEEDED

---

## Critique 6: Kraken Contradicts Narrative

**Problem:** On Kraken, attention beats mean by 40%+, proving conformer modeling DOES help when targets depend on 3D ensembles.

**Required work:**
- [ ] Unified theoretical framework for when conformers help
- [ ] Clear taxonomy of property types (electronic vs steric vs solvation)
- [ ] Predictive model: given a new dataset, will conformers help?
- [ ] Reconcile Kraken results with MoleculeNet results
- [ ] Develop decision framework with theoretical grounding

**Status:** PARTIAL (results documented, no framework)

---

## Critique 7: No Prospective/Experimental Validation

**Problem:** Nature strongly prefers papers with wet-lab validation or real-world impact.

**Required work:**
- [ ] Collaboration with experimentalists
- [ ] Predict solubility of novel compounds
- [ ] Validate predictions experimentally (10-20 molecules minimum)
- [ ] Application to real drug discovery campaign
- [ ] Retrospective analysis on known drug failures/successes

**Status:** NOT STARTED

---

## Critique 8: Scale Too Small

**Problem:** Largest dataset is Drugs-75K. MoleculeNet datasets are tiny (ESOL: 1,128 molecules).

**Required work:**
- [ ] Large-scale validation:
  - Full QM9 (134K molecules)
  - ChEMBL solubility/lipophilicity (100K+)
  - ZINC subsets
  - PubChem data
- [ ] Pre-training experiments
- [ ] Scaling analysis (how does gap change with dataset size?)
- [ ] Learning curves across scales

**Status:** NOT STARTED

---

## Critique 9: No Theoretical Contribution

**Problem:** Paper is purely empirical. Nature values theoretical grounding.

**Required work:**
- [ ] Information-theoretic analysis of conformer vs fingerprint representations
- [ ] Formal characterization of when Σ adds information beyond μ
- [ ] Connection to statistical physics:
  - Boltzmann averaging
  - Free energy perturbation
- [ ] Generalization bounds or sample complexity analysis
- [ ] Prove/conjecture under what conditions Σ contributes
- [ ] Connect to ensemble thermodynamics

**Status:** NOT STARTED

---

## Critique 10: Reproducibility Concerns

**Problem:** Original implementation had 5 critical bugs causing 10x performance degradation.

**Required work:**
- [ ] Independent reproduction by second lab
- [ ] Comprehensive test suite with synthetic validation
- [ ] Public leaderboard for submissions
- [ ] Code review by external party
- [ ] Extensive documentation

**Status:** PARTIAL (bugs fixed, tests exist, code public)

---

## Critique 11: "Fingerprints Win" Not Novel

**Problem:** Everyone knows fingerprints are strong. GNNs struggling on small datasets isn't Nature-level insight.

**What would be novel:**
- [ ] Prove fingerprints theoretically optimal for certain tasks
- [ ] Show HOW to combine fingerprints with 3D effectively
- [ ] Demonstrate task where conformers necessary and FP fail
- [ ] Reframe around positive contribution (hybrid) not negative (neural loses)

**Status:** NOT STARTED

---

## Critique 12: Statistical Rigor Incomplete

| Gap | Current | Nature Standard |
|-----|---------|-----------------|
| Seeds | 3 (most), 10 (ESOL/Lipo) | 10+ everywhere |
| Effect sizes | Not reported | Cohen's d, CIs |
| Multiple testing | Not corrected | Bonferroni/FDR |
| Power analysis | None | Required for negative results |

**Required work:**
- [ ] Run 10 seeds everywhere
- [ ] Report effect sizes (Cohen's d)
- [ ] Report confidence intervals
- [ ] Apply multiple testing correction
- [ ] Power analysis for negative results

**Status:** PARTIAL (10 seeds on 2 datasets)

---

## Critique 13: Missing Ablations

**Required ablations:**
- [ ] Number of conformers (K=1, 5, 10, 20, 50)
- [ ] Conformer generation method (MMFF vs DFT vs ML)
- [ ] Feature dimensionality (PCA compression levels)
- [ ] Training data size (learning curves)
- [ ] Model architecture variations
- [ ] Regularization strategies

**Status:** NOT STARTED

---

## Critique 14: No Drug Discovery Connection

**Problem:** Nature cares about impact. Need connection to real outcomes.

**Required work:**
- [ ] Connection to ADMET prediction pipeline
- [ ] Analysis of drugs that failed due to solubility
- [ ] Cost-benefit analysis (compute cost vs prediction improvement)
- [ ] Calculate compounds correctly triaged using hybrid vs FP-only
- [ ] Frame in terms of drug discovery value

**Status:** NOT STARTED

---

## Action Plan

### Phase 1: Critical Fixes (Must Have)
1. **Implement proper 3D features** (Critique 2)
2. **Add SOTA baselines** (Critique 3)
3. **Diagnose BDE failure** (Critique 5)

### Phase 2: Validation (Must Have)
4. **10-seed validation everywhere** (Critique 12)
5. **Scale up to larger datasets** (Critique 8)
6. **Comprehensive ablations** (Critique 13)

### Phase 3: Insight Development (Must Have)
7. **Mechanistic analysis** (Critique 1)
8. **Theoretical framework** (Critique 9)
9. **When-conformers-help framework** (Critique 6)

### Phase 4: Impact (Should Have)
10. **Hybrid as centerpiece** (Critique 4)
11. **Drug discovery connection** (Critique 14)
12. **Reframe narrative** (Critique 11)

### Phase 5: Stretch Goals (Nice to Have)
13. **Prospective validation** (Critique 7)
14. **External reproduction** (Critique 10)

---

## Progress Log

| Date | Critique | Work Done |
|------|----------|-----------|
| 2026-02-21 | #2, #5 | Investigated current feature setup - features ARE 3D-varying (distances, angles, torsions). Created enhanced 3D features module (dko/data/features_3d.py) with PMI, SASA, USR, etc. |
| 2026-02-21 | #5 | Diagnosed BDE failure: features have ~0 correlation with targets, dummy energies bug found, task requires quantum descriptors |
| 2026-02-21 | #1 | Created mechanistic analysis script (scripts/mechanistic_analysis.py) for feature-target correlations, MI, XGBoost importance |
