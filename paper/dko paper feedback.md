# DKO Paper Review — Comprehensive Feedback Checklist

**Paper:** "When Does Conformer Geometry Help? Complementarity of 3D Ensemble Statistics and 2D Fingerprints for Molecular Property Prediction"

---

## Part 1: Content & Writing

### Critical Errors

- [ ] **SCC decision rule is contradicted by your own data.** The Discussion claims "datasets with median SCC > 25 are more likely to benefit from conformer features." But Table 7 shows FreeSolv has median SCC = **0.01**, yet Table 3 shows FreeSolv improves by **−3.9%** in the hybrid. This directly contradicts the SCC > 25 rule. The resolution is that FreeSolv's improvement comes from **μ** (first-order), not **Σ** (second-order) — Table 3 shows FP+μ already captures most of the gain (2.939 → 2.831) while FP+Σ actually hurts (2.939 → 3.119). But this critical distinction between "when do *any* conformer features help" vs. "when do *covariance* features help" is never made explicit. The SCC rule as stated is wrong or at best misleading.

- [ ] **BACE appears out of nowhere in Section 4.8.** "effective rank at 90% ranging from 353 (QM9) to 685 (BACE)" — BACE is never introduced in Section 4.1 (Datasets) and is never mentioned elsewhere in the paper. This is either an error (wrong dataset name) or a missing dataset description.

- [ ] **Table 7 contains a non-numeric value.** FreeSolv's "Max" column reads **"low"** instead of a number. This is clearly a placeholder/bug that was never fixed before submission. This alone could get the paper desk-rejected.

- [ ] **Feature encoding is critically underspecified (Section 3.1).** "feature vector x_i ∈ ℝ^D encoding interatomic distances, angles, and torsions (D = 1024 for real datasets)" — *how* are distances, angles, and torsions encoded into a 1024-dimensional vector? Are these all pairwise distances? Selected atom pairs? Binned histograms? How are they ordered for molecules of different sizes? This is a core methodological detail that is completely missing. A reviewer cannot reproduce the method without it.

- [ ] **"dko_separate_nets" is never described.** It appears in Table 1 (ranking last at 13th) and Table 6 but is never defined in Section 3.3 (DKO Variants) or anywhere in the text. Similarly, "single_conformer" and "deepsets" appear in Table 6 without any description.

### Undefined Terms / Acronyms

- [ ] **"n.s."** in Table 4 — presumably "not significant" but never defined. Spell out on first use or add a table footnote.
- [ ] **"−"** in Table 3's Δ row — ambiguous. Does this mean "no improvement," "not computed," or "not applicable"? Must be defined in the caption.
- [ ] **|ρ|** in abstract and Section 4.6 — is this Spearman or Pearson? The abstract says "feature–target |ρ| = 0.044" but uses the same symbol ρ that typically denotes Spearman. Section 4.7 later uses "|r|" for Pearson. Be consistent and explicit.
- [ ] **SCC name is misleading.** "Structural Change Coefficient" implies measuring *change* in structure, but Equation 4 shows it's just tr(Σ) = total variance across conformer features. It measures conformer *diversity* or *variation*, not *change*. The name will confuse readers who associate "structural change" with temporal or causal processes.
- [ ] **ETKDG** (Section 3.1) — never spelled out (Experimental Torsion-angle preference with Knowledge and Distance Geometry).
- [ ] **MMFF94** (Section 3.1) — never spelled out (Merck Molecular Force Field 94).
- [ ] **PMI, SASA, USR** — introduced in the abstract as "(PMI, SASA, USR)" but only SASA and USR are defined in Section 3.4. PMI is never spelled out (Principal Moments of Inertia).

### Logical / Argumentative Gaps

- [ ] **SCC is not scale-invariant.** tr(Σ) depends entirely on the units of the features in x_i. If distances are in Angstroms and angles in radians, the trace mixes incomparable scales. Normalizing features would change SCC values entirely. The "SCC > 25" threshold is therefore arbitrary and non-transferable to other feature encodings. This fundamental limitation is never discussed.

- [ ] **The "700+ experiments" claim is marketing.** 13 models × 4–8 datasets × 3–10 seeds easily produces 700+ runs. This framing inflates the perceived scope. Consider dropping this or replacing with a more precise description of the experimental matrix.

- [ ] **BDE failure is over-interpreted.** Section 4.6 and Discussion claim conformer geometry is "fundamentally uninformative for bond-breaking energetics" based on |ρ| = 0.044. But |ρ| measures *linear* correlation only. There could be nonlinear relationships that linear correlation cannot detect. Soften to "linearly uncorrelated" or use mutual information to support the stronger claim.

- [ ] **The claim that simple invariants beat complex representations (Section 4.8) lacks rigorous explanation.** The argument is: top-10 eigenvalues capture 4–8% of variance → PCA loses information → scalar invariants are better. But this doesn't follow. If information is spread across many dimensions, *more* dimensions should help, not fewer. The real reason scalar invariants win is likely regularization / overfitting on small datasets, not information content. This should be discussed.

- [ ] **Classification failure is buried in Limitations.** "Classification tasks (BACE, BBBP) showed degenerate behavior for DKO variants" — if the method fails entirely on classification, this is a major scope limitation that should be stated prominently (e.g., in the Introduction when scoping the contributions), not hidden in the final paragraph.

- [ ] **No train/val/test split details.** Section 4.1 describes datasets but never specifies how data is split. Are scaffold splits used? Random splits? This is critical for molecular property prediction — random splits are known to inflate performance via memorization of similar molecules.

- [ ] **"Cholesky-like factorization" (Section 3.2)** is imprecise. K = LL⊤ with L ∈ ℝ^{k×D} and k = 64 < D = 1024 is a low-rank PSD parameterization, not a Cholesky factorization (which produces lower-triangular L with k = D). Calling it "Cholesky-like" is misleading. Call it what it is: a low-rank kernel parameterization.

- [ ] **Why n = 50 conformers?** Section 3.1 states this without justification. Is this enough for conformational diversity? Does performance change with n = 10, 100, 200? This is a critical hyperparameter that is never ablated.

### Writing Quality Issues

- [ ] **Abstract is overloaded.** At ~200 words, it tries to cover: the benchmark finding (FP wins), the complementarity result, the taxonomy, the mechanistic analysis, AND the SCC metric. For a 6-page paper, focus on 2–3 key points. The abstract reads like a compressed version of the entire paper rather than a hook.

- [ ] **"5-scalar covariance invariants" in abstract is jargon.** A reader unfamiliar with the paper has no idea what this means. Rephrase: "five simple summary statistics of the covariance matrix."

- [ ] **Section 4.4 title "The Key Result" is editorializing.** Let the reader decide what's key. Use a descriptive title like "Hybrid FP + Conformer Complementarity."

- [ ] **Contribution (5) is inflated.** "Empirical decision rule: The SCC metric can provide a priori guidance" — this is one observation (threshold > 25), not a contribution on par with the framework or benchmark. It's also undermined by the FreeSolv contradiction (see Critical Errors). Consider merging into contribution (3) or (4).

- [ ] **Inconsistent dataset naming.** The paper uses "Lipo" (Tables 1, 4) and "Lipophilicity" (Section 4.1, Table 3, Discussion) interchangeably. Pick one.

- [ ] **The Discussion is well-structured** with italicized topic sentences (taxonomy, simplicity principle, practical implications) but the "Why attention beats DKO on Kraken but not ESOL" subsection feels wedged in. It breaks the taxonomy flow. Move it earlier or integrate it.

- [ ] **Conclusion and Limitations section conflates two things.** Separate them: have a brief Conclusion paragraph, then a Limitations paragraph. Currently the transition from "our central finding is complementarity" to "Our conformer features are derived from RDKit ETKDG conformers..." is abrupt.

- [ ] **15 references is unusually low** for a paper claiming a "comprehensive benchmark." Missing obvious citations: SchNet, DimeNet/DimeNet++, PaiNN (all mentioned in Limitations but never cited), SphereNet, GemNet, TorchMD-NET, Uni-Mol. The Related Work section needs significant expansion.

### Minor Writing Nits

- [ ] Abstract: "ESOL −9.9% RMSE" is ambiguous — reads as "RMSE of negative 9.9 percent." Rephrase to "9.9% RMSE reduction on ESOL."
- [ ] Section 3.4: "28 property-relevant 3D descriptors" — but the individual counts in each category don't visibly add to 28. Provide per-category counts (e.g., "6 shape, 5 surface, 12 USR, 5 global").
- [ ] Section 4.2: "The original DKO with PCA-compressed covariance ranks 12th" — burying the fact that your base method performs poorly is not ideal. Address it head-on: "The original DKO ranks 12th, motivating our eigendecomposition variants."
- [ ] Section 4.6: "a 6.7× gap" — gap in what metric? RMSE? R²? Unclear.
- [ ] The 5 scalar invariants (trace, log-determinant, Frobenius norm, top eigenvalue ratio, spectral ratio) — note that some are mathematically redundant. Frobenius norm² = Σλ_i², trace = Σλ_i. These aren't independent. Briefly acknowledge this.
- [ ] "QM9-Gap/HOMO/LUMO" — never clarified whether these are HOMO, LUMO, and HOMO-LUMO gap, or something else. Define once.

---

## Part 2: Figures & Formatting

### Critical Issue: No Figures At All

- [ ] **The paper contains zero figures.** This is a major weakness. For a paper asking "when does conformer geometry help?", visual communication is essential. At minimum, add:
  - **(1) A DKO architecture diagram** showing the pipeline: conformers → features → μ/Σ → kernel → branch prediction → fusion. Without this, the method is hard to visualize.
  - **(2) A performance comparison plot** (e.g., grouped bar chart or radar plot) showing FP-only vs. hybrid across datasets. This would make the complementarity finding immediately visible instead of requiring readers to parse Table 3.
  - **(3) A property taxonomy visual** — a simple 2×2 or categorized diagram showing which property types benefit from conformer geometry. This is the paper's most actionable contribution and deserves a figure.
  - **(4) An SCC vs. improvement scatter plot** showing the relationship between conformer diversity and hybrid improvement across datasets. This would make the SCC decision rule tangible (or expose its weakness).

### Table Issues

- [ ] **Table 1 uses only 3 seeds.** For 13 models compared by average rank, 3 seeds provides very low statistical power to distinguish ranks. The mean±std format is appropriate but the standard deviations show that many models have overlapping confidence intervals (e.g., dko_gated 1.635±0.023 vs. dko_router 1.670±0.023 on ESOL). State explicitly when rank differences are/aren't significant.

- [ ] **Table 2 bold formatting is inconsistent.** FP+XGB values are bolded in all rows, even though the point is that FP beats neural methods. Bold the *better* value in each row, or bold neither and let the Gap column speak.

- [ ] **Table 3's Δ row has "−" entries without explanation.** Add a footnote: "−: no improvement observed" or whatever it means.

- [ ] **Table 5 shows only 3 of 13 models.** Why? Where are the other DKO variants on MARCEL? If they were run but omitted, state why. If not run, say so. As-is, it looks like cherry-picking.

- [ ] **Table 6 (Appendix A)** introduces model names ("single_conformer", "deepsets", "dko_separate") that aren't defined anywhere in the paper.

- [ ] **Table 7 (Appendix C)**: The "low" value for FreeSolv Max is not a number. Fix immediately.

### Formatting / Layout

- [ ] **Appendix B ("Bug Discovery Narrative") is unusual and risky.** Describing 5 implementation bugs in your own method raises questions about code quality and what bugs remain undiscovered. While transparency is admirable, reframe this as "Implementation Ablation" or "Hyperparameter Sensitivity Analysis" — describing the same findings as systematic investigation rather than bug reports. The current framing ("Bug Discovery") undermines confidence.

- [ ] **Appendix C (SCC Analysis)** is only 1 table and 2 sentences. Either expand it to be substantive or fold it into the main text.

- [ ] **No code/data availability statement.** Even for blind review, state "Code will be released upon acceptance" or provide an anonymous repository link.

- [ ] **No reproducibility checklist.** Many ML venues now require one. Check submission requirements.

---

## Summary Assessment

**Strengths:** The fingerprint baseline result is genuinely important and honestly reported. The property taxonomy (solvation vs. steric vs. electronic) is the paper's most useful contribution. The complementarity finding is well-supported on solvation tasks. The mechanistic analysis adds depth beyond pure benchmarking.

**Weaknesses:** Zero figures makes the paper unusually hard to parse. The SCC decision rule is contradicted by FreeSolv. Feature encoding is critically underspecified. The "Bug Discovery" appendix undermines confidence. Missing train/test split details and important baselines (3D GNNs).

**Top 7 fixes before submission (priority order):**

1. Fix Table 7's "low" placeholder — this is a desk-reject risk
2. Add at least 2 figures (architecture diagram + performance comparison)
3. Specify the feature encoding (how x_i ∈ ℝ^1024 is constructed)
4. Resolve the SCC vs. FreeSolv contradiction — either refine the rule or drop Contribution (5)
5. Add train/val/test split details
6. Define BACE or remove it from Section 4.8
7. Rename Appendix B from "Bug Discovery Narrative" to "Implementation Ablation"
