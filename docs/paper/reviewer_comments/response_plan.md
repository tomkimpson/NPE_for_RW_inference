# Response Plan for Reviewer Comments

## Strategic Decision (Resolved)

- **Venue**: Stay with Journal of Theoretical Biology
- **Structure**: Full IMRaD restructure
- **Framing**: Biology-first paper (cell migration inference problem), not a methods paper
- **Parameters**: Follow S&P2025 convention — infer U (initial condition) without growth; fix U with growth
- **NPE vs NRE**: Discussion paragraph only, no new experiments
- **1D vs 2D**: Selective — apply 2D CNN to one new model (bias model) where spatial anisotropy matters

---

## High Priority

### Item 1: Full IMRaD Restructure + Biological Reframing [R1 + R2]

Both reviewers question the paper's fit for JTB. R1: "significant importance on the biology per se being presented." R2: scope too narrow as a methods paper. The fix is to restructure as a biology paper that uses NPE, not a methods paper that uses biology.

**New paper structure:**

- **Introduction**: Pose the biological research question first (cell migration inference from scratch/barrier assay data), then introduce SBI/NPE as the tool to answer it
- **Methods**:
  - (a) Random walk models on 2D lattice with increasing complexity (4 models)
  - (b) Classical inference approaches and their limitations (MLE, profile likelihood, MCMC, ABC)
  - (c) Neural posterior estimation (NPE) method
  - (d) Summary statistics and CNN architecture
- **Results**: Systematic application across 4 models with benchmarking against classical methods
- **Discussion**: Biological insights, when SBI is necessary vs classical methods sufficient, limitations

**Sub-tasks:**

- [x] Rewrite Introduction to lead with biology (scratch/barrier assay problem, cell migration, wound healing)
- [x] Adopt IMRaD section headings throughout
- [x] Move all RW model descriptions into Methods
- [ ] Present results model-by-model, each with biological interpretation (not just parameter recovery)
- [x] Merge Discussion + Conclusion into a single Discussion section
- [ ] Ensure biological insights are emphasized throughout — what do inferred parameters tell us about cell behaviour?

---

### Item 2: Add 3 New RW Models [R1 + R2]

R1 wants "a series of examples, including comparison and benchmarking." R2 notes the current model "does not strictly require simulation-based inference." Adding models of increasing complexity culminates in one where classical methods genuinely fail.

**Shared setup for all models:**

- 2D square lattice: I rows, J columns, spacing Delta, timestep tau
- Crowding = exclusion process (max 1 agent per site)
- Noise model: binomial (exclusion means 0 <= N_i <= H, where H = number of sites in column i)
- Boundary conditions: zero-flux (same as current model)

---

#### Model A — Crowding + Bias + No Growth

- **Mean-field PDE**: du/dt = D d^2u/dx^2 - v d[u(1-u)]/dx
- **Lattice parameters**: movement probability P, bias rho (probability of moving in preferred direction vs other)
- **Continuum relationships**: D = P Delta^2 / (4 tau), v = P rho Delta / (2 tau)
- **Parameters to infer**: U, D, v (U is identifiable without growth, per S&P2025)
- **Prior**: BoxUniform over ranges TBD (match S&P2025 Table 2 ranges)
- **Summary statistics**: 1D column counts at observation time(s)
- **Also apply 2D CNN** to this model (see Item 6) — bias creates anisotropic spatial patterns that 2D should capture better than 1D column counts
- **Benchmark vs**: S&P2025 classical results if available for this model

#### Model B — Crowding + No Bias + Growth (Fisher-Kolmogorov)

- **Mean-field PDE**: du/dt = D d^2u/dx^2 + r u(1-u)
- **Lattice parameters**: movement probability P, proliferation probability R
- **Continuum relationships**: D = P Delta^2 / (4 tau), r = R / tau
- **Parameters to infer**: D, r (fix U as known — identifiability issue with growth per S&P2025)
- **Prior**: BoxUniform over ranges TBD
- **Summary statistics**: 1D column counts
- **Note**: This is the Fisher-Kolmogorov model. S&P2025 case study applies this to real scratch assay data.
- **Benchmark vs**: S&P2025 MLE/MCMC results

#### Model C — Crowding + Bias + Growth (Novel)

- **Mean-field PDE**: du/dt = D d^2u/dx^2 - v d[u(1-u)]/dx + r u(1-u)
- **Lattice parameters**: P, rho, R
- **Continuum relationships**: D = P Delta^2 / (4 tau), v = P rho Delta / (2 tau), r = R / tau
- **Parameters to infer**: D, v, r (fix U — growth present)
- **Prior**: BoxUniform over ranges TBD
- **Summary statistics**: 1D column counts
- **NOTE: This model is NOT in S&P2025.** This is where NPE genuinely shines: 3-parameter inference with complex nonlinear dynamics and no tractable likelihood. No classical benchmark available — this demonstrates when SBI is *necessary*, not just convenient.

---

**Code changes required:**

- [ ] `src/simulator.py`: Add 3 new simulator classes (or parameterize existing class)
  - Implement exclusion (crowding): check target site is empty before moving agent
  - Implement bias: directional movement preference via parameter rho
  - Implement proliferation: agents attempt to place daughter cell in random adjacent site (blocked if occupied, per exclusion)
  - Zero-flux boundary conditions (same as current)
- [ ] `src/inference.py`: Extend to support variable parameter dimensions (2-param for Model B, 3-param for Models A and C)
  - Update prior definitions for each model
  - May need separate NPE instances per model, or a parameterized approach
- [ ] `src/main.py`: Add model selection (CLI flag or config parameter)
- [ ] Generate training data for each of the 3 new models
- [ ] Train NPE for each of the 3 new models
- [ ] Produce posterior plots, coverage tables, and comparisons with classical methods where available

---

### Item 3: SBI Diagnostics Workflow [R2]

R2: "The analysis does not report diagnostic of the simulation-based inference... which would, in general, be a problem." References Cranmer et al. (Reference 15 in manuscript).

- [ ] Implement simulation-based calibration (SBC) checks:
  - Draw theta from prior, simulate x|theta, obtain posterior samples, compute rank statistics
  - Rank statistics should be uniformly distributed if posterior is well-calibrated
- [ ] Implement expected coverage tests:
  - For credible intervals at levels (e.g., 50%, 80%, 90%, 95%), check that empirical coverage matches nominal level
- [ ] Reference Cranmer et al. guidelines explicitly in text
- [ ] Report diagnostics for all 4 models (existing model + 3 new)
- [ ] Present as a subsection in Results, or as supplementary material if space-constrained

---

### Item 4: Justify NPE over Other SBI Methods [R2]

R2: "It is not clear why authors chose NPE specifically? What about other algorithms? (e.g., NRE)"

- [x] Add a paragraph in Discussion (or Methods) comparing NPE vs NRE vs ABC-based SBI methods
- [x] Key points to make:
  - NPE learns the posterior directly (amortized) — once trained, posterior for any new observation is immediate
  - NRE learns the likelihood-to-evidence ratio — also amortized, but requires MCMC sampling to obtain posterior
  - NPE is more natural for multi-observation problems where you want posteriors for many datasets
  - NPE has a proven track record for scientific inverse problems
- [x] Cite relevant comparison papers (e.g., Lueckmann et al. 2021 benchmarks)
- [x] Justify NPE for this specific application: direct posterior samples, fast amortized inference, good scaling with observation count
- [x] **Decision**: Discussion paragraph only — no NRE experiments needed

---

## Medium Priority

### Item 5: Fix Figure 6 / 1D vs 2D Overlay [R2]

R2: "Section 4.3.2 states: 'Figure 6 shows the joint posterior distribution... overlaid with the result from the 1D column count analysis.' However, this does not seem to be the case."

- [ ] Check `docs/paper/template.tex` Section 4.3.2 and Figure 6
- [ ] Either fix Figure 6 to actually show the 1D/2D overlay comparison
- [ ] Or correct the text to match what the figure currently shows

---

### Item 6: 2D CNN on Bias Model [R2]

R2: "It would be much more convincing to present a case in which switching from 1D to 2D actually impacts the quality of the parameter estimate."

This item directly addresses R2's criticism that the 2D approach showed no improvement over 1D.

- [ ] Apply 2D CNN approach to Model A (crowding + bias + no growth)
- [ ] Rationale: bias creates anisotropic spatial patterns (agents preferentially moving in one direction), which 2D data should capture but 1D column counts collapse
- [ ] Compare 1D column-count posteriors vs 2D CNN posteriors for Model A — expect 2D to give tighter posteriors, especially for bias parameter v
- [ ] Reframe the narrative: 2D is valuable when spatial structure is informative (bias, clustering, anisotropy), not always
- [ ] Code: extend existing CNN pipeline to handle Model A's 2D lattice output as input
- [ ] **Depends on**: Item 2 (Model A must be implemented first)

---

### Item 7: Hyperparameter Table in Paper [R2]

R2: "It would be helpful to include a summary table of the hyperparameters in the paper itself."

- [ ] Create table listing:
  - Network architecture: Neural Spline Flows
  - Hidden features: 128
  - Number of transforms: 5
  - Batch size: 512
  - Learning rate: 1e-4
  - Training epochs
  - Number of simulations used for training
- [ ] Add to Methods section or Appendix
- [ ] Values currently only in code repository — move into paper

---

### Item 8: Separate Simulation vs Training Time [R2]

R2: "It would be informative to separate the training of the network and the generation of the simulation for training, since simulations are usually the bottleneck."

- [ ] Split current Table 3 into three components:
  - (a) Simulation generation time
  - (b) Network training time
  - (c) Amortized inference time (per observation)
- [ ] Re-run timing benchmarks if current data does not separate these
- [ ] Report for all 4 models

---

### Item 9: Reproducibility Across Seeds [R2]

R2: "It is unclear from the text whether training and parameter inference were performed only once, or if repeating the experiment with different seeds would lead to similar results."

- [ ] Run each model's training + inference pipeline with multiple random seeds (e.g., 5 seeds)
- [ ] Report mean +/- std of key metrics:
  - Posterior means for each parameter
  - Credible interval widths
  - Any summary accuracy metric used
- [ ] Add statement about reproducibility to Methods or Results section

---

### Item 10: Define "Amortized" Earlier [R2]

R2 notes the word is used on p3 and in Table 1 before its definition on p7.

- [ ] Move definition of "amortized" to its first use in the Introduction
- [ ] Suggested phrasing: "amortized inference (where the upfront cost of training is offset by near-instant posterior estimation for any new observation)"
- [ ] Affects: `docs/paper/template.tex` Introduction section

---

### Item 11: Update SBI Package Reference [R2]

- [ ] Change Reference 63 to: Boelts et al., JOSS, https://joss.theoj.org/papers/10.21105/joss.07754
- [ ] Affects: `docs/paper/template.tex` bibliography / .bib file

---

### Item 12: Strengthen 2D vs 1D Argument for Original Model [R2]

R2: "Both the 1D column-wise approach and the 2D spatial approach use 'manually selected summary statistics.'"

- [ ] Reframe the argument: the 2D CNN approach automates feature extraction — you pass raw spatial data and the CNN learns relevant features, unlike 1D column counts which are hand-crafted
- [ ] Acknowledge that 1D column counts discard spatial information (clustering, anisotropy, local density fluctuations)
- [ ] Combined with Item 6, present a coherent narrative: 2D approach is valuable when spatial structure is informative, and its main advantage is automation of feature extraction even when it matches 1D precision

---

## Low Priority

### Item 13: Fix Abbreviation Issues [R1]

- [x] Remove ALL abbreviations from the abstract (spell out MLE, MCMC, ABC, NPE, CNN, PDE in full) — Spelled out NPE and CNN in full throughout the abstract
- [x] Ensure each abbreviation is defined exactly once at first use in the main text body — Verified, definitions in Introduction are correct
- [x] Check for duplicate definitions (e.g., defining "NPE" in both Introduction and Methods) — No duplicates found

---

### Item 14: Fix Table 1 "Cost per Inference" vs "Amortized" [R2]

R2: "Cost per Inference vs. Amortized - are they the same thing?"

- [x] Clarify these are different concepts:
  - "Amortized" = property of the method (train once, infer many times)
  - "Cost per inference" = the resulting metric (time/compute per posterior estimate)
- [x] Revise column headers or add a footnote explaining the distinction — Renamed to "Per-observation Cost" and added footnote to "Amortized?" column

---

### Item 15: Expand Table 2 with Comparative Results [R2]

R2: "Include results from alternative models (ABC, MCMC, etc) and the true values."

- [x] Add columns/rows for ABC and MCMC posterior estimates alongside NPE — Added rows for ABC, Surrogate+MLE/Laplace, and Surrogate+MCMC with qualitative references to Figure 3
- [x] Include ground truth parameter values in table — Added true value row (U=0.3, D=0.175)
- [x] Extend table to cover all 4 models (existing + 3 new) — Restructured table with Method and Data Type columns; new models deferred to Item 2

---

### Item 16: Fix Section Naming / Structure [R2]

- [x] Section 2.3 "Example Results" — replace with more descriptive title — Resolved by IMRaD restructure (Item 1)
- [x] Section 3.3 describes SNPE but says "we will exclusively use sequential NPE" — clarify sequential vs non-sequential, or just say NPE — Replaced with explanation of proposal-correction mechanism
- [x] Add a description of Neural Spline Flows (currently used but not described in text) — Added description: monotonic rational-quadratic spline transformations with analytic Jacobians
- [x] **Note**: Most of these issues will be resolved naturally by the IMRaD restructure (Item 1)

---

### Item 17: Clarify Section 3.2.1 Paragraph 1 [R2]

- [x] Review and rewrite paragraph 1 of Section 3.2.1 for clarity — Tightened first paragraph of Conditional Normalizing Flows section
- [x] R2 flagged this as unclear — check current `docs/paper/template.tex` Section 3.2.1 — Removed redundant framing, focused on technical content (invertible maps, tractable Jacobians, change-of-variables)

---

### Item 18: Fix SMC-ABC Mention [R2]

Section 2.1 mentions "More efficient variants of ABC, such as SMC-ABC" without further discussion.

- [x] Either expand to briefly explain SMC-ABC (sequential Monte Carlo ABC — iteratively refines proposal distribution to improve acceptance rate) — Expanded: iteratively refines proposal distributions, progressively lowers tolerance
- [ ] Or remove the mention if it's not relevant to the narrative — N/A, chose to expand
- [x] In the revised IMRaD structure, this goes in Methods under "classical inference approaches" — Already in correct location after IMRaD restructure

---

### Item 19: Fix "Minimally Informative Prior" Statement [R2]

Section 3.5.1: "uniform priors represent a minimally informative choice" — imprecise.

- [x] Revise to "weakly informative" or "relatively uninformative" — Already says "weakly informative" (line 284)
- [x] Optionally acknowledge that Jeffreys prior is the formal uninformative choice for continuous parameters, but uniform is standard in SBI applications — Not needed, current wording is appropriate
- [x] Affects: `docs/paper/template.tex` Section 3.5.1 — No changes needed, already correct

---

### Item 20: Remove Colloquialisms [R2]

- [x] Remove "deceptively simple" — Already removed in previous revision
- [x] Rephrase "familiar inverse transform sampling" (not all readers will find it familiar) — Already removed in previous revision
- [x] Do a general pass for colloquial or imprecise language throughout — Done in IMRaD restructure

---

### Item 21: Fix Section 3.1 "Fundamentally Different" [R2]

R2: "'NPE offers a fundamentally different approach' is incorrect or unclear."

- [x] NPE is a specific SBI method, not a fundamentally different paradigm from SBI itself — Already says "distinct approach" (line 216)
- [x] Rephrase to clarify NPE's contribution within the SBI framework (e.g., "NPE takes a distinct approach within SBI by directly estimating the posterior distribution using normalizing flows, rather than approximating the likelihood or relying on accept-reject sampling") — Already addressed in previous revision

---

### Item 22: Figure Placement and Captions [R1]

- [x] Move Figure 5 closer to its first reference in the text — Placement now correct after IMRaD restructure
- [x] Fix Figure 2 sub-captions — move panel descriptions into the main caption body — Moved subfigure caption text into main caption as (a)..., (b)..., (c)... descriptions; subfigures now use empty \caption{} for label only
- [x] Review R1's annotated PDF for any other specific figure issues — No additional issues found

---

### Item 23: Technical Typos [R1]

- [x] Review R1's annotated PDF (`JTB-D-25-00973_review-commented.pdf`) for all highlighted typos — Addressed colon-before-equation issues
- [x] Fix formula integration into sentences — no colon before a formula when the sentence continues after it — Removed colons at six locations: PDE form, diffusion equation, Fisher Information matrix, change of variables formula, proposal distribution bias, agents per column
- [ ] Do a final proofreading pass

---

### Item 24: Fix Missing Files in Code Repository [R2]

- [ ] Add `notebooks/example_results/results_extracted.pkl` to the repository — **MANUAL ACTION REQUIRED**: Retrieve from SLURM cluster. Also need `results_extracted_1D.pkl`, `results_extracted_2D.pkl`, `results_extracted_2D_v2.pkl`
- [ ] Verify all notebooks run end-to-end without errors
- [ ] Ensure README has clear setup/installation instructions

---

## Execution Order

### Phase 1 — Code (Items 2, 3, 9)

1. Implement 3 new simulator classes in `src/simulator.py` (exclusion, bias, proliferation mechanics)
2. Extend `src/inference.py` for variable parameter dimensions (2-param and 3-param models)
3. Add model selection to `src/main.py`
4. Generate training data for all 3 new models
5. Train NPE for all 4 models (existing + 3 new)
6. Run SBI diagnostics (SBC + coverage) for all 4 models
7. Run reproducibility tests (5 seeds per model)

### Phase 2 — Analysis (Items 6, 8, 15)

8. Apply 2D CNN to bias model (Model A), compare 1D vs 2D posterior precision
9. Collect timing data separated into simulation / training / inference components
10. Collect classical method results (MLE, MCMC) for Table 2 comparisons

### Phase 3 — Writing (Items 1, 4, 5, 7, 10-24)

11. IMRaD restructure of paper (this is the biggest single writing task)
12. Write all new sections: new models in Methods, new results, expanded Discussion
13. Create new tables (hyperparameters, expanded Table 2, split timing table)
14. Fix Figure 6 overlay issue
15. Address all text-level reviewer comments (Items 10, 13-14, 16-21)
16. Fix figures, captions, abbreviations, typos (Items 13, 22-23)
17. Fix code repository issues (Item 24)
18. Final polish pass — read full paper end-to-end for coherence
