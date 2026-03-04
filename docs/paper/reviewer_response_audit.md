# Reviewer Response Audit

Cross-check of `reviewer_response.tex` claims against `template.tex` (manuscript) and original reviewer comments.

---

## Issues Found

### 1. CRITICAL: Seed study uses different parameters than main results (Models B & C)

The **main results** for Models B and C use:
- Model B: **U=0.5, P=0.7, R=0.05** (3 free params: U, P, R) — Section 3.3
- Model C: **P=0.7, rho=0.5, R=0.05** — Section 3.4

But the **seed study** (Table 6) shows:
- Model B: **R=0.010**, only **P and R** (2 params, U missing/fixed)
- Model C: **R=0.010** (not 0.05)

The true value of R differs by 5x between the main results and the seed study. Also, Model B's seed study has 2 parameters while the main results have 3. The response claims "Across all four models and all parameters, posterior means are stable" — but the seed study was run on a different configuration than the main results. Either the seed study table needs updating to match the main results, or the discrepancy needs to be acknowledged.

**Action**: Rerun the seed study with the same parameters as the main results (R=0.05, 3 params for Model B), or update the response to note the seed study used different configurations and explain why.

### 2. MODERATE: Missing forward reference to diagnostics in Section 2.3

The response to R2 Comment 4 claims: *"We also reference the Cranmer et al. diagnostic protocol in the Implementation Details of Section 2.3, with a forward reference to the diagnostics results."*

This reference is **not present** in the manuscript. The Implementation Details section (lines 283-288) mentions neural spline flows, single-round NPE, and the GitHub repo, but contains no mention of Cranmer et al. or a forward reference to Section 3.5. The Cranmer reference only appears in the Diagnostics section itself (Section 3.5, line 537).

**Action**: Add the Cranmer et al. forward reference to Implementation Details, or remove that claim from the response.

### 3. MODERATE: Code repository claim may be premature

The response to R2 Comment 2 says: *"The code repository has been updated to include the missing result files needed to run the demonstration notebooks."*

However, `response_plan.md` Item 24 shows these sub-items are still **unchecked**:
- `results_extracted.pkl` retrieval: marked "MANUAL ACTION REQUIRED"
- "Verify all notebooks run end-to-end without errors" — unchecked
- "Ensure README has clear setup/installation instructions" — unchecked

**Action**: Verify the code repository actually has the missing files before submitting.

### 4. MINOR: "weakly informative" wording claim doesn't match manuscript

Response to R2 Comment 14 says: *"The wording has been changed to 'weakly informative'."*

The phrase "weakly informative" does **not appear** anywhere in the manuscript. The problematic "minimally informative" has been removed and replaced with "broad uniform priors determined by the physical constraints" (line 287), which is fine — but the response's claim about the specific replacement is inaccurate.

**Action**: Either add "weakly informative" to the manuscript, or update the response to accurately describe the replacement (e.g. "The wording has been changed to 'broad uniform priors determined by the physical constraints'").

### 5. MINOR: Timing table 1D/2D claim is overstated

Response to R2 Comment 17 says: *"for all four models in both 1D and 2D configurations"*

Table 7 actually has only **4 NPE rows** (one per model), not 8 (1D+2D). The caption notes that simulation time is the same for both formats, which is correct, but the response implies the table shows both configurations explicitly.

**Action**: Soften the claim, e.g. "for all four models (simulation time is independent of the 1D/2D output format, as noted in the table caption)."

### 6. MINOR: R2 additional point 4 not addressed

R2's additional point 4 states: *"Paragraph 1 of Section 3.2.1 is unclear."* This is not explicitly addressed in the response letter. It's resolved by the IMRaD restructure (old Section 3.2.1 no longer exists as such), but a brief acknowledgment would be appropriate.

**Action**: Add a line under Comment 13 or as a separate item, e.g. "The unclear paragraph in the former Section 3.2.1 has been rewritten as part of the Conditional Normalizing Flows subsection (Section 2.3.2)."

### 7. MINOR: "C2ST scores uniformly near 0.5" is slightly generous

The response to R1 Comment 3 claims *"C2ST scores are uniformly near 0.5"*. Table 4 shows values ranging from 0.55 to **0.64** (Model C 2D, rho). While 0.5-0.6 is conventionally "near 0.5" in SBI, 0.62-0.64 is getting marginal. Consider softening.

**Action**: Soften to "close to 0.5" or "between 0.5 and 0.65, indicating well-learned posteriors."

---

## Verified as Correct

- IMRaD structure (Sections 1-4)
- Four models with correct parameters in Sections 3.1-3.4
- Table numbering (Tables 1-7) and figure numbering (Figures 1-8) all match claims
- Hyperparameter table values (Table 2) match claimed specifics (NSF, 128/256 hidden, 8 transforms, batch 512, lr 1e-4, 100 epochs, patience 20, 10k/50k sims)
- Discussion subsections 4.1, 4.2, 4.3 exist with claimed content
- NPE vs NRE discussion in Section 4.1 with Lueckmann et al. citation
- SMC-ABC expanded description in Section 2.2
- "Fundamentally different" changed to "distinct approach" (line 217)
- Table 1 column renamed to "Per-observation Cost" with footnote
- Boelts et al. 2025 JOSS reference correctly updated in .bib
- Abbreviations spelled out in abstract (NPE, CNN)
- "Deceptively simple" and "familiar" removed
- Neural Spline Flows described in Section 2.3.2
- SNPE section commented out, single-round NPE clarified
- Amortized definition at first use in Introduction (line 113)
- Separate 1D/2D subfigures in Figures 4-7
- SBC plots in Figure 8, diagnostics in Table 4
- PPC in Appendix A
- Timing percentages verified: 74% for original, ~97% for exclusion models
- All reviewer comments addressed (except the one noted in Issue 6)
- Seed study details correct for Original model and Model A
- Model C correctly described as having no tractable likelihood
- Figure captions match claimed content
- SBI reference (Boelts2025sbi) correctly points to JOSS 10(108), 7754
