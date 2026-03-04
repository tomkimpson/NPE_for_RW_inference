# Claims Review: `template.tex`

Systematic review of paper claims against the actual evidence. Each issue is tagged by severity and location.

---

## Critical: 1D vs 2D confound

All 1D results use **10k training sims with 128 hidden features**. All 2D results use **50k training sims with 256 hidden features**. Any comparison of 1D vs 2D posterior precision is confounded by a 5x difference in training budget and 2x difference in network capacity. This affects multiple claims throughout the paper.

---

## Factually incorrect

### Table 2 (line ~321): Training simulations listed as 50,000 for 1D
The table body says "Training simulations: 50,000" for the 1D column. The actual 1D runs used 10,000. The red note in the caption acknowledges this, but the table body is wrong and will mislead any reader who doesn't see the red note (which will be removed before submission).

**Fix:** Either update the table to reflect actual values (10k) or rerun 1D with 50k first and then the table is correct.

---

## Unsupported due to confound (high severity)

### 1. Figure 3 caption (line ~398): "as informative as the curated 1D summary statistics"
> "Both approaches recover comparable posteriors, demonstrating that the CNN automatically extracts features from the raw spatial data that are as informative as the curated 1D summary statistics."

The 2D pipeline used 5x more training data and a 2x larger network. Despite this advantage, 2D posteriors are comparable or slightly *wider* for the Original model. The correct reading is that column counts are near-sufficient for the isotropic model and the CNN is less sample-efficient — the opposite of the implied claim.

**Fix:** Reframe: *"Both approaches recover comparable posteriors, indicating that the CNN can automatically extract features from the raw spatial data without requiring hand-crafted summary statistics."*

### 2. Section 3.1.3 (line ~409): "precision that is highly comparable"
> "leveraging the full spatial data yields a posterior with a precision that is highly comparable to that obtained using curated 1D summary statistics"

Same confound. With 5x more training data, "comparable" precision actually implies less efficiency per simulation.

**Fix:** Remove "highly" and add caveat about differing training budgets, or rerun 1D first.

### 3. Section 3.3 (line ~479): "spatial pattern information helps break the U–R degeneracy"
> "The 2D representation narrows the U posterior considerably compared to 1D, suggesting that spatial pattern information helps break the U–R degeneracy."

Cannot attribute the narrowing to spatial information when training budget differs 5x.

**Fix:** After rerunning 1D with 50k, this claim may or may not hold. If it holds, keep it. If not, reframe as: *"The 2D representation recovers U with comparable precision to 1D, and the U–R partial degeneracy is present in both representations."*

### 4. Section 3.4 (line ~509): "demonstrating the additional constraining power of spatial information"
> "The 2D posteriors are substantially tighter, particularly for P and ρ, demonstrating the additional constraining power of spatial information in this four-parameter setting."

Strongest over-claim in the paper. "Demonstrating" implies a controlled comparison that does not exist.

**Fix:** After 50k 1D rerun, if 2D is still tighter, reframe as: *"The 2D posteriors are tighter for P and ρ, suggesting that spatial information provides additional constraining power in this four-parameter setting."* If the difference vanishes, remove the claim.

### 5. Discussion (line ~726): "genuine additional constraining power"
> "2D posteriors for P and ρ are roughly two to four times tighter than 1D posteriors (Table 6), demonstrating that spatial information provides genuine additional constraining power in the most complex model."

This is the single most over-egged sentence in the paper. "Genuine" and "demonstrating" imply controlled evidence that doesn't exist.

**Fix:** Depends entirely on 50k 1D rerun results.

---

## Over-egged (moderate severity)

### 6. Abstract (line ~95): "well-calibrated uncertainty estimates"
> "recovering biologically interpretable parameters ... with well-calibrated uncertainty estimates"

The SBC diagnostics (Table 4) show multiple KS failures across models. The diagnostics section itself discusses "mild overconfidence." Calling the estimates "well-calibrated" overstates the evidence.

**Fix:** *"with uncertainty estimates validated through simulation-based calibration diagnostics"*

### 7. Introduction (line ~113): "removing the need for hand-crafted summary statistics"
> "removing the need for hand-crafted summary statistics"

The CNN offers an *alternative*. Summary statistics still work well (and arguably better for some models). "Removing the need" implies they are no longer useful.

**Fix:** *"reducing reliance on hand-crafted summary statistics"*

### 8. Section 2.4 (line ~299): "maximally informative"
> "This end-to-end training ensures that the features extracted by the CNN are maximally informative for parameter inference."

"Maximally informative" is an information-theoretic claim that has not been demonstrated. The features are optimized, not proven maximal.

**Fix:** *"ensures that the features extracted by the CNN are optimized for parameter inference"*

### 9. Discussion conclusion (line ~744): "substantial information loss"
> "removing the need for hand-crafted summary statistics that risk substantial information loss"

For the Original model and Model A, column counts lose little or no information. "Substantial" is unjustified by the evidence.

**Fix:** *"removing the need for hand-crafted summary statistics that may discard information"*

---

## Misleading by omission (lower severity)

### 10. Section 3.2 — Model A 2D results (line ~449): shifted medians not flagged
The 2D medians for Model A are P = 0.801 (true 0.7) and ρ = 0.416 (true 0.5). Both are further from truth than the 1D medians (P = 0.627, ρ = 0.593). The text discusses the P–ρ degeneracy well, but does not acknowledge that the 2D medians sit in a different region of the degenerate manifold, further from the true values. This is related to the diagnostic failures (all three KS tests fail for 2D Model A).

**Fix:** Add a sentence: *"We note that the 2D medians for P and ρ lie in a different region of the degenerate manifold compared to 1D, reflecting the sensitivity of the posterior mode to the data representation when parameters are weakly identifiable."*

### 11. Diagnostics discussion (line ~543): generic explanation for 2D failures
> "These failures likely reflect the challenge of learning accurate posterior widths in the higher-dimensional observation space."

This is reasonable as a general statement, but for Model A specifically the failures appear to be related to CNN-induced bias along the degenerate P–ρ manifold, not just posterior width. A model-specific comment would be more informative.

**Fix:** Add after the generic explanation: *"For Model A, where P and ρ are strongly degenerate, the KS failures may additionally reflect the CNN concentrating posterior mass on a different region of the degenerate manifold than the 1D embedding, as evidenced by the shifted marginal medians (Section 3.2)."*

---

## Claims that ARE well-supported

For reference, these central claims are solid and well-evidenced:

1. **NPE is effective for inference across all four RW models using 1D data.** All true values within 95% CIs, 100% seed coverage, C2ST passing everywhere.
2. **NPE scales with model complexity** from 2-param Original to 4-param Model C.
3. **The CNN-NPE pipeline is a viable approach** for inference from 2D spatial data (works across all four models).
4. **Column counts are near-sufficient for isotropic models** — 2D data provides no additional constraining power for the Original model (this is actually *strengthened* by the confound, since 2D had more data and still couldn't beat 1D).
5. **The P–ρ degeneracy in Model A is physics-driven** and persists regardless of data representation.
6. **NPE avoids surrogate bias and noise model specification** — this is the core methodological advantage and is fully supported.
7. **NPE is amortized** — near-instant inference after training. Supported by runtime table.
8. **For Model C, surrogate-based methods are poorly validated** — NPE provides the most principled route. This framing is well-supported.

---

## Resolution path

The confound-related issues (items 1–5) all depend on the same fix: **rerun 1D with 50k training sims** (and ideally 256 hidden features to fully control the comparison). After that rerun:
- If 2D is still tighter for Models B/C → claims can be made with appropriate hedging
- If 2D is comparable → reframe CNN contribution as "viable alternative, not improvement"
- If 2D is worse → important finding about CNN efficiency that should be discussed honestly

The over-egged items (6–9) and omissions (10–11) can be fixed with text edits regardless of the rerun.
