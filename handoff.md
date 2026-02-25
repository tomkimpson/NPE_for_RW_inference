# Handoff: `reviewer-comments` branch

## Summary

Paper revision in progress. Core infrastructure complete: 4 random walk models (original + A/B/C), parallelized data generation, CNN embedding for 2D spatial data, SBI diagnostics, seed study, and classical baselines. Paper text largely written in IMRaD format. Currently refining figures and text for space/clarity.

---

## What was done

### 1. Multi-model infrastructure (earlier commits on this branch)

- **`src/simulator.py`**: Added `ExclusionRandomWalkSimulator` supporting crowding (exclusion process), directional bias (rho), and proliferation (R) on a 2D lattice with zero-flux boundary conditions.
- **`src/models.py`**: Created `ModelConfig` dataclass and `MODEL_CONFIGS` registry for all 4 models:
  - **original**: infer U, P (no exclusion)
  - **Model A**: infer U, P, rho (exclusion + bias, no growth)
  - **Model B**: infer U, P, R (exclusion + growth)
  - **Model C**: infer U, P, rho, R (exclusion + bias + growth, 4 parameters)
- **`src/inference.py`**: Generalized `RandomWalkNPE` for variable parameter dimensions via `ModelConfig`.
- **`src/main.py`**: Added `--model {original,A,B,C}` routing.
- **`src/predict.py`**: Generalized posterior predictive sampling for all models.
- **`src/utils.py`**: Generalized posterior statistics for N parameters.

### 2. Parallelized data generation (commits 92a7a46, b4b543e, 07c9398)

- Training data generation uses `ProcessPoolExecutor` with fork-based workers.
- Fixed CPU affinity bug: forked workers inherited restricted affinity from PyTorch/CUDA, starving them of cores. Fix restores affinity via `os.sched_setaffinity(0, range(os.cpu_count()))` in each worker.

### 3. Fixed injection values (commit 5936820)

Changed `DEFAULT_THETA_TRUE` in `src/main.py` to move true parameter values away from prior boundaries:
- P: 1.0 → 0.7 (models A, B, C)
- R: 0.001 → 0.01 (models B, C)

This prevents posteriors from railing at prior edges.

### 4. CNN embedding for 2D spatial data (commit 5936820)

Cherry-picked from the `2D_data` branch and integrated with the current multi-model + parallelization infrastructure:

- **`src/cnn_utils.py`**: `SpatialCNN` (4-stage residual conv net, 1→32→64→128→256 channels, stride-2 downsampling, global avg pool → FC layers → 256-dim output) and `SpatialEmbeddingNet` wrapper that handles sbi's internal observation flattening (reshapes `(batch, Ly*Lx)` back to `(batch, Ly, Lx)`).
- **`src/training_monitor.py`**: Training diagnostics (gradient norms, loss tracking).
- **`src/simulator.py`**: Added `use_2d_output` parameter and `get_2d_grid()` to both simulator classes.
- **`src/inference.py`**: Added `use_2d_data` / `spatial_dims` constructor params, conditional CNN embedding net creation, 2D-aware worker functions.
- **`src/main.py`**: Added `--use_2d_data` flag.

Usage: `sbatch slurm/run_production.sh A --use_2d_data`

**Known issue**: CNN conv2d operations fail on P100 GPUs (compute capability 6.0) with PyTorch 2.5.1 + CUDA 12.4. Works on V100/A100. The SLURM script targets A100s via `--partition=milan-gpu --gres=gpu:a100:1`.

### 5. ABC baseline (commit 5936820)

- **`src/abc_inference.py`**: `RandomWalkABC` class wrapping sbi's `SMCABC`. Includes `make_sbi_simulator()` factory that adapts our simulators to sbi's interface.
- **`src/main.py`**: Added `--method {npe,abc}` with ABC-specific args (`--abc_num_particles`, `--abc_num_simulations`, `--abc_num_initial_pop`, `--abc_epsilon_decay`).
- **`slurm/run_abc.sh`**: 8h wall time, 8 CPUs, 32GB RAM, no GPU.

**Strategic decision**: ABC is **not** being used as the paper's baseline. The exclusion-process simulator is too slow for ABC to converge within reasonable compute budgets (models A/B/C all timed out at 8 hours). Profile likelihood / MLE / MCMC from Simpson & Plank 2025 is the stronger classical comparison. The ABC code remains in the codebase for internal reference.

---

## Production run results

All 12 jobs submitted on 2026-02-04. Results in `results/`:

| Model | NPE 1D | NPE 2D (CNN) | ABC |
|-------|--------|--------------|-----|
| original | `workflow_original_npe_20260204_225605` | `workflow_original_npe2d_20260223_104550` | `workflow_original_abc_20260204_225446` |
| A | `workflow_A_npe_20260204_230502` | `workflow_A_npe2d_20260224_104426` (retrained) | `workflow_A_abc_20260204_225446` |
| B | `workflow_B_npe_20260224_082130` (R=0.05) | `workflow_B_npe2d_20260224_091648` (R=0.05, 50k) | `workflow_B_abc_20260204_225446` |
| C | `workflow_C_npe_20260224_171233` (4-param U,P,rho,R) | `workflow_C_npe2d_20260224_171232` (4-param, 50k) | `workflow_C_abc_20260204_225446` |

**Status**:
- All 8 NPE jobs (4 × 1D, 4 × 2D): **completed successfully**. All true values fall within 95% credible intervals.
- ABC original: **completed** (original model uses the faster non-exclusion simulator).
- ABC A/B/C: **timed out** at 8 hours. Exclusion simulator too slow (~1-2 sims/s) for SMCABC to converge.

Each successful NPE result directory contains: `config.txt`, `npe_model.pkl`, `training_data.pkl`, and under `inference_results/`: posterior samples, marginal/pairwise plots, prediction intervals, and predictive results.

**Note**: 2D runs require more training samples than 1D. The enhanced 2D settings (50k samples, hidden_features=256) have been validated on Model A and are now being applied to the original model.

---

## Production run configuration

**1D NPE** (`slurm/run_production.sh`):
- 10,000 training samples, 8 parallel workers
- NSF: 128 hidden features, 8 transforms
- Training: max 100 epochs, early stopping after 20, lr=1e-4, batch size 512

**2D NPE** (`slurm/run_production_2d_enhanced.sh`):
- 50,000 training samples, 16 parallel workers
- NSF: 256 hidden features, 8 transforms
- Training: max 100 epochs, early stopping after 20, lr=1e-4, batch size 512

**Common**: Lattice Lx=200, Ly=50, T=100, initial_region_half_width=25. Posterior: 5000 samples; predictive: 500 forward simulations. GPU: A100 on milan-gpu partition.

---

## Files changed/created on this branch

| File | Status | Description |
|------|--------|-------------|
| `src/main.py` | modified | Multi-model routing, fixed theta_true, --method abc, --use_2d_data |
| `src/simulator.py` | modified | ExclusionRandomWalkSimulator, 2D grid methods |
| `src/inference.py` | modified | ModelConfig support, parallelization, CNN embedding, 2D workers |
| `src/models.py` | created | ModelConfig dataclass + MODEL_CONFIGS registry |
| `src/predict.py` | modified | Multi-model posterior predictive sampling |
| `src/utils.py` | modified | N-parameter posterior statistics |
| `src/abc_inference.py` | created | SMCABC wrapper (not for paper) |
| `src/cnn_utils.py` | created | SpatialCNN + SpatialEmbeddingNet |
| `src/training_monitor.py` | created | Training diagnostics |
| `slurm/run_production.sh` | modified | A100 targeting, extra args passthrough |
| `slurm/run_abc.sh` | created | ABC SLURM script |

---

## Current status (2026-02-26)

### Claims review completed
- Systematic review of paper claims against evidence: **`docs/claims_review.md`**
- Found 1 factually incorrect item (Table 2 training sims), 5 claims unsupported due to the 1D/2D confound, 4 over-egged claims, and 2 misleading omissions.
- Core issue: all 1D results use 10k sims / 128 hidden features; all 2D results use 50k sims / 256 hidden features. Any 1D-vs-2D precision comparison is confounded.
- Well-supported claims: NPE works across all 4 models (1D), scales with complexity, CNN pipeline is viable, column counts are near-sufficient for isotropic models, P-ρ degeneracy is physics-driven, amortization advantage.
- Resolution: rerun 1D with 50k sims to enable fair comparison, then fix text. Over-egged wording (abstract, intro, discussion) can be fixed independently.

### Completed
- SBI diagnostics (SBC rank plots, KS tests, C2ST, TARP) for all 4 models
- Seed study (5 seeds per model, Table in paper)
- Classical baselines (ABC, surrogate+MLE/Laplace, surrogate+MCMC) for original model
- IMRaD restructure and full paper text
- Model A 2D re-run with enhanced settings (50k samples, hidden_features=256)
- **Standardized all corner plots to `corner.corner`**: All posterior figures (original, A, B, C) now use `corner.corner` as separate 1D/2D subfigures. Removed overlay figures (old Figures 4 and 8). Added P→D reparameterization for original model. `paper_figures.py` generates all 8 plots (`corner_{model}_{1d,2d}.png`). LaTeX updated with new image paths and references.

- **Original model 2D NPE re-run completed**: Re-ran with enhanced settings (50k samples, hidden_features=256). New result: `workflow_original_npe2d_20260223_104550`. D median improved from 0.131 → 0.156 (true 0.175, within 95% CI). Updated `paper_figures.py` RESULT_PATHS, regenerated corner plots, and updated all three locations in `template.tex` with new posterior statistics (U: 0.294±0.011/0.010, D: 0.156±0.026/0.023).

- **R injection increase completed (R=0.01 → R=0.05 for Models B & C)**: Changed `DEFAULT_THETA_TRUE` in `src/main.py`. Re-ran all 4 jobs (B 1D, B 2D, C 1D, C 2D) reusing existing training data. All true values within 95% CIs. Updated `paper_figures.py` RESULT_PATHS, regenerated corner plots, updated Sections 3.3-3.4, Table 6, and figure captions in `template.tex`. Also fixed R prior in table caption from U(0,0.02) to U(0.001,0.2). Note: Model C 2D used 10k training samples (from original run); all others use enhanced settings.

- **All diagnostics completed (2026-02-24)**:
  - Model B 1D: re-ran with 3-param (U, P, R). Results at `workflow_B_npe_20260224_082130/diagnostics/`.
  - Model C 1D: copied from `workflow_C_npe_20260204_225447/diagnostics/` (same training data + seed → identical model).
  - 2D diagnostics for all 4 models completed. Added `--use_2d` flag to `src/run_diagnostics.py` and `src/diagnostics.py`, extra args passthrough to `slurm/run_diagnostics.sh`.
  - Model A 2D retrained (old pickle incompatible with current `SpatialCNN`). New result: `workflow_A_npe2d_20260224_104426`.
  - Paper diagnostics table now shows both 1D and 2D results side-by-side. SBC figure overlays 1D (solid) and 2D (dashed) on same panels in 2x2 grid.

- **Model B seed study re-run (2026-02-24)**: Re-ran 5-seed study with 3-param model (U, P, R, R=0.05). Results at `results/seed_study/B/seed_{42,123,456,789,1024}/`. Updated reproducibility table in paper. All 5/5 coverage.

- **Model A 2D result path updated**: `paper_figures.py` RESULT_PATHS now points to retrained `workflow_A_npe2d_20260224_104426`. Model A 2D posterior values and correlation updated in paper text and Table 6.

- **Model C extended to 4 parameters (2026-02-25)**: Freed U as an inferred parameter. Model C now infers (U, P, rho, R) — 4 params, no fixed params. Creates clear complexity progression: original(2) → A(3) → B(3) → C(4). Changes:
  - `src/models.py`: param_names, prior bounds, removed fixed_params
  - `src/main.py`: DEFAULT_THETA_TRUE `[0.5, 0.7, 0.5, 0.05]`
  - 1D NPE: `workflow_C_npe_20260224_171233` — all 4 params in 95% CI
  - 2D NPE (50k): `workflow_C_npe2d_20260224_171232` — U, P, R in CI; rho slightly outside (0.392 vs true 0.5)
  - Diagnostics: completed for both 1D and 2D. C2ST scores all near 0.5. KS failures: P & R (1D), P & rho (2D)
  - Seed study (5 seeds): all 100% coverage for all 4 params. Results at `results/seed_study/C/seed_{42,123,456,789,1024}/`
  - `paper_figures.py` RESULT_PATHS updated, figures regenerated, paper text fully updated (Section 3.4, Tables 4/5/6, SBC discussion, figure captions)

- **SBC clamping fix and diagnostics re-run (2026-02-25)**: Fixed `src/diagnostics.py` to clamp raw flow samples to prior bounds before SBC rank computation (previously used unclamped flow samples, which violated prior support). Added flow leakage measurement to diagnostics output. Re-ran all 8 models (4 × 1D, 4 × 2D). Flow leakage was small (0.2–1.9%), ruling out leakage as the dominant cause of KS failures.

- **Model A 2D retrained with SPP + no standardization (2026-02-25)**: Retrained with `--disable_sbi_standardization --cnn_spatial_pyramid` using existing 50k training data. New result: `workflow_A_npe2d_20260225_140808`. TARP ATC improved dramatically (0.43→0.08, now passing). rho KS improved (0.00→0.02) but U KS unchanged. P KS worsened. Decision: accept results — C2ST passes everywhere, TARP passes, all true values in 95% CIs. See `docs/sbc_investigation.md` for full analysis.

- **Paper updated with new Model A 2D results**: Updated `paper_figures.py` RESULT_PATHS, regenerated corner/SBC figures, updated posterior statistics in text and Table 6, updated diagnostics Table and discussion text.

### In progress (2026-02-26)

- **1D NPE 50k rerun — jobs submitted, awaiting results**: Converted 2D training data to 1D via `src/convert_2d_to_1d.py` (summing along Ly axis). Converted data at `results/training_data_1d_50k/{original,A,B,C}_training_data_1d.pkl`. Submitted 4 production training jobs (SLURM 9973666-9973669) using `slurm/run_production_1d_50k.sh` with `--skip_data --data_path`. All use 50k sims, 256 hidden features, 16 workers.
- **Seed study 50k — jobs submitted**: Created `slurm/run_seed_single_50k.sh`. Submitted 20 jobs (4 models × 5 seeds, SLURM 9973670-9973689) to `results/seed_study_50k/{model}/seed_{seed}/`.
- **Over-egged wording — FIXED**: Applied all 6 text fixes from `docs/claims_review.md` items 6–11 to `template.tex`:
  - Abstract: "well-calibrated" → "validated through SBC diagnostics"
  - Intro: "removing the need" → "reducing reliance"
  - Section 2.4: "maximally informative" → "optimized"
  - Discussion: "substantial information loss" → "may discard information"
  - Section 3.2: Added sentence about shifted 2D medians for Model A
  - Diagnostics: Added Model A specific KS failure explanation
- **Table 2 fixed**: Hidden features 128→256 for 1D. Red note removed.
- **Table 5 and 5b red notes removed**.
- **Runtime caption**: Updated "8 CPU workers" → "16 CPU workers".

### Still needed (after jobs complete)
- **Submit SBC diagnostics** for new 1D 50k models (4 jobs)
- **Update `paper_figures.py` RESULT_PATHS** to point to new 1D 50k result directories
- **Regenerate all figures** via `python results/paper_figures.py`
- **Update template.tex** with new posterior values (Tables 3, 4, 5, 5b, 6; Sections 3.1–3.4; runtime table)
- **Reassess confound-dependent claims** (`docs/claims_review.md` items 1–5) based on new 50k 1D posteriors
- **Update handoff.md** with final result directory names

### Paper text edits (2026-02-23)
- Moved posterior predictive check (PPC) discussion and figures from Section 3.1 to new Appendix A. Main text retains brief references to appendix.
- Removed subsubsection headings from Sections 3.2, 3.3, 3.4 (Models A, B, C) for better prose flow.
- Updated Model B section text: removed U-fixed explanation, added 3-param description with U-R degeneracy discussion, updated all posterior values.
