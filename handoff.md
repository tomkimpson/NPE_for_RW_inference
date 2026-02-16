# Handoff: `item2-new-models` branch

## Summary

This branch implements the core code infrastructure for the paper revision: 4 random walk models (original + A/B/C), parallelized data generation, CNN embedding for 2D spatial data, and an ABC baseline (retained in code but not for the paper). All production NPE runs (1D and 2D) have completed successfully for all 4 models.

---

## What was done

### 1. Multi-model infrastructure (earlier commits on this branch)

- **`src/simulator.py`**: Added `ExclusionRandomWalkSimulator` supporting crowding (exclusion process), directional bias (rho), and proliferation (R) on a 2D lattice with zero-flux boundary conditions.
- **`src/models.py`**: Created `ModelConfig` dataclass and `MODEL_CONFIGS` registry for all 4 models:
  - **original**: infer U, P (no exclusion)
  - **Model A**: infer U, P, rho (exclusion + bias, no growth)
  - **Model B**: infer P, R (exclusion + growth, U fixed at 0.5)
  - **Model C**: infer P, rho, R (exclusion + bias + growth, U fixed at 0.5)
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
| original | `workflow_original_npe_20260204_225605` | `workflow_original_npe2d_20260204_230541` | `workflow_original_abc_20260204_225446` |
| A | `workflow_A_npe_20260204_230502` | `workflow_A_npe2d_20260204_230837` | `workflow_A_abc_20260204_225446` |
| B | `workflow_B_npe_20260204_230914` | `workflow_B_npe2d_20260204_234828` | `workflow_B_abc_20260204_225446` |
| C | `workflow_C_npe_20260204_225447` | `workflow_C_npe2d_20260205_001134` | `workflow_C_abc_20260204_225446` |

**Status**:
- All 8 NPE jobs (4 × 1D, 4 × 2D): **completed successfully**. All true values fall within 95% credible intervals.
- ABC original: **completed** (original model uses the faster non-exclusion simulator).
- ABC A/B/C: **timed out** at 8 hours. Exclusion simulator too slow (~1-2 sims/s) for SMCABC to converge.

Each successful NPE result directory contains: `config.txt`, `npe_model.pkl`, `training_data.pkl`, and under `inference_results/`: posterior samples, marginal/pairwise plots, prediction intervals, and predictive results.

**Observation**: NPE 1D produces tighter posteriors than NPE+CNN 2D with 10k training samples. This is expected — the CNN adds model capacity that needs more data. The key comparison for the paper is whether 2D gives tighter posteriors for the bias parameter rho in Model A, where spatial anisotropy should be informative.

---

## Production run configuration

From `slurm/run_production.sh`:
- 10,000 training samples, 8 parallel workers
- NSF architecture: 128 hidden features, 8 transforms
- Training: max 100 epochs, early stopping after 20, lr=1e-4, batch size 512
- Lattice: Lx=200, Ly=50, T=100, initial_region_half_width=25
- Posterior: 5000 samples; predictive: 500 forward simulations
- GPU: A100 on milan-gpu partition

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

## Next steps (from response_plan.md)

### Immediate priorities

1. **Item 3 — SBI diagnostics** (response_plan.md): Implement simulation-based calibration (SBC) and expected coverage tests for all 4 models. Reviewer 2 specifically asked for this.

2. **Item 6 — 2D CNN on bias model analysis**: Compare 1D vs 2D posterior precision for Model A, specifically for the bias parameter rho. This is the key result that justifies the CNN approach — bias creates anisotropic spatial patterns that 2D data should capture.

3. **Item 9 — Reproducibility across seeds**: Run each model with 5 random seeds, report mean ± std of posterior means and CI widths.

4. **Classical baseline comparison**: Implement or reference profile likelihood / MLE / MCMC results from Simpson & Plank 2025 for Models A and B. Model C (bias + growth) has no classical benchmark — this is where NPE demonstrates necessity.

### Paper writing

5. **Item 1 — IMRaD restructure**: Already partially done (intro rewritten, IMRaD headings adopted). Still need: model-by-model results with biological interpretation, biological insights throughout.

6. **Items 4, 7, 8, 10-24**: Various text fixes, table additions, figure corrections per reviewer comments. See `docs/paper/reviewer_comments/response_plan.md` for the full checklist.

### Possible enhancements

7. **Increase 2D training samples**: If 2D posteriors are still wider than 1D, try 50k or 100k samples to let the CNN learn effectively.

8. **Multi-round NPE**: Currently using single-round NPE. Sequential (multi-round) NPE could improve sample efficiency for the more complex models.
