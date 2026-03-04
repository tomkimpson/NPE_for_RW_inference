# Project Instructions

## Handoff

- Always consult `handoff.md` at the start of a session for current project state, pending work, and result paths.
- When completing work or discovering important context, update `handoff.md` to keep it current. Remove stale information.

## Git Commits

- Never include "Co-Authored-By" lines in commit messages.

## Training Data

- Before generating new training data, check if existing data can be reused.
- Training data only needs regeneration when simulation parameters change (Lx, Ly, T, initial_region_half_width, model type).
- Changes to the neural network architecture (CNN options, hidden features, etc.) do NOT require new training data.
- Existing 2D training data is typically in `results/workflow_*_npe2d_*/training_data.pkl`.
- Use `--skip_data --data_path <path>` to reuse existing data.

## SLURM Jobs

- Always use GPU partition (`milan-gpu` with `gpu:a100:1`) for neural network training.
- GPU training: ~15-20 minutes for 50k samples.
- CPU training: ~3-4 hours for 50k samples.
- Use `conda activate NPE_LV` in SLURM scripts.
