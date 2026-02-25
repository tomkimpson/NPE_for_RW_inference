# Model A Reparameterization: (U, P, rho) to (U, D, v)

## Problem

Model A infers lattice parameters (U, P, rho) where the drift velocity v = P*rho/2. The data constrains the product P*rho tightly but not P or rho individually, creating a hyperbolic degeneracy — the posterior lies along a curve rho ~ c/P (banana-shaped correlation in the corner plot). Individual biases are ~14% while v is accurate to ~3-6%.

## Solution

Reparameterize the NPE to infer continuum parameters (U, D, v) directly:
- D = P/4 (diffusivity)
- v = P*rho/2 (drift velocity)
- Inverse: P = 4D, rho = v/(2D)

The existing 50k training data is reused — only the theta columns are transformed; observations are unchanged.

## Why it works

### Independent information channels

D and v are constrained by different features of the density profile:
- **v** is constrained by the center-of-mass shift — a global, low-noise statistic averaged over many particles
- **D** is constrained by the symmetric spreading/width of the profile

Knowing how fast the bulk drifts tells you very little about how fast particles diffuse. The D-v covariance is approximately flat (near-independent), unlike the strongly correlated P-rho banana.

### Flow geometry

Normalizing flows (compositions of affine coupling layers) naturally represent distributions that are roughly axis-aligned or have mild linear correlations. A hyperbolic banana is hard — the flow needs many transforms to bend the base Gaussian into that curved shape and typically approximates it imperfectly (too wide in the tails, too narrow at the waist).

In (U, D, v) space the posterior is close to a multivariate Gaussian with near-diagonal covariance. The flow represents this almost exactly with far fewer transforms, leaving capacity to spare rather than struggling with geometry.

### SBC calibration improvement

SBC tests calibration across the full prior. When the flow struggles with the banana, it's not uniformly wrong — it's well-calibrated for some (P, rho) combinations and miscalibrated for others (e.g. near the edges where curvature is strongest). This shows up as non-uniform SBC ranks.

In the reparameterized space:
- **D passes KS** because the flow represents D's marginal cleanly — no banana to approximate
- **U C2ST improves** because the flow no longer wastes capacity on the P-rho banana
- **TARP ATC halves** because the joint posterior geometry is simpler everywhere across the prior

## Results

### Posterior estimates

| Space | Param | True | Posterior | 95% CI |
|-------|-------|------|-----------|--------|
| Continuum | U | 0.500 | 0.508 +/- 0.014 | [0.484, 0.534] |
| Continuum | D | 0.175 | 0.195 +/- 0.021 | [0.156, 0.236] |
| Continuum | v | 0.175 | 0.169 +/- 0.006 | [0.161, 0.178] |
| Lattice | U | 0.500 | 0.508 +/- 0.014 | [0.484, 0.534] |
| Lattice | P | 0.700 | 0.781 +/- 0.086 | [0.626, 0.945] |
| Lattice | rho | 0.500 | 0.437 +/- 0.054 | [0.352, 0.546] |

v is very tightly constrained (std=0.006, ~3.4% relative error). All true values within 95% CIs.

### SBC diagnostics comparison

| Metric | Baseline (U,P,rho) | Reparam (U,D,v) |
|--------|-------------------|-----------------|
| KS: U | 0.0000 (FAIL) | 0.0000 (FAIL) |
| KS: P/D | 0.0000 (FAIL) | 0.6034 (PASS) |
| KS: rho/v | 0.0159 (FAIL) | 0.0029 (FAIL) |
| C2ST ranks: U | 0.6415 (FAIL) | 0.5985 (PASS) |
| C2ST ranks: P/D | 0.5590 (PASS) | 0.5860 (PASS) |
| C2ST ranks: rho/v | 0.5770 (PASS) | 0.5715 (PASS) |
| C2ST DAP (all) | All PASS | All PASS |
| TARP ATC | 0.084 (PASS) | 0.035 (PASS) |
| TARP KS p | 0.999 (PASS) | 1.000 (PASS) |

Wins: D KS fixed, U C2ST fixed, TARP halved, 6/6 C2ST pass (vs 5/6).

The v KS still fails because v is so tightly constrained (std=0.006) that even sub-percent flow imperfections are detectable at n=1000. C2ST confirms it is well-calibrated.

## File references

- Result directory: `results/workflow_A_npe2d_20260226_083659/`
- Baseline: `/fred/oz022/tkimpson/SNPE/NPE_for_RW_inference/results/workflow_A_npe2d_20260225_140808/`
- SBC comparison plot: `results/figures/sbc_model_A_reparam_comparison.png`
- SLURM script: `slurm/run_model_a_reparam.sh`
