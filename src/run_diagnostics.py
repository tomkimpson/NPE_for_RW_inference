"""
CLI entry point for running SBI diagnostics (SBC + TARP) on a trained NPE model.

Usage
-----
python src/run_diagnostics.py \
    --model_path results/workflow_A_npe_20260204_230502/npe_model.pkl \
    --model A \
    --n_sbc_sims 1000 \
    --n_posterior_samples 1000 \
    --n_workers 8
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure src/ is on the path when invoked from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from models import get_model_config
from inference import RandomWalkNPE
from diagnostics import generate_sbc_data, run_all_diagnostics
from utils import configure_warnings, check_device_availability, print_device_info

configure_warnings()


def main():
    parser = argparse.ArgumentParser(
        description="Run SBI diagnostics (SBC + TARP) on a trained NPE model."
    )
    parser.add_argument("--model_path", required=True, help="Path to trained npe_model.pkl")
    parser.add_argument("--model", required=True, choices=["original", "A", "B", "C"],
                        help="Model name")
    parser.add_argument("--n_sbc_sims", type=int, default=1000,
                        help="Number of SBC simulations (default: 1000)")
    parser.add_argument("--n_posterior_samples", type=int, default=1000,
                        help="Posterior samples per SBC sim (default: 1000)")
    parser.add_argument("--n_workers", type=int, default=8,
                        help="Parallel workers for simulation (default: 8)")
    parser.add_argument("--Lx", type=int, default=200)
    parser.add_argument("--Ly", type=int, default=50)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--initial_region_half_width", type=int, default=25)
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: <model_dir>/diagnostics/)")
    args = parser.parse_args()

    # Resolve output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.model_path).parent / "diagnostics")

    print("=" * 50)
    print("SBI Diagnostics: SBC + TARP")
    print("=" * 50)
    print(f"Model:            {args.model}")
    print(f"Model path:       {args.model_path}")
    print(f"SBC sims:         {args.n_sbc_sims}")
    print(f"Posterior samples: {args.n_posterior_samples}")
    print(f"Workers:          {args.n_workers}")
    print(f"Grid:             Lx={args.Lx}, Ly={args.Ly}, T={args.T}")
    print(f"Output dir:       {args.output_dir}")
    print("=" * 50)

    # Device
    device, device_info = check_device_availability()
    print_device_info(device, device_info)

    # Load model
    print("\nLoading trained model...")
    npe = RandomWalkNPE.load_model(args.model_path, device=device)
    posterior = npe.posterior
    model_config = get_model_config(args.model)
    print(f"  Parameters: {model_config.param_names}")

    # Generate SBC data
    print("\nGenerating SBC calibration data...")
    thetas, xs = generate_sbc_data(
        model_config=model_config,
        Lx=args.Lx,
        Ly=args.Ly,
        T=args.T,
        initial_region_half_width=args.initial_region_half_width,
        n_sims=args.n_sbc_sims,
        n_workers=args.n_workers,
    )
    print(f"  thetas: {thetas.shape}, xs: {xs.shape}")

    # Run diagnostics
    print("\nRunning diagnostics...")
    run_all_diagnostics(
        posterior=posterior,
        thetas=thetas,
        xs=xs,
        param_names=model_config.param_names,
        n_posterior_samples=args.n_posterior_samples,
        output_dir=args.output_dir,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
