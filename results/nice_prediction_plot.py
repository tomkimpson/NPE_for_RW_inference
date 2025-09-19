#!/usr/bin/env python3
"""
Standalone Prediction Intervals Plot Generator

This script recreates the prediction intervals plot from the NPE workflow by:
1. Loading posterior samples from posterior_samples.pkl
2. Regenerating observed data using true parameters
3. Running posterior predictive sampling
4. Creating the same plot as the main workflow

Usage:
    python nice_prediction_plot.py [path_to_posterior_samples.pkl]
    python nice_prediction_plot.py --help
"""

import argparse
import pickle
import sys
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Project imports
from simulator import RandomWalkSimulator
from predict import posterior_predictive_sample, compute_prediction_intervals, plot_prediction_intervals


def find_latest_posterior_samples():
    """Find the most recent posterior_samples.pkl file in results directory."""
    results_dir = Path(__file__).parent

    # Find all workflow directories
    workflow_dirs = [d for d in results_dir.iterdir()
                    if d.is_dir() and d.name.startswith("workflow_")]

    if not workflow_dirs:
        raise FileNotFoundError("No workflow directories found in results/")

    # Sort by modification time (most recent first)
    workflow_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Look for posterior_samples.pkl in each directory
    for workflow_dir in workflow_dirs:
        posterior_file = workflow_dir / "inference_results" / "posterior_samples.pkl"
        if posterior_file.exists():
            return posterior_file

    raise FileNotFoundError("No posterior_samples.pkl found in any workflow directory")


def load_posterior_data(posterior_path):
    """
    Load posterior samples and metadata from pickle file.

    Parameters:
    -----------
    posterior_path : Path
        Path to posterior_samples.pkl file

    Returns:
    --------
    Dict containing posterior_samples, true_parameters, and metadata
    """
    print(f"📂 Loading posterior data from: {posterior_path}")

    if not posterior_path.exists():
        raise FileNotFoundError(f"Posterior samples file not found: {posterior_path}")

    with open(posterior_path, 'rb') as f:
        data = pickle.load(f)

    # Validate required keys
    required_keys = ['posterior_samples', 'true_parameters', 'metadata']
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValueError(f"Missing required keys in posterior data: {missing_keys}")

    # Extract key information
    posterior_samples = data['posterior_samples']
    true_parameters = data['true_parameters']
    metadata = data['metadata']

    print(f"✅ Loaded {posterior_samples.shape[0]} posterior samples")
    print(f"🎯 True parameters: U={true_parameters[0]}, P={true_parameters[1]}")
    print(f"📐 Lattice size: {metadata['lattice_size']}")
    print(f"⏰ Time steps: {metadata['time_steps']}")
    print(f"🌱 Seed: {metadata['seed']}")

    return data


def main():
    parser = argparse.ArgumentParser(
        description='Generate prediction intervals plot from posterior samples',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('posterior_path', type=str, nargs='?',
                       help='Path to posterior_samples.pkl file (default: find latest)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as posterior file)')
    parser.add_argument('--n_pred_samples', type=int, default=None,
                       help='Number of predictive samples (default: all posterior samples)')
    parser.add_argument('--seed_offset', type=int, default=2000,
                       help='Seed offset for predictive sampling (to match main workflow)')
    parser.add_argument('--obs_seed_offset', type=int, default=1000,
                       help='Seed offset for observed data generation (to match main workflow)')

    args = parser.parse_args()

    start_time = time.time()

    # Determine posterior file path
    if args.posterior_path:
        posterior_path = Path(args.posterior_path)
    else:
        print("🔍 No path specified, searching for latest posterior_samples.pkl...")
        posterior_path = find_latest_posterior_samples()
        print(f"📍 Found: {posterior_path}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = posterior_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Starting Standalone Prediction Plot Generation")
    print(f"📁 Input file: {posterior_path}")
    print(f"📁 Output directory: {output_dir}")
    print()

    # Load posterior data
    data = load_posterior_data(posterior_path)
    posterior_samples = data['posterior_samples']
    true_parameters = data['true_parameters']
    metadata = data['metadata']

    # Extract simulation parameters from metadata
    Lx, Ly = metadata['lattice_size']
    T = metadata['time_steps']
    seed = metadata['seed']

    print(f"\n📐 Setting up simulator (Lx={Lx}, Ly={Ly}, T={T})")

    # Initialize simulator
    simulator = RandomWalkSimulator(
        Lx=Lx,
        Ly=Ly,
        initial_region_half_width=25  # Default.
    )

    # Generate observed data using true parameters (matching main.py logic)
    print(f"\n🎯 Generating observed data with true parameters U={true_parameters[0]}, P={true_parameters[1]}")
    print(f"   Using seed: {seed + args.obs_seed_offset} (original seed + {args.obs_seed_offset})")

    observation, initial_positions, final_positions = simulator.simulate(
        U=true_parameters[0],
        P=true_parameters[1],
        T=T,
        random_seed=seed + args.obs_seed_offset,
        use_2d_output=False  # Get 1D column counts for plotting
    )

    print(f"   Observed data: {len(observation)} columns, {observation.sum()} total agents")

    # Generate posterior predictive samples
    print(f"\n🔮 Generating posterior predictive samples...")
    print(f"   Using seed: {seed + args.seed_offset} (original seed + {args.seed_offset})")

    n_pred = args.n_pred_samples or len(posterior_samples)
    predictions = posterior_predictive_sample(
        posterior_samples=posterior_samples,
        simulator=simulator,
        T=T,
        n_pred_samples=n_pred,
        random_seed=seed + args.seed_offset,
        use_2d_output=False  # Get 1D column counts
    )

    # Compute prediction intervals
    print(f"\n📊 Computing prediction intervals...")
    prediction_results = compute_prediction_intervals(predictions)

    # Create visualization
    print(f"\n📈 Creating prediction intervals plot...")
    fig = plot_prediction_intervals(
        prediction_results=prediction_results,
        observed_data=observation,
        Lx=Lx,
        title_suffix=f" (T={T}, Standalone)"
    )

    # Save plot
    plot_path = output_dir / "prediction_intervals_standalone.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save summary data
    summary_data = {
        'prediction_results': prediction_results,
        'observed_data': observation,
        'input_metadata': {
            'posterior_file': str(posterior_path),
            'posterior_samples_shape': posterior_samples.shape,
            'true_parameters': true_parameters,
            'simulation_params': {
                'Lx': Lx, 'Ly': Ly, 'T': T,
                'original_seed': seed
            },
            'generation_params': {
                'n_pred_samples': n_pred,
                'obs_seed_offset': args.obs_seed_offset,
                'pred_seed_offset': args.seed_offset
            }
        }
    }

    summary_path = output_dir / "prediction_intervals_standalone.pkl"
    with open(summary_path, 'wb') as f:
        pickle.dump(summary_data, f)

    # Print summary
    elapsed = time.time() - start_time
    global_stats = prediction_results['global_stats']

    print(f"\n🎉 PREDICTION PLOT GENERATION COMPLETED!")
    print(f"⏱️  Total time: {elapsed:.1f} seconds")
    print(f"📁 Output saved to: {output_dir}")
    print(f"\n📊 Files created:")
    print(f"   📈 Plot: prediction_intervals_standalone.png")
    print(f"   💾 Data: prediction_intervals_standalone.pkl")

    print(f"\n📈 Prediction Summary:")
    print(f"   Total agents: {global_stats['total_agents_mean']:.1f} ± {global_stats['total_agents_std']:.1f}")
    print(f"   Range: [{global_stats['total_agents_min']}, {global_stats['total_agents_max']}]")

    observed_total = np.sum(observation)
    predictions_total = np.sum(predictions, axis=1)
    p2_5 = np.percentile(predictions_total, 2.5)
    p97_5 = np.percentile(predictions_total, 97.5)
    in_interval = p2_5 <= observed_total <= p97_5
    print(f"   Observed total: {observed_total}")
    print(f"   95% interval: [{p2_5:.1f}, {p97_5:.1f}]")
    print(f"   Observed in interval: {'Yes' if in_interval else 'No'}")


if __name__ == '__main__':
    main()