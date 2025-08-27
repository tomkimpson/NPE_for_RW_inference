#!/usr/bin/env python3
"""
Posterior Predictive Sampling for NPE Random Walk Model.

This script performs posterior predictive sampling by taking posterior samples
from a trained NPE model and pushing them through the simulator to generate
probabilistic predictions with uncertainty quantification.
"""

import argparse
import pickle
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional, List

# Project imports
from simulator import RandomWalkSimulator
from inference import RandomWalkNPE
from utils import check_device_availability, print_device_info, configure_warnings

# Configure warning filters
configure_warnings()


def load_results(results_path: str) -> Dict[str, Any]:
    """
    Load results from NPE inference workflow.
    
    Parameters:
    -----------
    results_path : str
        Path to results.pkl or results_extracted.pkl file
        
    Returns:
    --------
    Dict containing posterior_samples, true_parameters, and metadata
    """
    results_path = Path(results_path)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    print(f"📂 Loading results from: {results_path}")
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    # Validate required keys
    required_keys = ['posterior_samples', 'true_parameters']
    missing_keys = [key for key in required_keys if key not in results]
    if missing_keys:
        raise ValueError(f"Missing required keys in results: {missing_keys}")
    
    # Ensure posterior samples are numpy arrays
    if isinstance(results['posterior_samples'], torch.Tensor):
        results['posterior_samples'] = results['posterior_samples'].cpu().numpy()
    
    print(f"✅ Loaded {results['posterior_samples'].shape[0]} posterior samples")
    print(f"🎯 True parameters: U={results['true_parameters'][0]}, P={results['true_parameters'][1]}")
    
    return results


def posterior_predictive_sample(
    posterior_samples: np.ndarray,
    simulator: RandomWalkSimulator, 
    T: int,
    n_pred_samples: Optional[int] = None,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate posterior predictive samples by running simulator with posterior parameter samples.
    
    Parameters:
    -----------
    posterior_samples : np.ndarray of shape (n_samples, 2)
        Posterior samples for parameters [U, P]
    simulator : RandomWalkSimulator
        Configured simulator instance
    T : int
        Number of time steps for simulation
    n_pred_samples : int, optional
        Number of predictive samples to generate (default: all posterior samples)
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray of shape (n_pred_samples, n_columns)
        Predicted column counts for each parameter sample
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_posterior = posterior_samples.shape[0]
    n_pred = n_pred_samples or n_posterior
    
    # Subsample posterior if requested
    if n_pred < n_posterior:
        indices = np.random.choice(n_posterior, size=n_pred, replace=False)
        selected_samples = posterior_samples[indices]
    else:
        selected_samples = posterior_samples
        n_pred = n_posterior
    
    print(f"🔮 Generating {n_pred} posterior predictive samples...")
    
    # Initialize results array
    n_columns = simulator.Lx
    predictions = np.zeros((n_pred, n_columns), dtype=int)
    
    # Generate predictions
    start_time = time.time()
    for i, (U, P) in enumerate(selected_samples):
        if i % max(1, n_pred // 10) == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_pred - i - 1) / rate if rate > 0 else 0
            print(f"   Progress: {i+1}/{n_pred} ({100*(i+1)/n_pred:.1f}%) - "
                  f"{rate:.1f} samples/sec - ETA: {eta:.1f}s")
        
        # Run simulation with current parameter sample
        column_counts, _, _ = simulator.simulate(
            U=float(U), 
            P=float(P), 
            T=T,
            random_seed=random_seed + i if random_seed is not None else None
        )
        
        predictions[i] = column_counts
    
    elapsed = time.time() - start_time
    print(f"✅ Predictive sampling completed in {elapsed:.1f} seconds")
    print(f"   Average rate: {n_pred/elapsed:.1f} samples/sec")
    
    return predictions


def compute_prediction_intervals(
    predictions: np.ndarray,
    percentiles: List[float] = [2.5, 25, 50, 75, 97.5]
) -> Dict[str, Any]:
    """
    Compute prediction intervals and summary statistics.
    
    Parameters:
    -----------
    predictions : np.ndarray of shape (n_samples, n_columns)
        Predicted column counts
    percentiles : List[float]
        Percentiles to compute for intervals
        
    Returns:
    --------
    Dict containing intervals, summary statistics, and metadata
    """
    print("📊 Computing prediction intervals...")
    
    n_pred, n_columns = predictions.shape
    
    # Compute percentiles for each column
    intervals = {}
    for p in percentiles:
        intervals[f"p{p}"] = np.percentile(predictions, p, axis=0)
    
    # Summary statistics
    stats = {
        'mean': np.mean(predictions, axis=0),
        'std': np.std(predictions, axis=0),
        'min': np.min(predictions, axis=0),
        'max': np.max(predictions, axis=0)
    }
    
    # Global summary statistics
    global_stats = {
        'total_agents_mean': np.mean(np.sum(predictions, axis=1)),
        'total_agents_std': np.std(np.sum(predictions, axis=1)),
        'total_agents_min': np.min(np.sum(predictions, axis=1)),
        'total_agents_max': np.max(np.sum(predictions, axis=1))
    }
    
    results = {
        'intervals': intervals,
        'column_stats': stats,
        'global_stats': global_stats,
        'predictions': predictions,
        'metadata': {
            'n_predictions': n_pred,
            'n_columns': n_columns,
            'percentiles': percentiles
        }
    }
    
    print(f"✅ Computed intervals for {n_columns} columns")
    print(f"   Total agents - Mean: {global_stats['total_agents_mean']:.1f} ± {global_stats['total_agents_std']:.1f}")
    print(f"   Total agents - Range: [{global_stats['total_agents_min']}, {global_stats['total_agents_max']}]")
    
    return results


def plot_prediction_intervals(
    prediction_results: Dict[str, Any],
    observed_data: Optional[np.ndarray] = None,
    Lx: int = 21,
    title_suffix: str = ""
) -> plt.Figure:
    """
    Create visualization of prediction intervals with uncertainty bands.
    
    Parameters:
    -----------
    prediction_results : Dict
        Results from compute_prediction_intervals
    observed_data : np.ndarray, optional
        Observed column counts for comparison
    Lx : int
        Number of columns (for x-axis)
    title_suffix : str
        Additional text for plot title
        
    Returns:
    --------
    matplotlib Figure
    """
    intervals = prediction_results['intervals']
    stats = prediction_results['column_stats']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    columns = np.arange(Lx)
    
    # Main prediction plot with uncertainty bands
    ax1.fill_between(columns, intervals['p2.5'], intervals['p97.5'], 
                     alpha=0.2, color='blue', label='95% Prediction Interval')
    ax1.fill_between(columns, intervals['p25'], intervals['p75'], 
                     alpha=0.3, color='blue', label='50% Prediction Interval')
    ax1.plot(columns, intervals['p50'], 'b-', linewidth=2, label='Median Prediction')
    ax1.plot(columns, stats['mean'], 'b--', linewidth=1, alpha=0.8, label='Mean Prediction')
    
    if observed_data is not None:
        ax1.plot(columns, observed_data, 'ro-', linewidth=2, markersize=6, 
                label='Observed Data')
    
    ax1.set_xlabel('Column Index')
    ax1.set_ylabel('Agent Count')
    ax1.set_title(f'Posterior Predictive Distribution{title_suffix}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribution of total agents
    total_agents = np.sum(prediction_results['predictions'], axis=1)
    ax2.hist(total_agents, bins=50, density=True, alpha=0.7, color='skyblue', 
             edgecolor='navy')
    
    # Add statistics lines
    mean_total = prediction_results['global_stats']['total_agents_mean']
    median_total = np.median(total_agents)
    ax2.axvline(mean_total, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_total:.1f}')
    ax2.axvline(median_total, color='orange', linestyle='--', linewidth=2, 
                label=f'Median: {median_total:.1f}')
    
    if observed_data is not None:
        observed_total = np.sum(observed_data)
        ax2.axvline(observed_total, color='green', linestyle='-', linewidth=3, 
                    label=f'Observed: {observed_total}')
    
    ax2.set_xlabel('Total Agent Count')
    ax2.set_ylabel('Density')
    ax2.set_title('Predictive Distribution of Total Agent Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Posterior predictive sampling for NPE Random Walk model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument('results_path', type=str,
                       help='Path to results.pkl or results_extracted.pkl file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as results file)')
    
    # Simulation parameters
    parser.add_argument('--Lx', type=int, default=21,
                       help='Lattice width (must match training data)')
    parser.add_argument('--Ly', type=int, default=21,
                       help='Lattice height (must match training data)')
    parser.add_argument('--T', type=int, default=100,
                       help='Number of time steps (must match training data)')
    parser.add_argument('--initial_region_half_width', type=int, default=None,
                       help='Half-width of initial region (default: Lx//4)')
    
    # Prediction parameters
    parser.add_argument('--n_pred_samples', type=int, default=None,
                       help='Number of predictive samples (default: all posterior samples)')
    parser.add_argument('--percentiles', type=float, nargs='+', 
                       default=[2.5, 25, 50, 75, 97.5],
                       help='Percentiles for prediction intervals')
    
    # General parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device (predictions run on CPU regardless)')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    results_path = Path(args.results_path)
    if args.output_dir is None:
        output_dir = results_path.parent / "predictions"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔮 Starting Posterior Predictive Sampling")
    print(f"📁 Results file: {results_path}")
    print(f"📁 Output directory: {output_dir}")
    print(f"⚙️  Simulation settings: Lx={args.Lx}, Ly={args.Ly}, T={args.T}")
    print()
    
    total_start = time.time()
    
    # Load results
    results = load_results(args.results_path)
    posterior_samples = results['posterior_samples']
    true_parameters = results['true_parameters']
    
    # Get observed data if available
    observed_data = results.get('observed_data', None)
    if observed_data is not None:
        print(f"📊 Observed data available: {len(observed_data)} columns, {np.sum(observed_data)} total agents")
    
    # Initialize simulator
    print(f"\n📐 Setting up simulator...")
    simulator = RandomWalkSimulator(
        Lx=args.Lx,
        Ly=args.Ly,
        initial_region_half_width=args.initial_region_half_width
    )
    
    # Generate predictive samples
    print(f"\n🔮 Generating posterior predictive samples...")
    predictions = posterior_predictive_sample(
        posterior_samples=posterior_samples,
        simulator=simulator,
        T=args.T,
        n_pred_samples=args.n_pred_samples,
        random_seed=args.seed
    )
    
    # Compute prediction intervals
    prediction_results = compute_prediction_intervals(
        predictions=predictions,
        percentiles=args.percentiles
    )
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    
    # Main prediction plot
    fig = plot_prediction_intervals(
        prediction_results=prediction_results,
        observed_data=observed_data,
        Lx=args.Lx,
        title_suffix=f" (T={args.T})"
    )
    fig.savefig(output_dir / "prediction_intervals.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Save results
    print(f"\n💾 Saving results...")
    
    # Save prediction results
    output_data = {
        'prediction_results': prediction_results,
        'input_metadata': {
            'results_path': str(args.results_path),
            'posterior_samples_shape': posterior_samples.shape,
            'true_parameters': true_parameters,
            'simulation_params': {
                'Lx': args.Lx, 'Ly': args.Ly, 'T': args.T,
                'initial_region_half_width': args.initial_region_half_width
            },
            'prediction_params': {
                'n_pred_samples': args.n_pred_samples or len(posterior_samples),
                'percentiles': args.percentiles,
                'seed': args.seed
            }
        }
    }
    
    if observed_data is not None:
        output_data['observed_data'] = observed_data
    
    with open(output_dir / "predictive_results.pkl", 'wb') as f:
        pickle.dump(output_data, f)
    
    # Save summary
    summary_path = output_dir / "prediction_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Posterior Predictive Sampling Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Results file: {args.results_path}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        f.write("Simulation Parameters:\n")
        f.write(f"  Lattice size: {args.Lx} x {args.Ly}\n")
        f.write(f"  Time steps: {args.T}\n")
        f.write(f"  Initial region half-width: {args.initial_region_half_width or args.Lx//4}\n\n")
        
        f.write("Prediction Parameters:\n")
        f.write(f"  Posterior samples available: {len(posterior_samples)}\n")
        f.write(f"  Predictive samples generated: {prediction_results['metadata']['n_predictions']}\n")
        f.write(f"  Percentiles computed: {args.percentiles}\n")
        f.write(f"  Random seed: {args.seed}\n\n")
        
        f.write("Results:\n")
        global_stats = prediction_results['global_stats']
        f.write(f"  Total agents - Mean: {global_stats['total_agents_mean']:.2f}\n")
        f.write(f"  Total agents - Std: {global_stats['total_agents_std']:.2f}\n")
        f.write(f"  Total agents - Range: [{global_stats['total_agents_min']}, {global_stats['total_agents_max']}]\n")
        
        if observed_data is not None:
            observed_total = np.sum(observed_data)
            f.write(f"  Observed total agents: {observed_total}\n")
            
            # Check if observed falls within prediction interval
            predictions_total = np.sum(predictions, axis=1)
            p2_5 = np.percentile(predictions_total, 2.5)
            p97_5 = np.percentile(predictions_total, 97.5)
            in_interval = p2_5 <= observed_total <= p97_5
            f.write(f"  Observed in 95% interval: {in_interval} (interval: [{p2_5:.1f}, {p97_5:.1f}])\n")
    
    # Workflow completed
    total_elapsed = time.time() - total_start
    
    print(f"\n🎉 POSTERIOR PREDICTIVE SAMPLING COMPLETED!")
    print(f"⏱️  Total time: {total_elapsed:.1f} seconds")
    print(f"📁 Results saved in: {output_dir}")
    print(f"\n📊 Files created:")
    print(f"   📈 Prediction intervals plot: prediction_intervals.png")
    print(f"   💾 Full results: predictive_results.pkl")
    print(f"   📝 Summary: prediction_summary.txt")
    
    print(f"\n📈 Prediction Summary:")
    global_stats = prediction_results['global_stats']
    print(f"   Total agents: {global_stats['total_agents_mean']:.1f} ± {global_stats['total_agents_std']:.1f}")
    print(f"   Range: [{global_stats['total_agents_min']}, {global_stats['total_agents_max']}]")
    
    if observed_data is not None:
        observed_total = np.sum(observed_data)
        predictions_total = np.sum(predictions, axis=1)
        p2_5 = np.percentile(predictions_total, 2.5)
        p97_5 = np.percentile(predictions_total, 97.5)
        in_interval = p2_5 <= observed_total <= p97_5
        print(f"   Observed total: {observed_total}")
        print(f"   95% interval: [{p2_5:.1f}, {p97_5:.1f}]")
        print(f"   Observed in interval: {'Yes' if in_interval else 'No'}")
    
    print(f"\n🔍 Next steps:")
    print(f"   - Examine prediction_intervals.png for spatial uncertainty patterns")
    print(f"   - Check if observed data falls within prediction intervals")
    print(f"   - Compare predictions with different T values or parameter ranges")
    print(f"   - Use predictions for model validation or experimental design")


if __name__ == '__main__':
    main()