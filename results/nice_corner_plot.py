#!/usr/bin/env python3
"""
Script to generate corner plots for NPE posterior results.
Incorporates logic from notebooks/demo.py with command-line interface.
"""

import argparse
import pickle
import sys
from pathlib import Path

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import corner
    import scienceplots
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Please install required packages: numpy, matplotlib, corner, scienceplots")
    sys.exit(1)


def load_results(results_path):
    """
    Load posterior results from pickle file.
    
    Args:
        results_path (str): Path to pickle file containing results
        
    Returns:
        tuple: (posterior_samples, true_parameters)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file doesn't contain expected data structure
    """
    results_file = Path(results_path)
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    if not results_file.is_file():
        raise ValueError(f"Path is not a file: {results_path}")
    
    try:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load pickle file: {e}")
    
    # Validate required keys
    required_keys = ['posterior_samples', 'true_parameters']
    missing_keys = [key for key in required_keys if key not in results]
    if missing_keys:
        raise ValueError(f"Missing required keys in results file: {missing_keys}")
    
    posterior_samples = results['posterior_samples']
    true_parameters = results['true_parameters']
    
    # Validate data shapes
    if not isinstance(posterior_samples, np.ndarray):
        raise ValueError("posterior_samples must be a numpy array")
    
    if len(posterior_samples.shape) != 2:
        raise ValueError(f"posterior_samples must be 2D array, got shape: {posterior_samples.shape}")
    
    if posterior_samples.shape[1] != 2:
        raise ValueError(f"Expected 2 parameters (U, P), got {posterior_samples.shape[1]}")
    
    if len(true_parameters) != 2:
        raise ValueError(f"Expected 2 true parameters, got {len(true_parameters)}")
    
    return posterior_samples, true_parameters


def create_corner_plot(posterior_samples, true_parameters, smooth=1.0, output_path=None):
    """
    Create corner plot using the same logic as notebooks/demo.py
    
    Args:
        posterior_samples (np.ndarray): Posterior samples array of shape (n_samples, 2)
        true_parameters (list/array): True parameter values [U, P]
        smooth (float): Smoothing parameter for corner plot
        output_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Set up scientific plotting style
    plt.style.use('science')
    
    print("Creating corner plot visualization...")
    
    # Parameter names and labels
    param_names = ['U', 'P']
    
    # Create corner plot with same configuration as notebook
    fig = corner.corner(
        posterior_samples,
        labels=param_names,
        truths=true_parameters,
        truth_color='red',
        color='skyblue',
        show_titles=True,
        range=((0, 1), (0, 1)),
        title_kwargs={'fontsize': 12},
        label_kwargs={'fontsize': 14},
        title_fmt='.4f',
        smooth=smooth,
        smooth1d=smooth,
        bins=50,
        quantiles=[0.16, 0.5, 0.84],  # Show 68% credible intervals
        plot_density=True,
        plot_datapoints=True,
        fill_contours=True,
        max_n_ticks=5
    )
    
    if output_path:
        try:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Corner plot saved to: {output_path}")
        except Exception as e:
            print(f"Warning: Failed to save plot to {output_path}: {e}")
    
    return fig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate corner plots for NPE posterior results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--smooth', 
        type=float, 
        required=True,
        help="Smoothing parameter applied to both smooth and smooth1d (must be positive)"
    )
    
    parser.add_argument(
        '--results-path', 
        type=str, 
        required=True,
        help="Path to pickle file containing posterior results"
    )
    
    parser.add_argument(
        '--output', 
        type=str,
        help="Output file path to save the plot (optional)"
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help="Don't display the plot interactively"
    )
    
    args = parser.parse_args()
    
    # Validate smoothing parameter
    if args.smooth <= 0:
        parser.error("Smoothing parameter must be positive")
    
    return args


def main():
    """Main function to run the corner plot generation."""
    try:
        args = parse_args()
        
        print(f"Loading results from: {args.results_path}")
        print(f"Using smoothing parameter: {args.smooth}")
        
        # Load data
        posterior_samples, true_parameters = load_results(args.results_path)
        
        print(f"✅ Successfully loaded results!")
        print(f"📊 Posterior samples shape: {posterior_samples.shape}")
        print(f"🎯 True parameters: U={true_parameters[0]}, P={true_parameters[1]}")
        
        # Determine output path if not provided
        output_path = args.output
        if output_path is None:
            # Create plots_updated directory relative to the results file
            results_path = Path(args.results_path)
            plots_dir = results_path.parent / "plots_updated"
            plots_dir.mkdir(exist_ok=True)
            output_path = plots_dir / f"corner_plot_smooth_{args.smooth}.png"
        
        # Create corner plot
        fig = create_corner_plot(
            posterior_samples=posterior_samples,
            true_parameters=true_parameters,
            smooth=args.smooth,
            output_path=str(output_path)
        )
        
        # Display plot if requested
        if not args.no_display:
            plt.show()
        
        print("✅ Corner plot generation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()