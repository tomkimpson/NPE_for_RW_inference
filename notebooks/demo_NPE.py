import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md(
        r"""
        # Neural Posterior Estimation (NPE) for Random Walk Parameter Inference

        This notebook demonstrates **Neural Posterior Estimation (NPE)** for inferring parameters of a 2D random walk model, 
        as described in [Simpson & Planck](https://www.biorxiv.org/content/10.1101/2025.05.25.656057v4). 

        ## What is NPE?

        **Neural Posterior Estimation** is a simulation-based inference method that uses neural networks to learn the posterior distribution p(θ|x) directly from simulated data. Instead of requiring likelihood calculations, NPE:

        1. **Generates training data**: Samples parameters θ from priors, simulates corresponding observations x
        2. **Trains a neural network**: Learns to map observations x to posterior distributions over parameters θ  
        3. **Performs inference**: Given real observed data x_obs, the network outputs p(θ|x_obs)

        ## Sequential NPE (SNPE)

        **Sequential NPE** improves upon standard NPE by iteratively refining the posterior estimate:
        - **Round 1**: Sample from priors, train initial posterior estimate
        - **Round 2+**: Sample from previous posterior estimate, retrain with focused data
        - **Convergence**: Stop when posterior estimates stabilize

        This focuses computational effort on the most relevant parameter regions, leading to more accurate inference.

        ## Our Results

        We'll examine results from a successful **SNPE run** with:
        - **Target parameters**: U=0.3 (occupancy), P=0.7 (movement probability) 
        - **Sequential rounds**: 10 rounds with convergence monitoring
        - **Lattice size**: 100×50 with 100 time steps
        - **Training**: 2000 simulations per round, advanced neural architecture
        """
    )
    return (mo,)


@app.cell
def _():
    """
    Import required modules for NPE analysis and visualization
    """
    import sys
    import os
    import pickle
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import corner
    from pathlib import Path

    # Add src directory to path for our modules
    sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))

    # Import our NPE and simulator modules
    from inference import RandomWalkNPE
    from simulator import RandomWalkSimulator, plot_column_counts

    print("✅ Successfully imported NPE analysis modules!")
    print(f"🔧 NumPy version: {np.__version__}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"📊 Corner version: Available for posterior visualization")

    return RandomWalkNPE, corner, pickle, plt, torch


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Loading Inference Results

    We'll load the results from our successful SNPE workflow run `workflow_20250826_130359`. 
    This run used Sequential NPE to infer the parameters U and P from simulated random walk data.

    The results contain:
    - **Posterior samples**: 5000 samples from the final posterior distribution
    - **True parameters**: The ground truth values used to generate the test data
    - **Observed data**: The column counts that served as our observation
    - **Summary statistics**: Means, standard deviations, and credible intervals
    - **SNPE metadata**: Information about the sequential training process
    """
    )
    return


@app.cell
def _(pickle):
    """
    Load the inference results from the workflow run.
    We only have the compressed results here. See full .pkl on cluster.
    """
    results_path = "/Users/tkimpson/projects/NPE_for_RW_Inference/results/workflow_20250826_130359/inference_results/results_extracted.pkl"

    print("📂 Loading inference results...")

    with open(results_path, 'rb') as f1:
        results = pickle.load(f1)

    # Extract key components
    posterior_samples = results['posterior_samples']  # Shape: (5000, 2) for [U, P]
    true_parameters = results['true_parameters']      # [0.3, 0.7]


    print(f"✅ Successfully loaded results!")
    print(f"📊 Posterior samples shape: {posterior_samples.shape}")
    print(f"🎯 True parameters: U={true_parameters[0]}, P={true_parameters[1]}")


    return posterior_samples, true_parameters


@app.cell
def _(mo):
    mo.md(r"""## Corner Plot""")
    return


@app.cell
def _(corner, plt, posterior_samples, true_parameters):
    """
    Create a corner plot of the posterior distribution
    """
    import scienceplots
    plt.style.use('science')

    # Set up the corner plot
    print("Creating corner plot visualization...")

    # Parameter names and labels
    param_names = ['U', 'P']

    # Create corner plot
    fig = corner.corner(
        posterior_samples,
        labels=param_names,
        truths=true_parameters,
        truth_color='red',
        color='skyblue',
        show_titles=True,
        range=((0,1),(0,1)),
        title_kwargs={'fontsize': 12},
        label_kwargs={'fontsize': 14},
        title_fmt='.4f',
        smooth=1.0,
        smooth1d=1.0,
        bins=50,
        quantiles=[0.16, 0.5, 0.84],  # Show 68% credible intervals
        plot_density=True,
        plot_datapoints=True,
        fill_contours=True,
        max_n_ticks=5
    )
    plt.xlim(0,1)
    plt.show()

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Posterior Summary Statistics

    Let's examine the key results from our Sequential NPE inference. The summary statistics show how well our 
    neural network learned to map the observed random walk data back to the original parameters.

    **Key Metrics:**

    - **Point estimates**: Mean posterior values for U and P
    - **Uncertainty**: Standard deviations showing inference confidence  
    - **Credible intervals**: 95% intervals containing the most probable parameter values
    - **Coverage**: Whether the true parameters fall within the credible intervals
    """
    )
    return


@app.cell
def _(summary_stats, true_parameters):
    """
    Display detailed summary statistics from the inference
    """
    print("📈 POSTERIOR SUMMARY STATISTICS")
    print("=" * 50)

    # Parameter estimates
    print(f"\n🎲 Parameter Estimates:")
    print(f"   U (Initial Occupancy):")
    print(f"     Posterior mean: {summary_stats['U_mean']:.4f}")
    print(f"     Posterior std:  {summary_stats['U_std']:.4f}")
    print(f"     True value:     {true_parameters[0]:.4f}")
    print(f"     Absolute error: {abs(summary_stats['U_mean'] - true_parameters[0]):.4f}")

    print(f"\n   P (Movement Probability):")
    print(f"     Posterior mean: {summary_stats['P_mean']:.4f}")
    print(f"     Posterior std:  {summary_stats['P_std']:.4f}")
    print(f"     True value:     {true_parameters[1]:.4f}")
    print(f"     Absolute error: {abs(summary_stats['P_mean'] - true_parameters[1]):.4f}")

    # Credible intervals
    print(f"\n📊 95% Credible Intervals:")
    U_ci = summary_stats['U_ci']
    P_ci = summary_stats['P_ci']

    print(f"   U: [{U_ci[0]:.4f}, {U_ci[1]:.4f}]")
    print(f"   P: [{P_ci[0]:.4f}, {P_ci[1]:.4f}]")

    # Coverage check
    U_covered = U_ci[0] <= true_parameters[0] <= U_ci[1]
    P_covered = P_ci[0] <= true_parameters[1] <= P_ci[1]

    print(f"\n✅ Credible Interval Coverage:")
    print(f"   U parameter: {'✓ Covered' if U_covered else '✗ Not covered'}")
    print(f"   P parameter: {'✓ Covered' if P_covered else '✗ Not covered'}")

    # Overall assessment
    both_covered = U_covered and P_covered
    print(f"\n🎯 Overall Assessment: {'Excellent inference!' if both_covered else 'Good inference with some uncertainty'}")

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Sequential NPE Training Details

    Let's examine how the Sequential NPE process worked for this inference. SNPE iteratively refines 
    the posterior estimate by focusing training data in the most relevant parameter regions.

    **Training Process:**
    1. **Round 1**: Sample uniformly from priors, train initial network
    2. **Rounds 2-N**: Sample from previous posterior, retrain with focused data
    3. **Convergence**: Monitor posterior changes between rounds
    4. **Final Result**: Converged posterior ready for inference
    """
    )
    return


@app.cell
def _(metadata, snpe_info):
    """
    Display SNPE training details and configuration
    """
    print("🔄 SEQUENTIAL NPE TRAINING DETAILS")
    print("=" * 50)

    # Training configuration
    print(f"\n⚙️ Configuration:")
    print(f"   Training approach: {metadata.get('training_approach', 'Unknown')}")
    print(f"   Lattice size: {metadata.get('lattice_size', 'Unknown')}")
    print(f"   Time steps: {metadata.get('time_steps', 'Unknown')}")
    print(f"   Random seed: {metadata.get('seed', 'Unknown')}")
    print(f"   Posterior samples: {metadata.get('n_samples', 'Unknown')}")

    # SNPE specific details
    if snpe_info:
        print(f"\n🔄 Sequential Training:")
        print(f"   Rounds completed: {snpe_info.get('rounds_completed', 'N/A')}")
        print(f"   Convergence achieved: {snpe_info.get('converged', 'N/A')}")

        if 'final_convergence_metric' in snpe_info:
            print(f"   Final convergence metric: {snpe_info['final_convergence_metric']:.6f}")

        print(f"\n📈 Training Efficiency:")
        print(f"   Sequential approach focuses training data in relevant parameter regions")
        print(f"   Each round uses posterior from previous round as proposal distribution")
        print(f"   This leads to better inference with fewer total simulations")

    # Load configuration from the original run
    config_path = "/Users/tkimpson/projects/NPE_for_RW_Inference/results/workflow_20250826_130359/config.txt"
    try:
        print(f"\n📋 Original Workflow Configuration:")
        with open(config_path, 'r') as f:
            config_lines = f.readlines()

        # Extract key training parameters
        for line in config_lines:
            if any(param in line for param in ['snpe_rounds', 'samples_per_round', 'convergence_threshold', 
                                               'max_epochs', 'hidden_features', 'num_transforms']):
                print(f"   {line.strip()}")

    except FileNotFoundError:
        print(f"   Configuration file not found")

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Comparison with Built-in SBI Visualization

    Let's also create the standard SBI posterior plots for comparison. The `sbi` library (which powers our NPE implementation) 
    has built-in visualization functions that provide a different perspective on the same posterior distribution.

    """
    )
    return


@app.cell
def _(RandomWalkNPE, plt, posterior_samples, torch, true_parameters):
    """
    Create SBI-style posterior visualizations for comparison
    """
    print("🔧 Creating SBI-style posterior plots...")

    # Convert to torch tensors for SBI plotting functions
    posterior_tensor = torch.tensor(posterior_samples, dtype=torch.float32)
    true_tensor = torch.tensor(true_parameters, dtype=torch.float32)

    # Create a temporary NPE object to access plotting methods
    npe_viz = RandomWalkNPE(device='cpu')

    # Plot marginal distributions
    fig1 = npe_viz.plot_posterior_samples(
        posterior_tensor, 
        true_tensor,
        figsize=(12, 5)
    )
    fig1.suptitle('SBI-style Marginal Posterior Distributions', fontsize=14, y=1.02)
    plt.show()

    # Plot pairwise relationships
    fig2 = npe_viz.plot_pairwise(
        posterior_tensor,
        true_tensor, 
        figsize=(8, 8)
    )
    fig2.suptitle('SBI-style Pairwise Posterior Relationships', fontsize=14, y=0.98)
    plt.show()

    return


if __name__ == "__main__":
    app.run()
