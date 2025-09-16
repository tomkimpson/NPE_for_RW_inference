import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    mo.md(r"""
    # Reproducing Classic ABC and MCMC Results

    This notebook reproduces the results from the classic ABC and MCMC inference methods for random walk models,
    based on the Julia notebooks in Simpson & Planck's [RandomWalkInference repository](https://github.com/ProfMJSimpson/RandomWalkInference).

    We implement three inference approaches:

    1. **ABC with Stochastic Simulator** - Direct simulation-based ABC
    2. **ABC with Surrogate Model** - Fast PDE-based ABC
    3. **MCMC with Surrogate Model** - Metropolis-Hastings with PDE likelihood

    The goal is to infer two key parameters:

    - **U**: Initial occupancy probability
    - **D**: Diffusion coefficient (related to movement probability P)



    For the Gaussian noise model, we will also infer a noise parameter $\sigma$.

    ---
    """)
    return (mo,)


@app.cell
def _():
    """
    Import required modules and set up the environment
    """
    import sys
    import os
    import time
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import special
    from scipy.stats import norm
    import seaborn as sns
    from collections import defaultdict

    # Add src directory to path
    sys.path.append(os.path.join(os.getcwd(), 'src'))

    # Import our existing simulator
    from simulator import RandomWalkSimulator

    # Set random seed for reproducibility
    np.random.seed(42)

    # Control variable for forcing re-computation
    force_rerun = False

    # Create results directory if it doesn't exist
    results_dir = "notebooks/results"
    os.makedirs(results_dir, exist_ok=True)

    print("✅ Successfully imported all modules!")
    return (
        RandomWalkSimulator,
        np,
        os,
        pickle,
        plt,
        results_dir,
        sns,
        special,
        time,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1. Simulation Setup and Parameters

    We use the same lattice and simulation parameters as the original Julia notebooks:
    - **Lattice**: 200 × 50 (matching original setup)
    - **Time steps**: 72
    - **Initial region**: Central strip with half-width 10
    - **True parameters**: U ≈ 0.5, D ≈ 0.25 (for validation)
    """
    )
    return


@app.cell
def _(RandomWalkSimulator):
    """
    Set up simulation parameters matching the Julia notebooks
    """
    # Lattice parameters (matching Julia setup)
    Lx = 200  # Number of columns
    Ly = 50   # Number of rows
    T = 100   # Number of time steps (Julia uses T=100.0)
    initial_region_half_width = 25  # Central region for initial placement (Julia uses h=25)

    # True parameter values for generating synthetic observations
    U_true = 0.5   # Initial occupancy probability
    P_true = 1.0

    # Convert movement probability P to Diffusion coefficient D for the PDE model
    # From Julia code: D is related to P through diffusion relationship
    # For discrete random walk: D ≈ P * a^2 / (2 * dt) where a is lattice spacing
    convert_P_to_D = lambda P: P / 4
    convert_D_to_P = lambda D: min(D * 4, 1) #clipped to always be \leq 1
    assert convert_D_to_P(convert_P_to_D(1.0)) == 1.0 #check inverse transform always defined properly

    D_true = convert_P_to_D(P_true)

    # Create simulator instance
    simulator = RandomWalkSimulator(Lx, Ly, initial_region_half_width)

    print(f"📐 Lattice size: {Lx} × {Ly}")
    print(f"⏱️  Time steps: {T}")
    print(f"🎯 True parameters: U = {U_true}, D = {D_true}, P = {P_true}")
    return (
        D_true,
        Lx,
        Ly,
        P_true,
        T,
        U_true,
        convert_D_to_P,
        initial_region_half_width,
        simulator,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. Generate Synthetic Observed Data

    We generate synthetic "observed" data using our stochastic simulator with the true parameters.
    This will serve as the target data for our ABC and MCMC inference methods.
    """
    )
    return


@app.cell
def _(Lx, Ly, P_true, T, U_true, np, simulator):
    """
    Generate synthetic observed data using the stochastic simulator
    """
    print("🎲 Generating synthetic observed data...")
    # Generate new data with our simulator
    observed_column_counts, initial_positions, final_positions = simulator.simulate(U=U_true, P=P_true, T=T, random_seed=1234)

    print(f"📊 Generated observed data with {np.sum(observed_column_counts)} total agents")
    print(f"📈 Column counts shape: {observed_column_counts.shape}")
    print(f"📊 Non-zero columns: {np.sum(observed_column_counts > 0)}")



    #Plot it
    from simulator import plot_simulation_comparison
    # Create main comparison plot
    fig_comparison = plot_simulation_comparison(
        initial_positions=initial_positions,
        final_positions=final_positions,
        column_counts=observed_column_counts,
        Lx=Lx,
        Ly=Ly,
        U=U_true,
        P=P_true,
        T=T,
        figsize=(18, 6)
    )
    fig_comparison.suptitle('Random Walk Simulation Results', fontsize=16, y=1.02)
    return (observed_column_counts,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. Surrogate PDE Model Implementation

    The surrogate model uses the analytical PDE solution for a continuous diffusion process:

    $$u(x,t) = U \cdot \frac{1}{2}\left[\text{erf}\left(\frac{h-x}{\sqrt{4Dt}}\right) + \text{erf}\left(\frac{h+x}{\sqrt{4Dt}}\right)\right]$$

    where:
    - $h$ is the initial region half-width
    - $D$ is the diffusion coefficient
    - $U$ is the initial occupancy probability
    """
    )
    return


@app.cell
def _(
    D_true,
    Lx,
    T,
    U_true,
    initial_region_half_width,
    np,
    observed_column_counts,
    plt,
    special,
):
    """
    Implement the surrogate PDE model for fast likelihood computation
    """
    def surrogate_pde_model(U, D, t, Lx, h):
        """
        Compute the analytical PDE solution for the random walk model

        Parameters:
        - U: Initial occupancy probability
        - D: Diffusion coefficient
        - t: Time
        - Lx: Lattice width
        - h: Initial region half-width

        Returns:
        - column_counts: Predicted agent counts per column
        """
        # Create x coordinates (centered at 0)
        x_coords = np.arange(-Lx//2, Lx//2)

        # Compute PDE solution at each x coordinate
        if D <= 0 or t <= 0:
            return np.zeros(len(x_coords))

        sqrt_4Dt = np.sqrt(4 * D * t)

        # PDE solution: u(x,t) = U * 0.5 * [erf((h-x)/sqrt(4Dt)) + erf((h+x)/sqrt(4Dt))]
        term1 = special.erf((h - x_coords) / sqrt_4Dt)
        term2 = special.erf((h + x_coords) / sqrt_4Dt)

        density = U * 0.5 * (term1 + term2)

        # Convert density to counts (multiply by lattice height - matching Julia LY=50)
        column_counts = density * 50  # Julia uses LY=50, not Lx=200

        return np.maximum(column_counts, 0)  # Ensure non-negative

    # Test the surrogate model
    test_counts = surrogate_pde_model(U=U_true, D=D_true, t=T, Lx=Lx, h=initial_region_half_width)
    print(f"✅ Surrogate model implemented")
    print(f"📊 Test output shape: {test_counts.shape}")
    print(f"📈 Test total agents: {np.sum(test_counts):.1f}")




    #Plot it
    fig_pde, axes = plt.subplots(1, 1, figsize=(8, 6))
    x_min = -(Lx // 2)
    x_max = Lx // 2 if Lx % 2 == 1 else (Lx // 2) - 1

    # Column counts
    x_positions = np.arange(x_min, x_max + 1)
    axes.scatter(x_positions, observed_column_counts, alpha=0.7, color='skyblue', edgecolor='navy')
    axes.plot(x_positions, test_counts, color='r')

    return (surrogate_pde_model,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. ABC with Surrogate Model

    ABC using the fast analytical PDE solution instead of running full stochastic simulations.
    This is much faster and allows us to generate many more samples.
    """
    )
    return


@app.cell
def _(
    Lx,
    T,
    np,
    observed_column_counts,
    os,
    pickle,
    results_dir,
    surrogate_pde_model,
    time,
):
    """
    ABC rejection sampling using the surrogate PDE model
    """
    import tqdm 
    def abc_surrogate(observed_data,U_limits,D_limits, n_samples=100000, n_keep=1000):
        """
        ABC rejection sampling using surrogate PDE model
        """
        print(f"🔄 Running ABC with surrogate model ({n_samples:,} samples)...")
        start_time = time.time()

        #Extract limits 
        U_min,U_max = U_limits
        D_min,D_max = D_limits

        print(f"Limits for ABC sampling with PDE are: U: {U_limits}, D: ({D_min}, {D_max})")


        # Storage for results
        samples_U = []
        samples_D = []
        distances = []

        for i in tqdm.tqdm(range(n_samples)):

            # Sample parameters from priors
            U_sample = np.random.uniform(U_min, U_max)
            D_sample = np.random.uniform(D_min, D_max)

            # Generate prediction using surrogate model
            pred_counts = surrogate_pde_model(U_sample, D_sample, T, Lx, h=25)

            # Compute distance to observed data
            distance = np.linalg.norm(pred_counts - observed_data)

            samples_U.append(U_sample)
            samples_D.append(D_sample)
            distances.append(distance)

        # Convert to arrays
        samples_U = np.array(samples_U)
        samples_D = np.array(samples_D)
        distances = np.array(distances)

        # Keep the n_keep samples with smallest distances
        keep_indices = np.argsort(distances)[:n_keep]

        abc_results = {
            'U': samples_U[keep_indices],
            'D': samples_D[keep_indices],
            'distances': distances[keep_indices],
            'n_samples': n_samples,
            'n_keep': n_keep
        }

        elapsed = time.time() - start_time
        print(f"✅ ABC surrogate completed in {elapsed:.1f}s")
        print(f"📊 Kept {n_keep:,} best samples")

        return abc_results



    U_limits = (0.3,0.7)
    D_limits = (0.05,0.40)
    # Check if results exist and load if available
    abc_surrogate_file = f"{results_dir}/abc_surrogate_results.pkl"
    force_rerun_abc_surrogate = True
    if os.path.exists(abc_surrogate_file) and not force_rerun_abc_surrogate:
        print("📁 Loading existing ABC surrogate results...")
        with open(abc_surrogate_file, 'rb') as f1:
            abc_surrogate_results = pickle.load(f1)
        print(f"✅ Loaded {abc_surrogate_results['n_keep']} samples from {abc_surrogate_file}")
    else:
        # Run ABC with surrogate model
        abc_surrogate_results = abc_surrogate(observed_column_counts,U_limits,D_limits, n_samples=100000, n_keep=1000)

        # Save results
        with open(abc_surrogate_file, 'wb') as f1:
            pickle.dump(abc_surrogate_results, f1)
        print(f"💾 Saved ABC surrogate results to {abc_surrogate_file}")
    return (abc_surrogate_results,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7. Results and Visualizations

    Compare the results from all three inference methods:
    1. Parameter histograms showing posterior distributions
    2. 2D correlation plots
    3. MCMC trace plots
    4. Summary statistics
    """
    )
    return


@app.cell
def _(D_true, U_true, abc_surrogate_results, plt, sns):
    """
    Create comprehensive visualization of results
    """


    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create main comparison figure
    fig_hist, axes_hist = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Parameter U histograms


    axes_hist[0, 1].hist(abc_surrogate_results['U'], bins=50, alpha=0.7, label='ABC Surrogate', density=True, color='orange')
    axes_hist[0, 1].axvline(U_true, color='red', linestyle='--', linewidth=2, label=f'True U={U_true}')
    axes_hist[0, 1].set_xlabel('U (Initial Occupancy)')
    axes_hist[0, 1].set_ylabel('Density')
    axes_hist[0, 1].set_title('ABC with Surrogate Model')
    axes_hist[0, 1].legend()
    axes_hist[0, 1].grid(True, alpha=0.3)
    axes_hist[0,1].set_xlim(0.40,0.60)



    # Row 2: Parameter D histograms


    axes_hist[1, 1].hist(abc_surrogate_results['D'], bins=50, alpha=0.7, density=True, color='orange')
    axes_hist[1, 1].axvline(D_true, color='red', linestyle='--', linewidth=2, label=f'True D={D_true}')
    axes_hist[1, 1].set_xlabel('D (Diffusion Coefficient)')
    axes_hist[1, 1].set_ylabel('Density')
    axes_hist[1, 1].legend()
    axes_hist[1, 1].grid(True, alpha=0.3)
    axes_hist[1,1].set_xlim(0.0,0.5)



    plt.tight_layout()
    plt.suptitle('Comparison of ABC and MCMC Inference Results', fontsize=16, y=1.02)

    # Save the plot
    plt.show()
    return


@app.cell
def _(D_true, U_true, abc_surrogate_results, plt):
    """
    Create 2D parameter correlation plots
    """
    fig_corr, axes_corr = plt.subplots(1, 3, figsize=(15, 5))



    # ABC Surrogate
    axes_corr[1].scatter(abc_surrogate_results['U'], abc_surrogate_results['D'],
                   alpha=0.6, s=10, color='orange', label='ABC Samples')
    axes_corr[1].scatter(U_true, D_true, color='red', s=100, marker='*',
                   label=f'True: U={U_true}, D={D_true}', zorder=5)
    axes_corr[1].set_xlabel('U (Initial Occupancy)')
    axes_corr[1].set_ylabel('D (Diffusion Coefficient)')
    axes_corr[1].set_title('ABC with Surrogate Model')
    axes_corr[1].legend()
    axes_corr[1].grid(True, alpha=0.3)




    #     U_limits = (0.30, 0.70)
    # P_limits = (0.2, 1.0)

    plt.tight_layout()
    plt.suptitle('2D Parameter Correlations', fontsize=16, y=1.02)

    # Save the plot
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Corner plots


    Lets combine the above results into a nice corner plot
    """
    )
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 8. Posterior Predictive Checks

    To validate our inference results, we perform **posterior predictive checks** by taking parameter samples from each method and pushing them through the simulator to generate new predictions. This allows us to:

    1. **Validate model fit**: Check if our inferred parameters can reproduce the observed data
    2. **Compare methods**: See which inference approach provides the most accurate predictions
    3. **Assess uncertainty**: Visualize prediction intervals showing the full range of possible outcomes

    We'll take samples from each method (ABC Stochastic, ABC Surrogate, MCMC) and generate new simulations using those parameter values. The resulting predictions should ideally encompass the observed data within their uncertainty bands.
    """
    )
    return


@app.cell
def _(T, convert_D_to_P, np, simulator, time):
    """
    Helper functions for posterior predictive sampling
    """

    def posterior_predictive_sample(parameter_samples, method_name, n_pred_samples=100, random_seed=42):
        """
        Generate posterior predictive samples by running simulator with parameter samples.

        Parameters:
        -----------
        parameter_samples : dict with keys 'U' and 'D'
            Parameter samples from ABC or MCMC methods
        method_name : str
            Name of the method for progress reporting
        n_pred_samples : int
            Number of predictive samples to generate
        random_seed : int
            Random seed for reproducibility

        Returns:
        --------
        np.ndarray of shape (n_pred_samples, n_columns)
            Predicted column counts for each parameter sample
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        U_samples = parameter_samples['U']
        D_samples = parameter_samples['D']

        # Convert to arrays if not already
        U_samples = np.array(U_samples)
        D_samples = np.array(D_samples)

        n_available = len(U_samples)
        n_pred = min(n_pred_samples, n_available)

        # Subsample if we have more samples than requested
        if n_pred < n_available:
            indices = np.random.choice(n_available, size=n_pred, replace=False)
            U_selected = U_samples[indices]
            D_selected = D_samples[indices]
        else:
            U_selected = U_samples[:n_pred]
            D_selected = D_samples[:n_pred]

        print(f"🔮 Generating {n_pred} posterior predictive samples for {method_name}...")

        # Initialize results array
        n_columns = simulator.Lx
        predictions = np.zeros((n_pred, n_columns), dtype=int)

        # Generate predictions
        start_time = time.time()
        for i, (U, D) in enumerate(zip(U_selected, D_selected)):
            if i % max(1, n_pred // 10) == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (n_pred - i) / rate if rate > 0 else 0
                print(f"   Progress: {i}/{n_pred} ({100*i/n_pred:.1f}%) - ETA: {eta:.1f}s")

            # Convert D to P for simulator
            P = convert_D_to_P(D)

            # Run simulation with current parameter sample
            column_counts, _, _ = simulator.simulate(
                U=float(U),
                P=float(P),
                T=T,
                random_seed=random_seed + i if random_seed is not None else None
            )

            predictions[i] = column_counts

        elapsed = time.time() - start_time
        print(f"✅ Predictive sampling for {method_name} completed in {elapsed:.1f} seconds")

        return predictions

    return (posterior_predictive_sample,)


@app.cell
def _(np):
    """
    Function to compute prediction intervals and summary statistics
    """

    def compute_prediction_intervals(predictions, method_name, percentiles=[2.5, 25, 50, 75, 97.5]):
        """
        Compute prediction intervals and summary statistics.

        Parameters:
        -----------
        predictions : np.ndarray of shape (n_samples, n_columns)
            Predicted column counts
        method_name : str
            Name of the method for reporting
        percentiles : List[float]
            Percentiles to compute for intervals

        Returns:
        --------
        Dict containing intervals, summary statistics, and metadata
        """
        print(f"📊 Computing prediction intervals for {method_name}...")

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
                'method_name': method_name,
                'n_predictions': n_pred,
                'n_columns': n_columns,
                'percentiles': percentiles
            }
        }

        print(f"✅ Computed intervals for {n_columns} columns")
        print(f"   Total agents - Mean: {global_stats['total_agents_mean']:.1f} ± {global_stats['total_agents_std']:.1f}")

        return results

    return (compute_prediction_intervals,)


@app.cell
def _(Lx, T, np, surrogate_pde_model):
    """
    PDE-based posterior predictive sampling (matching Julia approach)
    """

    def posterior_predictive_pde_sample(parameter_samples, method_name, n_pred_samples=1000):
        """
        Generate posterior predictive samples using the PDE model (surrogate).
        This matches the Julia approach where samples are pushed through the analytical solution.

        Parameters:
        -----------
        parameter_samples : dict with keys 'U' and 'D'
            Parameter samples from ABC or MCMC methods
        method_name : str
            Name of the method for progress reporting
        n_pred_samples : int
            Number of predictive samples to generate

        Returns:
        --------
        np.ndarray of shape (n_pred_samples, n_columns)
            Predicted column counts for each parameter sample using PDE model
        """
        U_samples = parameter_samples['U']
        D_samples = parameter_samples['D']

        # Convert to arrays if not already
        U_samples = np.array(U_samples)
        D_samples = np.array(D_samples)

        n_available = len(U_samples)
        n_pred = min(n_pred_samples, n_available)

        # Subsample if we have more samples than requested
        if n_pred < n_available:
            indices = np.random.choice(n_available, size=n_pred, replace=False)
            U_selected = U_samples[indices]
            D_selected = D_samples[indices]
        else:
            U_selected = U_samples[:n_pred]
            D_selected = D_samples[:n_pred]

        print(f"🔮 Generating {n_pred} PDE-based predictions for {method_name}...")

        # Initialize results array
        n_columns = Lx
        predictions = np.zeros((n_pred, n_columns))

        # Generate PDE predictions for each parameter sample
        for i, (U, D) in enumerate(zip(U_selected, D_selected)):
            if i % max(1, n_pred // 10) == 0 and i > 0:
                print(f"   Progress: {i}/{n_pred} ({100*i/n_pred:.1f}%)")

            # Use surrogate PDE model to generate prediction
            pred_counts = surrogate_pde_model(U, D, T, Lx, h=25)
            predictions[i] = pred_counts

        print(f"✅ PDE-based predictive sampling for {method_name} completed")
        return predictions

    return (posterior_predictive_pde_sample,)


@app.cell
def _(np):
    """
    Function to compute envelope bounds for PDE predictions (matching Julia approach)
    """

    def compute_pde_prediction_envelope(pde_predictions, method_name):
        """
        Compute envelope bounds for PDE-based predictions, matching the Julia approach.
        This computes min/max bounds at each x location across all parameter samples.

        Parameters:
        -----------
        pde_predictions : np.ndarray of shape (n_samples, n_columns)
            PDE predictions for each parameter sample
        method_name : str
            Name of the method for reporting

        Returns:
        --------
        Dict containing envelope bounds and summary statistics
        """
        print(f"📊 Computing PDE prediction envelope for {method_name}...")

        n_pred, n_columns = pde_predictions.shape

        # Compute envelope bounds (matching Julia logic)
        envelope = {
            'lower': np.min(pde_predictions, axis=0),  # minimum at each x location
            'upper': np.max(pde_predictions, axis=0),  # maximum at each x location
        }

        # Also compute percentile-based intervals for comparison
        percentiles = [2.5, 25, 50, 75, 97.5]
        intervals = {}
        for p in percentiles:
            intervals[f"p{p}"] = np.percentile(pde_predictions, p, axis=0)

        # Summary statistics
        stats = {
            'mean': np.mean(pde_predictions, axis=0),
            'std': np.std(pde_predictions, axis=0),
            'envelope_width': envelope['upper'] - envelope['lower']
        }

        # Global summary statistics
        global_stats = {
            'total_agents_mean': np.mean(np.sum(pde_predictions, axis=1)),
            'total_agents_std': np.std(np.sum(pde_predictions, axis=1)),
            'envelope_mean_width': np.mean(envelope['upper'] - envelope['lower'])
        }

        results = {
            'envelope': envelope,
            'intervals': intervals,
            'column_stats': stats,
            'global_stats': global_stats,
            'predictions': pde_predictions,
            'metadata': {
                'method_name': method_name,
                'n_predictions': n_pred,
                'n_columns': n_columns,
                'percentiles': percentiles
            }
        }

        print(f"✅ Computed envelope for {n_columns} columns")
        print(f"   Mean envelope width: {global_stats['envelope_mean_width']:.2f}")
        print(f"   Total agents - Mean: {global_stats['total_agents_mean']:.1f} ± {global_stats['total_agents_std']:.1f}")

        return results

    return (compute_pde_prediction_envelope,)


@app.cell
def _(Lx, np, plt):
    """
    Visualization function for prediction intervals
    """

    def plot_prediction_comparison(prediction_results_list, observed_data, method_names, colors):
        """
        Create comparative visualization of prediction intervals from multiple methods.

        Parameters:
        -----------
        prediction_results_list : List[Dict]
            List of prediction results from compute_prediction_intervals
        observed_data : np.ndarray
            Observed column counts for comparison
        method_names : List[str]
            Names of the methods for legend
        colors : List[str]
            Colors for each method

        Returns:
        --------
        matplotlib Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        # Center column indices around 0 (same logic as other plots in notebook)
        x_min = -(Lx // 2)
        x_max = Lx // 2 if Lx % 2 == 1 else (Lx // 2) - 1
        columns = np.arange(x_min, x_max + 1)

        # Plot prediction intervals for each method
        for i, (prediction_results, method_name, color) in enumerate(zip(prediction_results_list, method_names, colors)):
            intervals = prediction_results['intervals']
            stats = prediction_results['column_stats']

            # Plot 95% prediction interval as filled area
            ax.fill_between(columns, intervals['p2.5'], intervals['p97.5'],
                           alpha=0.15, color=color, label=f'{method_name} 95% Interval')

            # Plot 50% prediction interval
            ax.fill_between(columns, intervals['p25'], intervals['p75'],
                           alpha=0.25, color=color, label=f'{method_name} 50% Interval')

            # Plot median prediction
            ax.plot(columns, intervals['p50'], '-', linewidth=2, color=color,
                   label=f'{method_name} Median', alpha=0.8)

        # Plot observed data
        ax.scatter(columns, observed_data, c='red', s=60,
                  label='Observed Data', zorder=10, marker='o', edgecolor='darkred')

        ax.set_xlabel('Column Index (centered)')
        ax.set_ylabel('Agent Count')
        ax.set_title('Posterior Predictive Checks: Comparison of ABC and MCMC Methods')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    return (plot_prediction_comparison,)


@app.cell
def _(Lx, np, plt):
    """
    PDE-specific visualization function matching Julia style
    """

    def plot_pde_prediction_envelope(envelope_results, observed_data, method_name, color='green'):
        """
        Create visualization of PDE prediction envelope matching Julia style.
        This replicates the Julia plotting approach with scatter + filled envelope.

        Parameters:
        -----------
        envelope_results : Dict
            Results from compute_pde_prediction_envelope
        observed_data : np.ndarray
            Observed column counts for comparison
        method_name : str
            Name of the method for title
        color : str
            Color for the envelope fill

        Returns:
        --------
        matplotlib Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Center column indices around 0 (matching notebook logic)
        x_min = -(Lx // 2)
        x_max = Lx // 2 if Lx % 2 == 1 else (Lx // 2) - 1
        xxloc = np.arange(x_min, x_max + 1)

        # Get envelope bounds
        lower = envelope_results['envelope']['lower']
        upper = envelope_results['envelope']['upper']
        mean_pred = envelope_results['column_stats']['mean']

        # Plot observed data as scatter (matching Julia: scatter(xxloc,data,mc=:blue))
        ax.scatter(xxloc, observed_data, c='blue', s=50,
                  label='Observed Data', zorder=10, marker='o', edgecolor='darkblue')

        # Plot envelope as filled region (matching Julia: fillrange=upper,fillalpha=0.25,color=:green)
        ax.fill_between(xxloc, lower, upper, alpha=0.25, color=color,
                       label=f'{method_name} Prediction Envelope')

        # Plot mean prediction line
        ax.plot(xxloc, mean_pred, '-', linewidth=2, color='darkgreen',
               label=f'{method_name} Mean', alpha=0.8)

        # Set axis properties matching Julia style
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('N, H u(x,t)', fontsize=12)
        ax.set_title(f'PDE-based Posterior Predictive Check: {method_name}', fontsize=14)

        # Set axis limits (matching Julia: xlims=(-100,100),ylims=(-10,40))
        ax.set_xlim(-100, 100)
        ax.set_ylim(-10, 40)

        # Set tick properties (matching Julia tick formatting)
        ax.set_xticks([-100, -50, 0, 50, 100])
        ax.set_xticklabels(['-100', '-50', '0', '50', '100'], fontsize=12)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_yticklabels(['0', '10', '20', '30', '40'], fontsize=12)

        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    return (plot_pde_prediction_envelope,)


@app.cell
def _(
    abc_surrogate_results,
    compute_prediction_intervals,
    posterior_predictive_sample,
):
    """
    Generate posterior predictive samples for ABC Surrogate method
    """
    # Generate predictions using ABC Surrogate posterior samples
    abc_surrogate_predictions = posterior_predictive_sample(
        parameter_samples=abc_surrogate_results,
        method_name="ABC Surrogate",
        n_pred_samples=100,
        random_seed=43
    )

    # Compute prediction intervals
    abc_surrogate_pred_results = compute_prediction_intervals(
        predictions=abc_surrogate_predictions,
        method_name="ABC Surrogate"
    )

    return (abc_surrogate_pred_results,)


@app.cell
def _(
    abc_surrogate_pred_results,
    observed_column_counts,
    plot_prediction_comparison,
    plt,
):
    """
    Create comparative visualization of posterior predictive checks
    """
    # Set up method comparison
    prediction_results_list = [

        abc_surrogate_pred_results,

    ]
    method_names = ['ABC Surrogate']
    colors = ['orange']


    for i,p in enumerate(prediction_results_list):

        fig_predictive = plot_prediction_comparison([p],
            observed_data=observed_column_counts,
            method_names=[method_names[i]],
            colors=[colors[i]]
        )
        plt.show()

    # Save the plot
    #plt.savefig('notebooks/images/posterior_predictive_comparison.png', dpi=300, bbox_inches='tight')


    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 9. PDE-based Posterior Predictive Checks (Matching Julia Approach)

    Here we implement the approach from the Julia code where posterior samples are pushed through the
    **PDE model** (not the stochastic simulator) to generate credible intervals. This matches the
    Julia logic:

    ```julia
    for i in 1:M
        C(x) = LY*smallU[i]*(erf((h-x)/sqrt(4*smallD[i]*T))+erf((h+x)/sqrt(4*smallD[i]*T)))/2;
        # compute envelope bounds...
    end
    ```

    This approach is much faster since it uses the analytical PDE solution rather than running
    stochastic simulations for each parameter sample.
    """
    )
    return


@app.cell
def _(
    abc_surrogate_results,
    compute_pde_prediction_envelope,
    np,
    observed_column_counts,
    plot_pde_prediction_envelope,
    plt,
    posterior_predictive_pde_sample,
):
    """
    Generate PDE-based posterior predictive checks using ABC surrogate samples
    """
    # Generate PDE-based predictions (much faster than stochastic simulation)
    pde_predictions = posterior_predictive_pde_sample(
        parameter_samples=abc_surrogate_results,
        method_name="ABC Surrogate PDE",
        n_pred_samples=1000  # Use all available samples for better envelope
    )

    # Compute envelope bounds (matching Julia approach)
    pde_envelope_results = compute_pde_prediction_envelope(
        pde_predictions=pde_predictions,
        method_name="ABC Surrogate PDE"
    )

    # Create visualization matching Julia style
    fig_pde_envelope = plot_pde_prediction_envelope(
        envelope_results=pde_envelope_results,
        observed_data=observed_column_counts,
        method_name="ABC Surrogate",
        color='green'
    )

    plt.show()

    # Print summary comparison
    print("\n" + "="*60)
    print("COMPARISON: Stochastic vs PDE-based Predictions")
    print("="*60)
    print(f"PDE envelope mean width: {pde_envelope_results['global_stats']['envelope_mean_width']:.2f}")
    print(f"PDE total agents: {pde_envelope_results['global_stats']['total_agents_mean']:.1f} ± {pde_envelope_results['global_stats']['total_agents_std']:.1f}")
    print(f"Observed total agents: {np.sum(observed_column_counts)}")

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
