import marimo

__generated_with = "0.15.2"
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
    P_true = 0.7

    # Convert movement probability P to Diffusion coefficient D for the PDE model
    # From Julia code: D is related to P through diffusion relationship
    # For discrete random walk: D ≈ P * a^2 / (2 * dt) where a is lattice spacing
    convert_P_to_D = lambda P: P / 4
    convert_D_to_P = lambda D: D * 4
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
        convert_P_to_D,
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
    ## 4. ABC with Stochastic Simulator

    Approximate Bayesian Computation using our stochastic random walk simulator.
    We generate many parameter samples, run simulations, and keep the ones that best match the observed data.
    """
    )
    return


@app.cell
def _(
    T,
    convert_P_to_D,
    np,
    observed_column_counts,
    os,
    pickle,
    results_dir,
    simulator,
    time,
):
    """
    ABC rejection sampling using the stochastic simulator
    """
    import tqdm
    def abc_stochastic(observed_data,U_limits,P_limits, n_samples=100000, n_keep=1000):
        """
        ABC rejection sampling using stochastic simulator
        """
        print(f"🔄 Running ABC with stochastic simulator ({n_samples:,} samples)...")
        start_time = time.time()



        # Storage for results
        samples_U = []
        samples_D = []
        distances = []

        #Extract limits 
        U_min,U_max = U_limits
        P_min,P_max = P_limits

        print(f"Limits for ABC sampling are: U: {U_limits}, P: {P_limits}")
    
        for i in tqdm.tqdm(range(n_samples)):

            # Sample parameters from priors
            U_sample = np.random.uniform(U_min, U_max)
            P_sample = np.random.uniform(P_min, P_max)

            # Run simulation
            sim_counts, _, _ = simulator.simulate(U=U_sample, P=P_sample, T=T, random_seed=None)

            # Compute distance to observed data
            distance = np.linalg.norm(sim_counts - observed_data)

            #Save
            samples_U.append(U_sample)
            samples_D.append(convert_P_to_D(P_sample))
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
            'n_samples': len(samples_U),
            'n_keep': n_keep
        }

        elapsed = time.time() - start_time
        print(f"✅ ABC stochastic completed in {elapsed:.1f}s")
        print(f"📊 Kept {n_keep:,} best samples out of {len(samples_U):,}")

        return abc_results






    # Parameter ranges
    U_limits = (0.30, 0.70)
    P_limits = (0.2, 1.0)


    # Check if results exist and load if available
    abc_stochastic_file = f"{results_dir}/abc_stochastic_results.pkl"
    force_rerun_abc_stochastic = True

    if os.path.exists(abc_stochastic_file) and not force_rerun_abc_stochastic:
        print("📁 Loading existing ABC stochastic results...")
        with open(abc_stochastic_file, 'rb') as f:
            abc_stochastic_results = pickle.load(f)
        print(f"✅ Loaded {abc_stochastic_results['n_keep']} samples from {abc_stochastic_file}")
    else:
        # Run ABC with stochastic simulator
        abc_stochastic_results = abc_stochastic(observed_column_counts,U_limits,P_limits, n_samples=10, n_keep=10)

        # Save results
        with open(abc_stochastic_file, 'wb') as f:
            pickle.dump(abc_stochastic_results, f)
        print(f"💾 Saved ABC stochastic results to {abc_stochastic_file}")
    return P_limits, U_limits, abc_stochastic_results, tqdm


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
    P_limits,
    T,
    U_limits,
    convert_P_to_D,
    np,
    observed_column_counts,
    os,
    pickle,
    results_dir,
    surrogate_pde_model,
    time,
    tqdm,
):
    """
    ABC rejection sampling using the surrogate PDE model
    """
    def abc_surrogate(observed_data,U_limits,P_limits, n_samples=100000, n_keep=1000):
        """
        ABC rejection sampling using surrogate PDE model
        """
        print(f"🔄 Running ABC with surrogate model ({n_samples:,} samples)...")
        start_time = time.time()

        #Extract limits 
        U_min,U_max = U_limits
        P_min,P_max = P_limits
        D_min,D_max = convert_P_to_D(P_min),convert_P_to_D(P_max)

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
        abc_surrogate_results = abc_surrogate(observed_column_counts,U_limits,P_limits, n_samples=10, n_keep=10)

        # Save results
        with open(abc_surrogate_file, 'wb') as f1:
            pickle.dump(abc_surrogate_results, f1)
        print(f"💾 Saved ABC surrogate results to {abc_surrogate_file}")
    return (abc_surrogate_results,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6. MCMC with Surrogate Model

    Metropolis-Hastings MCMC using the surrogate PDE model as the likelihood function.
    We also infer an observation noise parameter σ.
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
):
    """
    MCMC sampling using surrogate model - Julia style implementation
    """

    def loglikelihood(data, theta):
        """Log-likelihood function matching Julia implementation"""
        U, D, sigma = theta
        # Generate prediction using surrogate model
        pred = surrogate_pde_model(U, D, T, Lx, h=25)
        # Gaussian likelihood: log p(data | pred, sigma)
        residuals = data - pred
        log_lik = -0.5 * np.sum(residuals**2) / sigma**2
        log_lik -= len(data) * np.log(sigma)
        log_lik -= 0.5 * len(data) * np.log(2 * np.pi)
        return log_lik

    def mcmc(observed_data, n_samples=10000):
        """
        MCMC implementation matching Julia structure exactly:
        x = rand(d)                      # Proposal noise
        θn = max.(ε, θ .+ x)             # Proposed θn
        ℓn = loglhood(data, θn)         # Likelihood at proposed θn
        α = min(1.0, exp(ℓn - ℓ))      # Acceptance ratio
        """
        print(f"🔄 Running Julia-style MCMC ({n_samples:,} samples)...")


        # Initialize parameters (matching Julia)
        theta = np.array([0.1, 0.1, 1.0])  # [U, D, sigma]
        epsilon = 1e-6  # To keep parameters positive

        # Multivariate normal for proposals (matching Julia)
        # μ=[0,0,0]; var= [1e-4,1e-4,1e-3]; d = MvNormal(μ, Diagonal(var));
        proposal_cov = np.diag([1e-4, 1e-4, 1e-3])

        # Storage for samples
        samples = np.zeros((4, n_samples))  # [likelihood, U, D, sigma]

        # Initial likelihood
        current_loglik = loglikelihood(observed_data, theta)
        kount = 0

        while kount < n_samples:
            # Julia: x = rand(d) - sample from multivariate normal
            x = np.random.multivariate_normal([0, 0, 0], proposal_cov)

            # Julia: θn = max.(ε, θ .+ x) - proposed parameters with positivity
            theta_proposed = np.maximum(epsilon, theta + x)

            # Julia: ℓn = loglhood(data, θn) - proposed likelihood
            proposed_loglik = loglikelihood(observed_data, theta_proposed)

            # Julia: α = min(1.0, exp(ℓn - ℓ)) - acceptance ratio
            alpha = min(1.0, np.exp(proposed_loglik - current_loglik))

            # Accept/reject
            if np.random.rand() <= alpha:
                # Accept proposal (Julia only increments kount on accept)
                kount += 1
                theta = theta_proposed
                current_loglik = proposed_loglik

                # Store sample (matching Julia samples array)
                samples[0, kount-1] = current_loglik
                samples[1, kount-1] = theta[0]  # U
                samples[2, kount-1] = theta[1]  # D
                samples[3, kount-1] = theta[2]  # sigma

                if kount % 1000 == 0:
                    print(f"  Accepted sample: {kount:,}")

        mcmc_results = {
            'U': samples[1, :],
            'D': samples[2, :],
            'sigma': samples[3, :],
            'loglik': samples[0, :],
            'n_samples': n_samples
        }

        print(f"✅ Julia-style MCMC completed")
        print(f"📊 Generated {n_samples:,} posterior samples")

        return mcmc_results


    # Check if results exist and load if available
    mcmc_file = f"{results_dir}/mcmc_results.pkl"

    force_mcmc_rerun = True
    if os.path.exists(mcmc_file) and not force_mcmc_rerun:
        print("📁 Loading existing MCMC results...")
        with open(mcmc_file, 'rb') as f2:
            mcmc_results = pickle.load(f2)
        print(f"✅ Loaded {mcmc_results['n_samples']} samples from {mcmc_file}")
    else:
        # Run MCMC
        mcmc_results = mcmc(observed_column_counts, n_samples=8000)

        # Save results
        with open(mcmc_file, 'wb') as f2:
            pickle.dump(mcmc_results, f2)
        print(f"💾 Saved MCMC results to {mcmc_file}")
    return (mcmc_results,)


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
def _(
    D_true,
    U_true,
    abc_stochastic_results,
    abc_surrogate_results,
    mcmc_results,
    plt,
    sns,
):
    """
    Create comprehensive visualization of results
    """




    # Burnd first n MCMC samples
    burn_in = 2000
    # The 'if' condition prevents an error by not slicing the integer 'n_samples'
    mcmc_results_burned = {
        key: value[burn_in:]
        for key, value in mcmc_results.items()
        if key != 'n_samples'
    }





    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create main comparison figure
    fig_hist, axes_hist = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Parameter U histograms
    axes_hist[0, 0].hist(abc_stochastic_results['U'], bins=50, alpha=0.7, label='ABC Stochastic', density=True)
    axes_hist[0, 0].axvline(U_true, color='red', linestyle='--', linewidth=2, label=f'True U={U_true}')
    axes_hist[0, 0].set_xlabel('U (Initial Occupancy)')
    axes_hist[0, 0].set_ylabel('Density')
    axes_hist[0, 0].set_title('ABC with Stochastic Simulator')
    axes_hist[0, 0].legend()
    axes_hist[0, 0].grid(True, alpha=0.3)

    axes_hist[0, 1].hist(abc_surrogate_results['U'], bins=50, alpha=0.7, label='ABC Surrogate', density=True, color='orange')
    axes_hist[0, 1].axvline(U_true, color='red', linestyle='--', linewidth=2, label=f'True U={U_true}')
    axes_hist[0, 1].set_xlabel('U (Initial Occupancy)')
    axes_hist[0, 1].set_ylabel('Density')
    axes_hist[0, 1].set_title('ABC with Surrogate Model')
    axes_hist[0, 1].legend()
    axes_hist[0, 1].grid(True, alpha=0.3)

    axes_hist[0, 2].hist(mcmc_results_burned['U'], bins=50, alpha=0.7, label='MCMC', density=True, color='green')
    axes_hist[0, 2].axvline(U_true, color='red', linestyle='--', linewidth=2, label=f'True U={U_true}')
    axes_hist[0, 2].set_xlabel('U (Initial Occupancy)')
    axes_hist[0, 2].set_ylabel('Density')
    axes_hist[0, 2].set_title('MCMC with Surrogate Model')
    axes_hist[0, 2].legend()
    axes_hist[0, 2].grid(True, alpha=0.3)

    # Row 2: Parameter D histograms
    axes_hist[1, 0].hist(abc_stochastic_results['D'], bins=50, alpha=0.7, density=True)
    axes_hist[1, 0].axvline(D_true, color='red', linestyle='--', linewidth=2, label=f'True D={D_true}')
    axes_hist[1, 0].set_xlabel('D (Diffusion Coefficient)')
    axes_hist[1, 0].set_ylabel('Density')
    axes_hist[1, 0].legend()
    axes_hist[1, 0].grid(True, alpha=0.3)

    axes_hist[1, 1].hist(abc_surrogate_results['D'], bins=50, alpha=0.7, density=True, color='orange')
    axes_hist[1, 1].axvline(D_true, color='red', linestyle='--', linewidth=2, label=f'True D={D_true}')
    axes_hist[1, 1].set_xlabel('D (Diffusion Coefficient)')
    axes_hist[1, 1].set_ylabel('Density')
    axes_hist[1, 1].legend()
    axes_hist[1, 1].grid(True, alpha=0.3)

    axes_hist[1, 2].hist(mcmc_results_burned['D'], bins=50, alpha=0.7, density=True, color='green')
    axes_hist[1, 2].axvline(D_true, color='red', linestyle='--', linewidth=2, label=f'True D={D_true}')
    axes_hist[1, 2].set_xlabel('D (Diffusion Coefficient)')
    axes_hist[1, 2].set_ylabel('Density')
    axes_hist[1, 2].legend()
    axes_hist[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Comparison of ABC and MCMC Inference Results', fontsize=16, y=1.02)

    # Save the plot
    plt.savefig('notebooks/images/abc_mcmc_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    return (mcmc_results_burned,)


@app.cell
def _(
    D_true,
    U_true,
    abc_stochastic_results,
    abc_surrogate_results,
    mcmc_results_burned,
    plt,
):
    """
    Create 2D parameter correlation plots
    """
    fig_corr, axes_corr = plt.subplots(1, 3, figsize=(15, 5))

    # ABC Stochastic
    axes_corr[0].scatter(abc_stochastic_results['U'], abc_stochastic_results['D'],
                   alpha=0.6, s=10, label='ABC Samples')
    axes_corr[0].scatter(U_true, D_true, color='red', s=100, marker='*',
                   label=f'True: U={U_true}, D={D_true}', zorder=5)
    axes_corr[0].set_xlabel('U (Initial Occupancy)')
    axes_corr[0].set_ylabel('D (Diffusion Coefficient)')
    axes_corr[0].set_title('ABC with Stochastic Simulator')
    axes_corr[0].legend()
    axes_corr[0].grid(True, alpha=0.3)

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

    # MCMC
    axes_corr[2].scatter(mcmc_results_burned['U'], mcmc_results_burned['D'],
                   alpha=0.6, s=10, color='green', label='MCMC Samples')
    axes_corr[2].scatter(U_true, D_true, color='red', s=100, marker='*',
                   label=f'True: U={U_true}, D={D_true}', zorder=5)
    axes_corr[2].set_xlabel('U (Initial Occupancy)')
    axes_corr[2].set_ylabel('D (Diffusion Coefficient)')
    axes_corr[2].set_title('MCMC with Surrogate Model')
    axes_corr[2].legend()
    axes_corr[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('2D Parameter Correlations', fontsize=16, y=1.02)

    # Save the plot
    plt.savefig('notebooks/images/parameter_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    return


@app.cell
def _(mcmc_results, plt):
    """
    Create MCMC trace plots
    """
    fig_trace, axes_trace = plt.subplots(3, 1, figsize=(12, 8))

    # U trace
    axes_trace[0].plot(mcmc_results['U'], alpha=0.8, linewidth=0.5)
    axes_trace[0].set_ylabel('U')
    axes_trace[0].set_title('MCMC Trace Plots')
    axes_trace[0].grid(True, alpha=0.3)

    # D trace
    axes_trace[1].plot(mcmc_results['D'], alpha=0.8, linewidth=0.5, color='orange')
    axes_trace[1].set_ylabel('D')
    axes_trace[1].grid(True, alpha=0.3)

    # Sigma trace
    axes_trace[2].plot(mcmc_results['sigma'], alpha=0.8, linewidth=0.5, color='green')
    axes_trace[2].set_ylabel('σ (noise)')
    axes_trace[2].set_xlabel('MCMC Iteration')
    axes_trace[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plt.savefig('notebooks/images/mcmc_traces.png', dpi=300, bbox_inches='tight')
    plt.show()
    return


@app.cell
def _(
    D_true,
    U_true,
    abc_stochastic_results,
    abc_surrogate_results,
    mcmc_results,
    np,
):
    """
    Print summary statistics for all methods
    """
    print("="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    print(f"\n🎯 TRUE PARAMETERS:")
    print(f"   U = {U_true}")
    print(f"   D = {D_true}")

    print(f"\n📊 ABC WITH STOCHASTIC SIMULATOR:")
    print(f"   U: mean={np.mean(abc_stochastic_results['U']):.3f}, std={np.std(abc_stochastic_results['U']):.3f}")
    print(f"   D: mean={np.mean(abc_stochastic_results['D']):.3f}, std={np.std(abc_stochastic_results['D']):.3f}")
    print(f"   Samples: {abc_stochastic_results['n_keep']:,} kept from {abc_stochastic_results['n_samples']:,}")

    print(f"\n📊 ABC WITH SURROGATE MODEL:")
    print(f"   U: mean={np.mean(abc_surrogate_results['U']):.3f}, std={np.std(abc_surrogate_results['U']):.3f}")
    print(f"   D: mean={np.mean(abc_surrogate_results['D']):.3f}, std={np.std(abc_surrogate_results['D']):.3f}")
    print(f"   Samples: {abc_surrogate_results['n_keep']:,} kept from {abc_surrogate_results['n_samples']:,}")

    print(f"\n📊 MCMC WITH SURROGATE MODEL:")
    print(f"   U: mean={np.mean(mcmc_results['U']):.3f}, std={np.std(mcmc_results['U']):.3f}")
    print(f"   D: mean={np.mean(mcmc_results['D']):.3f}, std={np.std(mcmc_results['D']):.3f}")
    print(f"   σ: mean={np.mean(mcmc_results['sigma']):.3f}, std={np.std(mcmc_results['sigma']):.3f}")
    print(f"   Samples: {mcmc_results['n_samples']:,}")

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
