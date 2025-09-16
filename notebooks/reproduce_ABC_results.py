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
        norm,
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
    force_rerun_abc_stochastic = False 

    if os.path.exists(abc_stochastic_file) and not force_rerun_abc_stochastic:
        print("📁 Loading existing ABC stochastic results...")
        with open(abc_stochastic_file, 'rb') as f:
            abc_stochastic_results = pickle.load(f)
        print(f"✅ Loaded {abc_stochastic_results['n_keep']} samples from {abc_stochastic_file}")
    else:
        # Run ABC with stochastic simulator
        abc_stochastic_results = abc_stochastic(observed_column_counts,U_limits,P_limits, n_samples=10000, n_keep=1000)

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
        abc_surrogate_results = abc_surrogate(observed_column_counts,U_limits,P_limits, n_samples=100000, n_keep=1000)

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

    force_mcmc_rerun = False
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
    3. Full corner plots 
    4. MCMC trace plots
    5. Summary statistics
    """
    )
    return


@app.cell
def _(
    D_true,
    P_limits,
    U_limits,
    U_true,
    abc_stochastic_results,
    abc_surrogate_results,
    convert_P_to_D,
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





    for ax_j in axes_hist[0,:]:
        ax_j.set_xlim(U_limits[0],U_limits[1])


    for ax_k in axes_hist[1,:]:
        ax_k.set_xlim(convert_P_to_D(P_limits[0]),convert_P_to_D(P_limits[1])*1.05)



    plt.tight_layout()
    plt.suptitle('Comparison of ABC and MCMC Inference Results', fontsize=16, y=1.02)

    # Save the plot
    plt.savefig('notebooks/images/abc_mcmc_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    return (mcmc_results_burned,)


@app.cell
def _(
    D_true,
    P_limits,
    U_limits,
    U_true,
    abc_stochastic_results,
    abc_surrogate_results,
    convert_P_to_D,
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


    for ax_i in axes_corr:
        ax_i.set_xlim(U_limits[0],U_limits[1])
        ax_i.set_ylim(convert_P_to_D(P_limits[0]),convert_P_to_D(P_limits[1])*1.05)



    #     U_limits = (0.30, 0.70)
    # P_limits = (0.2, 1.0)

    plt.tight_layout()
    plt.suptitle('2D Parameter Correlations', fontsize=16, y=1.02)

    # Save the plot
    plt.savefig('notebooks/images/parameter_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Corner plots


    Lets give the results above as a nice, publication quality corner plot. The above plots are more pedagogicall useful, but we need to tidy up the presentation.
    """
    )
    return


@app.cell
def _(np, plt):

    from pathlib import Path
    import corner 
    import scienceplots
    def create_professional_corner_plot(
        posterior_samples,
        param_names,
        true_parameters,
        nbins=30,
        truth_color='orange',
        savefig_path=None,
        **kwargs
    ):
        """
        Create a professional, publication-quality corner plot.

        Parameters:
        -----------
        posterior_samples : array-like, shape (n_samples, n_params)
            Posterior samples to plot
        param_names : list of str
            Parameter names for axis labels (LaTeX formatting recommended, e.g., [r'$\theta_1$', r'$\theta_2$'])
        true_parameters : array-like
            True parameter values to mark on the plot
        nbins : int, default 30
            Number of bins for histograms
        truth_color : str, default 'orange'
            Color for truth value lines
        savefig_path : str or None, default None
            Path to save the figure. If None, figure is not saved.
            Will save both .png and .pdf versions if path provided.
        **kwargs : dict
            Additional keyword arguments passed to corner.corner()

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The corner plot figure

        Examples:
        ---------
        >>> samples = np.random.randn(1000, 2)
        >>> param_names = [r'$\theta_1$', r'$\theta_2$']
        >>> true_vals = [0.5, -0.3]
        >>> fig = create_professional_corner_plot(samples, param_names, true_vals, 
        ...                                       savefig_path='corner_plot')
        """

        # Set up professional styling
        plt.style.use(['science', 'no-latex'])  # Add 'no-latex' if LaTeX not available
        plt.rcParams.update({
            'font.size': 12,
            'axes.linewidth': 1.2,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'xtick.minor.width': 0.8,
            'ytick.minor.width': 0.8,
            'figure.dpi': 100,
            'savefig.dpi': 300,
        })

        # Convert inputs to numpy arrays
        posterior_samples = np.array(posterior_samples)
        true_parameters = np.array(true_parameters)

        # Professional color scheme
        posterior_color = '#2E86C1'   # Professional blue
        contour_colors = ['#AED6F1', '#5DADE2', '#2E86C1']  # Gradient blues

        # Default corner plot arguments
        corner_defaults = {
            'labels': param_names,
            'truths': true_parameters,
            'truth_color': truth_color,
            'color': posterior_color,
            'show_titles': True,
            'title_kwargs': {
                'fontsize': 14, 
                'fontweight': 'bold',
                'pad': 10
            },
            'label_kwargs': {
                'fontsize': 16, 
                'fontweight': 'bold'
            },
            'title_fmt': '.3f',
            'bins': nbins,
            'quantiles': [0.16, 0.5, 0.84],  # 68% credible intervals
            'plot_density': True,
            'plot_datapoints': True,  # Clean look without individual points
            'fill_contours': True,
            'contour_kwargs': {
                'colors': contour_colors, 
                'linewidths': 0.5
            },
            'hist_kwargs': {
                'alpha': 0.8, 
                'edgecolor': posterior_color, 
                'linewidth': 1.0
            },
            'max_n_ticks': 4,
            'use_math_text': True,
            'truth_kwargs': {
                'color': truth_color,
                'linewidth': 2.5,
                'alpha': 0.8,
                'linestyle': '--'
            }
        }

        # Update defaults with any user-provided kwargs
        corner_defaults.update(kwargs)

        # Create the corner plot
        fig = corner.corner(posterior_samples, **corner_defaults)

        # Post-processing improvements
        axes = fig.get_axes()

        # Enhance axis appearance
        for ax in axes:
            if ax is not None:
                # Improve tick appearance
                ax.tick_params(
                    which='major', 
                    labelsize=12, 
                    width=1.2, 
                    length=6,
                    direction='in',
                    top=True, 
                    right=True
                )
                ax.tick_params(
                    which='minor', 
                    width=0.8, 
                    length=3,
                    direction='in',
                    top=True, 
                    right=True
                )

                # Add minor ticks
                ax.minorticks_on()

                # Add subtle grid
                ax.grid(True, alpha=0.3, linewidth=0.5, linestyle=':')

        # Adjust layout
        plt.tight_layout()

        # Manually adjust layout with reduced spacing if spacing is not to your liking
        plt.subplots_adjust(
            hspace=0.05,  # Reduce vertical spacing between subplots
            wspace=0.05   # Reduce horizontal spacing between subplots
        )

        # Save figure if path provided
        if savefig_path is not None:
            savefig_path = Path(savefig_path)
            # Save PNG version
            png_path = savefig_path.with_suffix('.png')
            fig.savefig(png_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')

            # Save PDF version for publications
            pdf_path = savefig_path.with_suffix('.pdf')
            fig.savefig(pdf_path, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')

            print(f"Figures saved as:\n  - {png_path}\n  - {pdf_path}")

        return fig


    return (create_professional_corner_plot,)


@app.cell
def _(
    D_true,
    P_limits,
    U_limits,
    U_true,
    abc_stochastic_results,
    convert_P_to_D,
    create_professional_corner_plot,
    np,
    plt,
):
    # Set general parameters
    param_names = [r'$U$', r'$D$']
    true_parameters = [U_true, D_true]
    D_limits = (convert_P_to_D(P_limits[0]),convert_P_to_D(P_limits[1]))


    # Create the plot: ABC STOCHASTIC
    posterior_samples_abc_stochastic = np.array([abc_stochastic_results['U'], abc_stochastic_results['D']]).T
    create_professional_corner_plot(
        posterior_samples=posterior_samples_abc_stochastic,
        param_names=param_names,
        true_parameters=true_parameters,
        nbins=25,
        truth_color='orange',
        savefig_path='notebooks/images/corner_plot_abc_stochastic',
        range=(U_limits,D_limits)  # Custom ranges as additional kwarg
    )

    plt.show()
    return D_limits, param_names, true_parameters


@app.cell
def _(
    D_limits,
    U_limits,
    abc_surrogate_results,
    create_professional_corner_plot,
    np,
    param_names,
    plt,
    true_parameters,
):
    # Create the plot: ABC SURROGATE
    posterior_samples_abc_surrogate = np.array([abc_surrogate_results['U'], abc_surrogate_results['D']]).T

    create_professional_corner_plot(
        posterior_samples=posterior_samples_abc_surrogate,
        param_names=param_names,
        true_parameters=true_parameters,
        nbins=25,
        truth_color='orange',
        savefig_path='notebooks/images/corner_plot_abc_surrogate',
        range=(U_limits,D_limits)  # Custom ranges as additional kwarg
    )

    plt.show()
    return


@app.cell
def _(
    D_limits,
    D_true,
    U_limits,
    U_true,
    create_professional_corner_plot,
    mcmc_results_burned,
    np,
    plt,
):







    # Create the plot: MCMC SURROGATE
    posterior_samples_mcmc = np.array([mcmc_results_burned['U'], mcmc_results_burned['D'],mcmc_results_burned['sigma']]).T
    param_names_mcmc = [r'$U$', r'$D$',r'$\sigma$']
    true_parameters_mcmc = [U_true,D_true,None]
    sigma_limits = (1.0,3.0)

    create_professional_corner_plot(
        posterior_samples=posterior_samples_mcmc,
        param_names=param_names_mcmc,
        true_parameters=true_parameters_mcmc,
        nbins=25,
        truth_color='orange',
        savefig_path='notebooks/images/corner_plot_mcmc_surrogate',
        range=(U_limits,D_limits,sigma_limits)  # Custom ranges as additional kwarg
    )

    plt.show()





    return


@app.cell
def _(mo):
    mo.md(r"""### MCMC trace plots""")
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
        ax.scatter(columns, observed_data, c='blue', s=60,
                  label='Observed Data', zorder=10, marker='o', edgecolor='darkblue')

        ax.set_xlabel('Column Index (centered)')
        ax.set_ylabel('Agent Count')
        ax.set_title('Posterior Predictive Checks: Comparison of ABC and MCMC Methods')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    return (plot_prediction_comparison,)


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
def _(norm, np):
    """
    Function to compute envelope bounds for PDE predictions (matching Julia approach)
    Supports both ABC (no noise) and MCMC (with Gaussian noise) cases
    """

    def compute_pde_prediction_envelope(pde_predictions, method_name, sigma_samples=None):
        """
        Compute envelope bounds for PDE-based predictions, matching the Julia approach.
        For MCMC with noise, includes Gaussian noise quantiles.

        Parameters:
        -----------
        pde_predictions : np.ndarray of shape (n_samples, n_columns)
            PDE predictions for each parameter sample
        method_name : str
            Name of the method for reporting
        sigma_samples : np.ndarray, optional
            Noise parameter samples for MCMC case. If provided, will add
            Gaussian noise quantiles to envelope bounds.

        Returns:
        --------
        Dict containing envelope bounds and summary statistics
        """
        print(f"📊 Computing PDE prediction envelope for {method_name}...")

        n_pred, n_columns = pde_predictions.shape

        if sigma_samples is not None:
            # MCMC case with Gaussian noise (matching Julia MCMC approach)
            print(f"   Including Gaussian noise with σ samples (MCMC case)")

            # Ensure sigma_samples matches pde_predictions
            sigma_samples = np.array(sigma_samples)
            if len(sigma_samples) != n_pred:
                # Subsample sigma to match predictions
                indices = np.random.choice(len(sigma_samples), size=n_pred, replace=False)
                sigma_samples = sigma_samples[indices]

            # Initialize bounds with extreme values
            lower_bounds = np.full(n_columns, np.inf)
            upper_bounds = np.full(n_columns, -np.inf)

            # For each sample, compute PDE prediction + noise quantiles
            for i in range(n_pred):
                pde_pred = pde_predictions[i]
                sigma = sigma_samples[i]

                # Compute 2.5% and 97.5% quantiles of N(0, σ)
                # Julia: quantile(Normal(0,σsampled[i]),[0.025,0.975])
                noise_quantiles = norm.ppf([0.025, 0.975], loc=0, scale=sigma)

                # Add noise to PDE prediction
                lower_with_noise = pde_pred + noise_quantiles[0]
                upper_with_noise = pde_pred + noise_quantiles[1]

                # Update envelope bounds (taking min/max across all samples)
                lower_bounds = np.minimum(lower_bounds, lower_with_noise)
                upper_bounds = np.maximum(upper_bounds, upper_with_noise)

            envelope = {
                'lower': lower_bounds,
                'upper': upper_bounds,
            }
        else:
            # ABC case without noise (same as before)
            envelope = {
                'lower': np.min(pde_predictions, axis=0),
                'upper': np.max(pde_predictions, axis=0),
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
                'percentiles': percentiles,
                'includes_noise': sigma_samples is not None
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

        # Plot mean prediction line (red line like Julia: color=:red)
        ax.plot(xxloc, mean_pred, '-', linewidth=4, color='red',
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
    abc_stochastic_results,
    compute_prediction_intervals,
    posterior_predictive_sample,
):
    """
    Generate posterior predictive samples for ABC Stochastic method
    """
    # Generate predictions using ABC Stochastic posterior samples
    abc_stochastic_predictions = posterior_predictive_sample(
        parameter_samples=abc_stochastic_results,
        method_name="ABC Stochastic",
        n_pred_samples=1000,
        random_seed=42
    )

    # Compute prediction intervals
    abc_stochastic_pred_results = compute_prediction_intervals(
        predictions=abc_stochastic_predictions,
        method_name="ABC Stochastic"
    )

    return (abc_stochastic_pred_results,)


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
        n_pred_samples=1000,
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
    compute_prediction_intervals,
    mcmc_results_burned,
    posterior_predictive_sample,
):
    """
    Generate posterior predictive samples for MCMC method (using burned-in samples)
    """
    # Generate predictions using MCMC posterior samples (after burn-in)
    mcmc_predictions = posterior_predictive_sample(
        parameter_samples=mcmc_results_burned,
        method_name="MCMC",
        n_pred_samples=1000,
        random_seed=44
    )

    # Compute prediction intervals
    mcmc_pred_results = compute_prediction_intervals(
        predictions=mcmc_predictions,
        method_name="MCMC"
    )

    return (mcmc_pred_results,)


@app.cell
def _(
    abc_stochastic_pred_results,
    abc_surrogate_pred_results,
    mcmc_pred_results,
    observed_column_counts,
    plot_prediction_comparison,
    plt,
):
    """
    Create comparative visualization of posterior predictive checks
    """
    # Set up method comparison
    prediction_results_list = [
        abc_stochastic_pred_results,
        abc_surrogate_pred_results,
        mcmc_pred_results
    ]
    method_names = ['ABC Stochastic', 'ABC Surrogate', 'MCMC']
    colors = ['green', 'green', 'green']


    for i,p in enumerate(prediction_results_list):

        fig_predictive = plot_prediction_comparison([p],
            observed_data=observed_column_counts,
            method_names=[method_names[i]],
            colors=[colors[i]],
        )
        plt.savefig(f'notebooks/images/posterior_predictive_stochastic_{method_names[i]}.png', dpi=300, bbox_inches='tight')
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
    Julia logic for both ABC and MCMC methods:

    **ABC case (no noise):**
    ```julia
    for i in 1:M
        C(x) = LY*smallU[i]*(erf((h-x)/sqrt(4*smallD[i]*T))+erf((h+x)/sqrt(4*smallD[i]*T)))/2;
        # compute envelope bounds...
    end
    ```

    **MCMC case (with Gaussian noise):**
    ```julia
    for i in 1:M
        C(x) = LY*Usampled[i]*(erf((h-x)/sqrt(4*Dsampled[i]*T))+erf((h+x)/sqrt(4*Dsampled[i]*T)))/2;
        # Add noise quantiles: C(x) + quantile(Normal(0,σ), [0.025,0.975])
    end
    ```

    This approach is much faster since it uses the analytical PDE solution rather than running
    stochastic simulations for each parameter sample.
    """
    )
    return


@app.cell
def _(
    abc_stochastic_results,
    compute_pde_prediction_envelope,
    observed_column_counts,
    plot_pde_prediction_envelope,
    plt,
    posterior_predictive_pde_sample,
):
    """
    Generate PDE-based posterior predictive checks for ABC Stochastic method
    """
    # Generate PDE-based predictions for ABC Stochastic
    abc_stochastic_pde_predictions = posterior_predictive_pde_sample(
        parameter_samples=abc_stochastic_results,
        method_name="ABC Stochastic PDE",
        n_pred_samples=1000
    )

    # Compute envelope bounds (no noise for ABC)
    abc_stochastic_pde_envelope = compute_pde_prediction_envelope(
        pde_predictions=abc_stochastic_pde_predictions,
        method_name="ABC Stochastic PDE",
        sigma_samples=None  # No noise for ABC
    )

    # Create visualization
    fig_abc_stochastic_pde = plot_pde_prediction_envelope(
        envelope_results=abc_stochastic_pde_envelope,
        observed_data=observed_column_counts,
        method_name="ABC Stochastic",
        color='green'
    )
    plt.savefig(f'notebooks/images/posterior_predictive_pde_ABC_stochastic.png', dpi=300, bbox_inches='tight')
    plt.show()

    return (abc_stochastic_pde_envelope,)


@app.cell
def _(
    abc_surrogate_results,
    compute_pde_prediction_envelope,
    observed_column_counts,
    plot_pde_prediction_envelope,
    plt,
    posterior_predictive_pde_sample,
):
    """
    Generate PDE-based posterior predictive checks for ABC Surrogate method
    """
    # Generate PDE-based predictions for ABC Surrogate
    abc_surrogate_pde_predictions = posterior_predictive_pde_sample(
        parameter_samples=abc_surrogate_results,
        method_name="ABC Surrogate PDE",
        n_pred_samples=1000
    )

    # Compute envelope bounds (no noise for ABC)
    abc_surrogate_pde_envelope = compute_pde_prediction_envelope(
        pde_predictions=abc_surrogate_pde_predictions,
        method_name="ABC Surrogate PDE",
        sigma_samples=None  # No noise for ABC
    )

    # Create visualization
    fig_abc_surrogate_pde = plot_pde_prediction_envelope(
        envelope_results=abc_surrogate_pde_envelope,
        observed_data=observed_column_counts,
        method_name="ABC Surrogate",
        color='green'
    )
    plt.savefig(f'notebooks/images/posterior_predictive_pde_ABC_surrogate.png', dpi=300, bbox_inches='tight')

    plt.show()

    return (abc_surrogate_pde_envelope,)


@app.cell
def _(
    compute_pde_prediction_envelope,
    mcmc_results_burned,
    observed_column_counts,
    plot_pde_prediction_envelope,
    plt,
    posterior_predictive_pde_sample,
):
    """
    Generate PDE-based posterior predictive checks for MCMC method (with Gaussian noise)
    """
    # Generate PDE-based predictions for MCMC
    mcmc_pde_predictions = posterior_predictive_pde_sample(
        parameter_samples=mcmc_results_burned,
        method_name="MCMC PDE",
        n_pred_samples=1000
    )

    # Compute envelope bounds WITH Gaussian noise (matching Julia MCMC approach)
    mcmc_pde_envelope = compute_pde_prediction_envelope(
        pde_predictions=mcmc_pde_predictions,
        method_name="MCMC PDE",
        sigma_samples=mcmc_results_burned['sigma']  # Include noise for MCMC
    )

    # Create visualization
    fig_mcmc_pde = plot_pde_prediction_envelope(
        envelope_results=mcmc_pde_envelope,
        observed_data=observed_column_counts,
        method_name="MCMC (with noise)",
        color='green'
    )
    plt.savefig(f'notebooks/images/posterior_predictive_pde_MCMC_surrogate.png', dpi=300, bbox_inches='tight')

    plt.show()

    return (mcmc_pde_envelope,)


@app.cell
def _(
    abc_stochastic_pde_envelope,
    abc_surrogate_pde_envelope,
    mcmc_pde_envelope,
    np,
    observed_column_counts,
):
    """
    Compare PDE-based predictions across all methods
    """
    print("\n" + "="*80)
    print("COMPARISON: PDE-based Posterior Predictive Checks")
    print("="*80)

    print(f"Observed total agents: {np.sum(observed_column_counts)}")
    print()

    methods = [
        ("ABC Stochastic", abc_stochastic_pde_envelope),
        ("ABC Surrogate", abc_surrogate_pde_envelope),
        ("MCMC (with noise)", mcmc_pde_envelope)
    ]

    for name, envelope_results in methods:
        stats = envelope_results['global_stats']
        metadata = envelope_results['metadata']

        print(f"📊 {name.upper()}:")
        print(f"   Envelope mean width: {stats['envelope_mean_width']:.2f}")
        print(f"   Total agents: {stats['total_agents_mean']:.1f} ± {stats['total_agents_std']:.1f}")
        print(f"   Includes noise: {metadata['includes_noise']}")
        print()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
