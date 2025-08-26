import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md(
        r"""
        # 2D Random Walk Lattice Simulator

        This notebook demonstrates the **2D lattice random walk simulator** for biological population models, 
        as described in [Simpson & Planck](https://www.biorxiv.org/content/10.1101/2025.05.25.656057v4). The simulator implements a discrete random walk model where agents 
        move on a square lattice with zero-flux boundary conditions.

        ## Key Features:
        - **Centered coordinate system**: x ∈ [-Lx/2, Lx/2], y ∈ [0, Ly-1]
        - **Initial placement**: Agents placed with probability U in central region around x=0
        - **Random sequential updates**: Each time step, Q agents are selected (with replacement) to potentially move
        - **Movement probability**: Selected agents move with probability P to a random neighboring site
        - **Zero-flux boundaries**: Agents cannot move outside the lattice

        ## Parameters:
        - **U**: Initial occupancy probability (0 < U ≤ 1)
        - **P**: Movement probability (0 ≤ P ≤ 1)
        - **T**: Number of time steps
        """
    )
    return (mo,)


@app.cell
def _():
    """
    Import required modules and set up the simulator
    """
    import sys
    import os
    import time
    import psutil
    from collections import defaultdict

    # Add src directory to path
    sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))

    import numpy as np
    import matplotlib.pyplot as plt
    from simulator import RandomWalkSimulator, plot_simulation_comparison, plot_lattice, plot_column_counts

    # Set random seed for reproducibility
    np.random.seed(42)

    print("✅ Successfully imported simulator modules and timing utilities!")
    return (
        RandomWalkSimulator,
        np,
        plot_column_counts,
        plot_lattice,
        plot_simulation_comparison,
        plt,
        time,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Lattice Configuration

    We'll create a 2D lattice where agents start in a central region and can spread outward over time.
    The **centered coordinate system** means x=0 is at the center, with negative x values to the left 
    and positive x values to the right.
    """
    )
    return


@app.cell
def _(RandomWalkSimulator):
    """
    Initialize the simulator with lattice parameters
    """
    # Lattice parameters
    Lx = 200  # Number of columns
    Ly = 50  # Number of rows
    initial_region_half_width = 25  # Central region for initial placement

    # Create simulator instance
    simulator = RandomWalkSimulator(Lx, Ly, initial_region_half_width)

    print(f"✅ Created simulator with lattice size {Lx} x {Ly}")
    print(f"📍 Coordinate system: x ∈ [{-(Lx//2)}, {Lx//2 if Lx%2==1 else Lx//2-1}], y ∈ [0, {Ly-1}]")
    print(f"🎯 Initial region: x ∈ [{-initial_region_half_width}, {initial_region_half_width}]")

    return Lx, Ly, simulator


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Simulation Parameters

    Now we'll set the key parameters for our random walk simulation:
    - **U = 0.3**: Each site in the initial region has a 30% chance of containing an agent at t=0
    - **P = 0.5**: When selected, each agent has a 50% chance of moving to a neighboring site
    - **T = 100**: We'll run the simulation for 100 time steps
    """
    )
    return


@app.cell
def _():
    """
    Set simulation parameters
    """
    # Model parameters
    U = 0.5  # Initial occupancy probability
    P = 1.0  # Movement probability
    T = 100  # Number of time steps

    print(f"🎲 Simulation parameters:")
    print(f"   U (occupancy probability): {U}")
    print(f"   P (movement probability): {P}")
    print(f"   T (time steps): {T}")

    return P, T, U


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Running the Simulation

    Time to execute the random walk! The simulation will:
    1. Initialize agents randomly in the central region based on probability U
    2. For each time step, select Q agents (with replacement) to potentially move
    3. Each selected agent moves with probability P to a random neighboring site
    4. Record the final distribution of agents across columns
    """
    )
    return


@app.cell
def _(P, T, U, np, simulator):
    """
    Run the simulation
    """
    print("🏃 Running simulation...")

    # Run simulation with fixed seed
    column_counts, initial_positions, final_positions = simulator.simulate(
        U=U, P=P, T=T, random_seed=123
    )

    print(f"✅ Simulation completed!")
    print(f"🔢 Initial agents: {len(initial_positions)}")
    print(f"🔢 Final agents: {len(final_positions)}")
    print(f"🔢 Total agents in columns: {np.sum(column_counts)}")
    print(f"✅ Agent conservation: {'Yes' if len(initial_positions) == len(final_positions) == np.sum(column_counts) else 'No'}")

    return column_counts, final_positions, initial_positions


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Visualization: Before and After

    Let's visualize the complete simulation results! The plots show:
    - **Left**: Initial agent distribution (concentrated in central region)
    - **Middle**: Final agent distribution (spread due to random walks) 
    - **Right**: Column counts (the observation vector used for NPE)

    The red dashed line marks x=0 (center of the lattice).
    """
    )
    return


@app.cell
def _(
    Lx,
    Ly,
    P,
    T,
    U,
    column_counts,
    final_positions,
    initial_positions,
    plot_simulation_comparison,
):
    """
    Create comprehensive visualization
    """
    # Create main comparison plot
    fig_comparison = plot_simulation_comparison(
        initial_positions=initial_positions,
        final_positions=final_positions,
        column_counts=column_counts,
        Lx=Lx,
        Ly=Ly,
        U=U,
        P=P,
        T=T,
        figsize=(18, 6)
    )
    fig_comparison.suptitle('Random Walk Simulation Results', fontsize=16, y=1.02)

    return


@app.cell
def _(
    Lx,
    Ly,
    T,
    U,
    column_counts,
    final_positions,
    initial_positions,
    plot_simulation_comparison,
):
    """
    What if we change the P value?
    """
    # Create main comparison plot
    fig_comparison_P07 = plot_simulation_comparison(
        initial_positions=initial_positions,
        final_positions=final_positions,
        column_counts=column_counts,
        Lx=Lx,
        Ly=Ly,
        U=U,
        P=0.7,
        T=T,
        figsize=(18, 6)
    )
    fig_comparison_P07.suptitle('Random Walk Simulation Results', fontsize=16, y=1.02)

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Detailed Analysis: Column Distribution

    The column count distribution is the key **observation vector** used for Neural Posterior Estimation.
    This vector captures how agents have spread across the lattice columns after T time steps.
    """
    )
    return


@app.cell
def _(Lx, column_counts, np, plot_column_counts, plt):
    """
    Detailed column count analysis
    """
    # Plot column counts separately for better detail
    fig_counts = plot_column_counts(column_counts, Lx, title="Detailed Column Count Distribution")
    plt.show()

    # Calculate center of mass in centered coordinates
    x_min = -(Lx // 2)
    x_positions = np.arange(x_min, x_min + len(column_counts))
    center_of_mass = np.average(x_positions, weights=column_counts) if np.sum(column_counts) > 0 else 0

    # Print some statistics
    print(f"📊 Column count statistics:")
    print(f"   Mean agents per column: {np.mean(column_counts):.2f}")
    print(f"   Standard deviation: {np.std(column_counts):.2f}")
    print(f"   Min/Max agents: {np.min(column_counts)} / {np.max(column_counts)}")
    print(f"   Center of mass: x = {center_of_mass:.2f}")
    print(f"   Spread from center: {np.max(np.abs(x_positions[column_counts > 0])):.0f} columns")

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Individual Lattice Views

    Here are the initial and final lattice states shown separately for detailed examination.
    Notice how the agents start concentrated around x=0 and spread outward due to the random walks.
    """
    )
    return


@app.cell
def _(Lx, Ly, final_positions, initial_positions, plot_lattice, plt):
    """
    Individual lattice visualizations
    """
    # Plot initial and final states separately
    fig_initial = plot_lattice(initial_positions, Lx, Ly, "Initial Agent Positions")
    plt.show()

    fig_final = plot_lattice(final_positions, Lx, Ly, "Final Agent Positions")
    plt.show()

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Testing & Validation

    Let's verify that our simulator properly validates parameters and handles edge cases correctly.
    """
    )
    return


@app.cell
def _(simulator):
    """
    Test parameter validation and error handling
    """
    print("🧪 Testing parameter validation...")

    try:
        # Test invalid U
        simulator.simulate(U=1.5, P=0.5, T=10)
        print("❌ ERROR: Should have caught invalid U")
    except ValueError as e:
        print(f"✅ Caught invalid U: {e}")

    try:
        # Test invalid P
        simulator.simulate(U=0.5, P=1.5, T=10)
        print("❌ ERROR: Should have caught invalid P")
    except ValueError as e:
        print(f"✅ Caught invalid P: {e}")

    try:
        # Test invalid T
        simulator.simulate(U=0.5, P=0.5, T=-1)
        print("❌ ERROR: Should have caught invalid T")
    except ValueError as e:
        print(f"✅ Caught invalid T: {e}")

    print("🎉 Parameter validation tests passed!")
    return


@app.cell
def _(P, T, U, np, simulator):
    """
    Test reproducibility with random seeds
    """
    print("🔄 Testing reproducibility...")

    # Run same simulation twice with same seed
    result1 = simulator.simulate(U=U, P=P, T=T, random_seed=999)
    result2 = simulator.simulate(U=U, P=P, T=T, random_seed=999)

    # Check if results are identical
    counts_match = np.array_equal(result1[0], result2[0])
    initial_match = result1[1] == result2[1]
    final_match = result1[2] == result2[2]

    print(f"   Column counts match: {'✅' if counts_match else '❌'}")
    print(f"   Initial positions match: {'✅' if initial_match else '❌'}")
    print(f"   Final positions match: {'✅' if final_match else '❌'}")

    if counts_match and initial_match and final_match:
        print("🎉 Reproducibility test passed!")
    else:
        print("❌ Reproducibility test failed!")

    return


if __name__ == "__main__":
    app.run()
