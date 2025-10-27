import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    mo.md(r"""
    # PDE vs Simulation Comparison Figure

    This notebook creates a publication-quality figure comparing the stochastic random walk simulation
    with the analytical PDE solution. The figure consists of four subplots arranged in a 2×2 grid:

    - **(a)** Initial agent distribution at t=0 on 2D lattice
    - **(b)** Final agent distribution at t=100 on 2D lattice
    - **(c)** Column counts N_i at t=0 (blue dots) with PDE solution Hu(x,t) (red curve)
    - **(d)** Column counts N_i at t=100 (blue dots) with PDE solution Hu(x,t) (red curve)

    This visualization demonstrates how well the continuous PDE approximation matches the discrete stochastic simulation.

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

    # Add src directory to path
    sys.path.append(os.path.join(os.getcwd(), "src"))

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import special
    import scienceplots

    # Import simulator
    from simulator import RandomWalkSimulator

    # Apply science plots styling for publication quality
    plt.style.use(["science", "no-latex"])

    # Set random seed for reproducibility
    np.random.seed(42)

    print("✅ Successfully imported all modules!")
    return RandomWalkSimulator, np, os, plt, special


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1. Simulation Parameters

    We use standard parameters for the lattice random walk:
    - **Lattice**: 200 × 50 (Lx × Ly)
    - **Time steps**: 100 (T)
    - **Initial region**: Central strip with half-width 25
    - **True parameters**: U = 0.5, P = 0.7 (D = 0.175)
    """
    )
    return


@app.cell
def _(RandomWalkSimulator):
    """
    Set up simulation parameters
    """

    # Lattice parameters
    Lx = 200  # Number of columns
    Ly = 50  # Number of rows
    T = 100  # Number of time steps
    initial_region_half_width = 25  # Central region for initial placement

    # Model parameters
    U_true = 0.3  # Initial occupancy probability
    P_true = 0.7  # Movement probability
    D_true = P_true / 4  # Diffusion coefficient (P/4 conversion)

    # Create simulator instance
    simulator = RandomWalkSimulator(Lx, Ly, initial_region_half_width)

    print(f"📐 Lattice size: {Lx} × {Ly}")
    print(f"⏱️  Time steps: {T}")
    print(f"🎯 Parameters: U = {U_true}, P = {P_true}, D = {D_true}")
    return (
        D_true,
        Lx,
        Ly,
        P_true,
        T,
        U_true,
        initial_region_half_width,
        simulator,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. Run Simulation and Collect Data

    We run the simulation to collect:
    - Initial positions at t=0
    - Final positions at t=100
    - Column counts at both time points
    """
    )
    return


@app.cell
def _(P_true, T, U_true, np, simulator):
    """
    Execute simulation and extract data at t=0 and t=100
    """

    print("🎲 Running simulation...")

    # Run full simulation
    column_counts_final, initial_positions, final_positions = simulator.simulate(
        U=U_true, P=P_true, T=T, random_seed=42
    )

    # Get initial column counts from initial positions
    column_counts_initial = simulator.get_column_counts(initial_positions)

    print(f"✅ Simulation completed!")
    print(f"🔢 Initial agents: {len(initial_positions)}")
    print(f"🔢 Final agents: {len(final_positions)}")
    print(f"📊 Initial column counts sum: {np.sum(column_counts_initial)}")
    print(f"📊 Final column counts sum: {np.sum(column_counts_final)}")
    return (
        column_counts_final,
        column_counts_initial,
        final_positions,
        initial_positions,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. PDE Solution

    Implement the analytical PDE solution for the continuous diffusion process:

    $$u(x,t) = U \cdot \frac{1}{2}\left[\text{erf}\left(\frac{h-x}{\sqrt{4Dt}}\right) + \text{erf}\left(\frac{h+x}{\sqrt{4Dt}}\right)\right]$$

    where:
    - $h$ is the initial region half-width
    - $D$ is the diffusion coefficient
    - $U$ is the initial occupancy probability
    - $t$ is the time

    The column counts are obtained by multiplying the density by the lattice height: $N_i = H \cdot u(x_i, t)$ where $H = L_y$.
    """
    )
    return


@app.cell
def _(Ly, np, special):
    """
    Implement the surrogate PDE model
    """


    def surrogate_pde_model(U, D, t, Lx, h):
        """
        Compute the analytical PDE solution for the random walk model

        Parameters:
        -----------
        U : float
            Initial occupancy probability
        D : float
            Diffusion coefficient
        t : float
            Time
        Lx : int
            Lattice width
        h : int
            Initial region half-width

        Returns:
        --------
        column_counts : np.ndarray
            Predicted agent counts per column
        """
        # Create x coordinates (centered at 0)
        x_coords = np.arange(-Lx // 2, Lx // 2)

        # Handle edge cases
        if D <= 0 or t <= 0:
            return np.zeros(len(x_coords))

        sqrt_4Dt = np.sqrt(4 * D * t)

        # PDE solution: u(x,t) = U * 0.5 * [erf((h-x)/sqrt(4Dt)) + erf((h+x)/sqrt(4Dt))]
        term1 = special.erf((h - x_coords) / sqrt_4Dt)
        term2 = special.erf((h + x_coords) / sqrt_4Dt)

        density = U * 0.5 * (term1 + term2)

        # Convert density to counts (multiply by lattice height Ly)
        column_counts = density * Ly

        return np.maximum(column_counts, 0)  # Ensure non-negative


    print("✅ PDE solution function implemented")
    return (surrogate_pde_model,)


@app.cell
def _(
    D_true,
    Lx,
    T,
    U_true,
    initial_region_half_width,
    np,
    surrogate_pde_model,
):
    """
    Compute PDE solutions at t=0 and t=100
    """

    # PDE solution at t=0 (should match initial distribution closely)
    # Use small t to avoid division by zero
    t_initial = 0.01  # Small time to approximate t=0
    pde_counts_initial = surrogate_pde_model(
        U_true, D_true, t_initial, Lx, initial_region_half_width
    )

    # PDE solution at t=100
    pde_counts_final = surrogate_pde_model(U_true, D_true, T, Lx, initial_region_half_width)

    print(f"✅ PDE solutions computed")
    print(f"📈 PDE initial total: {np.sum(pde_counts_initial):.1f}")
    print(f"📈 PDE final total: {np.sum(pde_counts_final):.1f}")
    return pde_counts_final, pde_counts_initial


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. Create Publication-Quality Figure

    Generate a 2×2 subplot figure comparing the simulation and PDE solution at t=0 and t=100.
    """
    )
    return


@app.cell
def _(
    Lx,
    T,
    column_counts_final,
    column_counts_initial,
    final_positions,
    initial_positions,
    np,
    os,
    pde_counts_final,
    pde_counts_initial,
    plt,
):
    """
    Create comprehensive 2x2 comparison figure
    """

    # Set up publication-quality rcParams (matching professional corner plot standards)
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "figure.dpi": 100,
            "savefig.dpi": 300,
        }
    )

    # Create figure with 2x2 subplots
    # Use equal heights for both rows to make top subplots square
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.15, height_ratios=[1, 1])

    # Calculate coordinate system
    x_min = -(Lx // 2)
    x_max = Lx // 2 if Lx % 2 == 1 else (Lx // 2) - 1
    x_positions = np.arange(x_min, x_max + 1)

    # Vibrant professional color scheme (more saturated for visibility)
    color_initial = "#3498DB"  # Vibrant blue (lighter, more saturated)
    color_final = "#E67E22"  # Vibrant orange
    color_truth = "red"  # Standard red for truth lines

    # --- Subplot (a): Initial agent distribution at t=0 ---
    ax_a = fig.add_subplot(gs[0, 0])
    if initial_positions:
        x_coords_init, y_coords_init = zip(*initial_positions)
        ax_a.scatter(
            x_coords_init,
            y_coords_init,
            c=color_initial,
            s=15,
            alpha=0.6,
            edgecolors="#1B4F72",
            linewidth=0.3,
        )

    t_init_print = 0.0
    ax_a.set_xlim(x_min - 0.5, x_max + 0.5)
    # ax_a.set_ylim(25 - 100, 25 + 100)
    ax_a.set_ylim(0, 50)
    ax_a.set_xlabel("x", fontsize=12, fontweight="bold")
    ax_a.set_ylabel("y", fontsize=12, fontweight="bold")
    ax_a.set_title(
        f"(a) Initial Distribution (t={t_init_print:.2f})",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax_a.axvline(x=0, color=color_truth, linestyle="--", alpha=0.6, linewidth=2.0)
    ax_a.grid(True, alpha=0.3, linewidth=0.5, linestyle=":")
    ax_a.tick_params(
        which="major",
        labelsize=10,
        width=1.2,
        length=6,
        direction="in",
        top=True,
        right=True,
    )
    ax_a.tick_params(
        which="minor", width=0.8, length=3, direction="in", top=True, right=True
    )
    ax_a.minorticks_on()


    # --- Subplot (b): Final agent distribution at t=100 ---
    ax_b = fig.add_subplot(gs[0, 1])
    if final_positions:
        x_coords_final, y_coords_final = zip(*final_positions)
        ax_b.scatter(
            x_coords_final,
            y_coords_final,
            c=color_final,
            s=15,
            alpha=0.6,
            edgecolors="#BA4A00",
            linewidth=0.3,
        )

    ax_b.set_xlim(x_min - 0.5, x_max + 0.5)
    # ax_b.set_ylim(25 - 100, 25 + 100)
    ax_b.set_ylim(0, 50)

    ax_b.set_xlabel("x", fontsize=12, fontweight="bold")
    ax_b.set_ylabel("y", fontsize=12, fontweight="bold")
    ax_b.set_title(
        f"(b) Final Distribution (t={T})", fontsize=13, fontweight="bold", pad=10
    )
    ax_b.axvline(x=0, color=color_truth, linestyle="--", alpha=0.6, linewidth=2.0)
    ax_b.grid(True, alpha=0.3, linewidth=0.5, linestyle=":")
    ax_b.tick_params(
        which="major",
        labelsize=10,
        width=1.2,
        length=6,
        direction="in",
        top=True,
        right=True,
    )
    ax_b.tick_params(
        which="minor", width=0.8, length=3, direction="in", top=True, right=True
    )
    ax_b.minorticks_on()

    # --- Subplot (c): Column counts at t=0 with PDE solution ---
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.scatter(
        x_positions,
        column_counts_initial,
        color=color_initial,
        s=50,
        alpha=0.7,
        edgecolors="#1B4F72",
        linewidth=1.0,
        label="Simulation data",
        zorder=3,
    )
    ax_c.plot(
        x_positions,
        pde_counts_initial,
        color=color_truth,
        linewidth=2.5,
        label="PDE solution $Hu(x,t)$",
        alpha=0.8,
        zorder=2,
    )

    ax_c.set_xlabel("x", fontsize=12, fontweight="bold")
    ax_c.set_ylabel("$N_i$, $Hu(x,t)$", fontsize=12, fontweight="bold")
    ax_c.set_title(
        f"(c) Column Counts (t={t_init_print:.2f})", fontsize=13, fontweight="bold", pad=10
    )
    ax_c.axvline(x=0, color=color_truth, linestyle="--", alpha=0.6, linewidth=2.0)
    ax_c.grid(True, alpha=0.3, linewidth=0.5, linestyle=":")
    ax_c.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax_c.tick_params(
        which="major",
        labelsize=10,
        width=1.2,
        length=6,
        direction="in",
        top=True,
        right=True,
    )
    ax_c.tick_params(
        which="minor", width=0.8, length=3, direction="in", top=True, right=True
    )
    ax_c.minorticks_on()


    # --- Subplot (d): Column counts at t=100 with PDE solution ---
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.scatter(
        x_positions,
        column_counts_final,
        color=color_final,
        s=50,
        alpha=0.7,
        edgecolors="#BA4A00",
        linewidth=1.0,
        label="Simulation data",
        zorder=3,
    )
    ax_d.plot(
        x_positions,
        pde_counts_final,
        color=color_truth,
        linewidth=2.5,
        label="PDE solution $Hu(x,t)$",
        alpha=0.8,
        zorder=2,
    )

    ax_d.set_xlabel("x", fontsize=12, fontweight="bold")
    ax_d.set_ylabel("$N_i$, $Hu(x,t)$", fontsize=12, fontweight="bold")
    ax_d.set_title(f"(d) Column Counts (t={T})", fontsize=13, fontweight="bold", pad=10)
    ax_d.axvline(x=0, color=color_truth, linestyle="--", alpha=0.6, linewidth=2.0)
    ax_d.grid(True, alpha=0.3, linewidth=0.5, linestyle=":")
    ax_d.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax_d.tick_params(
        which="major",
        labelsize=10,
        width=1.2,
        length=6,
        direction="in",
        top=True,
        right=True,
    )
    ax_d.tick_params(
        which="minor", width=0.8, length=3, direction="in", top=True, right=True
    )
    ax_d.minorticks_on()
    ax_d.set_xlim(-100, 100)


    # # --- PEDAGOGICAL ENHANCEMENTS ---
    # import matplotlib.patches as patches
    # color_highlight = 'yellow'
    # # Choose an x-position to highlight for demonstration.
    # x_highlight = 25
    # highlight_width = 1.0

    # # Find the corresponding data point in the 1D plot
    # highlight_idx_initial = np.where(x_positions == x_highlight)[0][0]
    # y_highlight_initial = column_counts_initial[highlight_idx_initial]

    # highlight_idx_final = np.where(x_positions == x_highlight)[0][0]
    # y_highlight_final = column_counts_final[highlight_idx_final]

    # # Add a shaded vertical region to the 2D plots
    # ax_a.axvspan(
    #     x_highlight - highlight_width / 2,
    #     x_highlight + highlight_width / 2,
    #     color=color_highlight,
    #     alpha=0.2,
    #     zorder=0,
    # )
    # ax_b.axvspan(
    #     x_highlight - highlight_width / 2,
    #     x_highlight + highlight_width / 2,
    #     color=color_highlight,
    #     alpha=0.2,
    #     zorder=0,
    # )

    # # Highlight the corresponding points in the 1D plots
    # ax_c.scatter(
    #     x_highlight,
    #     y_highlight_initial,
    #     color=color_highlight,
    #     s=150,
    #     zorder=4,
    #     edgecolor="black",
    #     linewidth=1.5,
    #     marker="*",
    # )
    # ax_d.scatter(
    #     x_highlight,
    #     y_highlight_final,
    #     color=color_highlight,
    #     s=150,
    #     zorder=4,
    #     edgecolor="black",
    #     linewidth=1.5,
    #     marker="*",
    # )

    # y_min = 25-100
    # y_center = 25
    # y_max = 25+100
    # # Create connection lines between the plots
    # # Connection for the initial state (left side)
    # con_initial_1 = patches.ConnectionPatch(
    #     xyA=(x_highlight - highlight_width / 2, y_min),
    #     coordsA=ax_a.transData,
    #     xyB=(x_highlight, y_highlight_initial),
    #     coordsB=ax_c.transData,
    #     color=color_highlight,
    #     linestyle="--",
    #     linewidth=1.5,
    #     alpha=0.7,
    # )
    # con_initial_2 = patches.ConnectionPatch(
    #     xyA=(x_highlight + highlight_width / 2, y_min),
    #     coordsA=ax_a.transData,
    #     xyB=(x_highlight, y_highlight_initial),
    #     coordsB=ax_c.transData,
    #     color=color_highlight,
    #     linestyle="--",
    #     linewidth=1.5,
    #     alpha=0.7,
    # )
    # fig.add_artist(con_initial_1)
    # fig.add_artist(con_initial_2)

    # # Add an explanatory annotation
    # ax_a.annotate(
    #     "Sum over all y",
    #     xy=(x_highlight, y_center + 40),
    #     xytext=(x_highlight + 15, y_center + 60),
    #     fontsize=11,
    #     fontweight="bold",
    #     color=color_highlight,
    #     arrowprops=dict(
    #         arrowstyle="->",
    #         connectionstyle="arc3,rad=0.2",
    #         color=color_highlight,
    #         linewidth=1.5,
    #     ),
    # )
    # ax_a.annotate(
    #     r'$N_i = \sum_j n_{i,j}$',
    #     xy=(x_highlight, y_highlight_initial),
    #     xycoords=ax_c.transData, # Use the coordinate system of the bottom plot
    #     xytext=(x_highlight + 15, y_highlight_initial + 50),
    #     textcoords=ax_c.transData,
    #     fontsize=14,
    #     color=color_highlight,
    # )


    # # Connection for the final state (right side)
    # con_final_1 = patches.ConnectionPatch(
    #     xyA=(x_highlight - highlight_width / 2, y_min),
    #     coordsA=ax_b.transData,
    #     xyB=(x_highlight, y_highlight_final),
    #     coordsB=ax_d.transData,
    #     color=color_highlight,
    #     linestyle="--",
    #     linewidth=1.5,
    #     alpha=0.7,
    # )
    # con_final_2 = patches.ConnectionPatch(
    #     xyA=(x_highlight + highlight_width / 2, y_min),
    #     coordsA=ax_b.transData,
    #     xyB=(x_highlight, y_highlight_final),
    #     coordsB=ax_d.transData,
    #     color=color_highlight,
    #     linestyle="--",
    #     linewidth=1.5,
    #     alpha=0.7,
    # )
    # fig.add_artist(con_final_1)
    # fig.add_artist(con_final_2)


    # Overall figure title
    # Save the figure
    output_dir = "docs/paper"
    os.makedirs(output_dir, exist_ok=True)

    png_path = f"{output_dir}/pde_comparison_figure.png"
    pdf_path = f"{output_dir}/pde_comparison_figure.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", edgecolor="none")

    print(f"✅ Figure saved:")
    print(f"   📄 {png_path}")
    print(f"   📄 {pdf_path}")

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary

    This notebook successfully created a publication-quality comparison figure showing:

    1. **Spatial distributions**: Visual comparison of agent positions at initial and final times
    2. **Column count matching**: How well the PDE solution approximates the discrete simulation
    3. **Temporal evolution**: Spread of agents from initial concentrated distribution to final diffused state

    The figure demonstrates that the continuous PDE approximation provides an excellent match to the
    discrete stochastic simulation, validating the use of the analytical solution for fast inference methods.
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
