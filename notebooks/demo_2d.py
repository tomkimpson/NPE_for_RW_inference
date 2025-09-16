import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    mo.md(r"""

    # 2D Neural Posterior Estimation for Lattice Random Walk Model

    This notebook demonstrates the use of **2D Neural Posterior Estimation with CNN processing** for parameter estimation of a lattice random walk model. This extends the work of [Simpson & Planck 2025](https://www.biorxiv.org/content/10.1101/2025.05.25.656057v4) by using **full 2D spatial data** instead of compressed 1D column counts.

    ## 🚀 Key Innovation: 2D Spatial CNN Processing

    **Traditional Approach (1D)**: Compress 2D spatial data → 1D column counts → Standard NPE
    **Our Approach (2D)**: Preserve 2D spatial data → CNN feature extraction → Enhanced NPE

    The notebook is organised as follows:

    1. **Demonstrate the 2D simulator** - Full spatial grid output
    2. **Compare 1D vs 2D data formats** - Information preservation analysis
    3. **Show CNN architecture** - Spatial feature extraction
    4. **Demonstrate 2D NPE workflow** - End-to-end spatial inference

    Note that this notebook calls modules from `src/` with the new 2D functionality.

    ---""")
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1. 2D Simulator: Full Spatial Information

    Our enhanced simulator now supports **two output modes**:

    - **1D Mode** (classic): Returns column counts by summing across rows - compresses spatial structure
    - **2D Mode** (new): Returns full spatial grid - preserves complete spatial information

    ### Key Advantages of 2D Mode:
    - **No Information Loss**: Complete spatial structure preserved
    - **Spatial Patterns**: CNN can learn correlations and gradients
    - **Realistic Modeling**: Matches natural 2D biological processes
    - **Enhanced Inference**: More data for parameter estimation

    ---
    """
    )
    return


@app.cell
def _():
    """
    Import required modules including new 2D functionality
    """
    import sys
    import os
    import time
    from collections import defaultdict

    # Add src directory to path
    sys.path.append(os.path.join(os.getcwd(), 'src'))

    import numpy as np
    import matplotlib.pyplot as plt
    from simulator import (RandomWalkSimulator, plot_simulation_comparison,
                          plot_column_counts, plot_2d_grid, plot_2d_comparison)

    # Set random seed for reproducibility
    np.random.seed(42)

    print("✅ Successfully imported 2D simulator modules!")
    print("📊 Available visualization functions:")
    print("   - plot_column_counts() [1D visualization]")
    print("   - plot_2d_grid() [2D spatial visualization]")
    print("   - plot_2d_comparison() [2D before/after comparison]")
    return RandomWalkSimulator, np, plt


@app.cell
def _(RandomWalkSimulator):
    """
    Initialize the simulator for 2D demonstration
    """
    # Lattice parameters optimized for 2D visualization
    Lx = 80   # Number of columns (manageable for visualization)
    Ly = 40   # Number of rows
    initial_region_half_width = 10  # Central region for initial placement

    # Create simulator instance
    simulator = RandomWalkSimulator(Lx, Ly, initial_region_half_width)

    print(f"✅ Created 2D-capable simulator:")
    print(f"   Lattice size: {Lx} × {Ly} = {Lx*Ly} sites")
    print(f"   Coordinate system: x ∈ [{-(Lx//2)}, {Lx//2 if Lx%2==1 else Lx//2-1}], y ∈ [0, {Ly-1}]")
    print(f"   Initial region: x ∈ [{-initial_region_half_width}, {initial_region_half_width}]")
    print(f"   Data modes: 1D (column counts) + 2D (spatial grids)")
    return Lx, Ly, simulator


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Simulation Parameters

    We'll use the same parameters as the 1D demo for direct comparison:

    - **U = 0.3**: Each site in the initial region has a 30% chance of containing an agent at t=0
    - **P = 0.7**: When selected, each agent has a 70% chance of moving to a neighboring site
    - **T = 100**: We'll run the simulation for 100 time steps

    The key difference: we'll generate **both 1D and 2D outputs** from the same simulation.
    """
    )
    return


@app.cell
def _():
    """
    Set simulation parameters (identical to 1D demo)
    """
    # Model parameters
    U = 0.3  # Initial occupancy probability
    P = 0.7  # Movement probability
    T = 100  # Number of time steps

    print(f"🎲 Simulation parameters:")
    print(f"   U (occupancy probability): {U}")
    print(f"   P (movement probability): {P}")
    print(f"   T (time steps): {T}")
    print(f"   Output modes: Both 1D and 2D")
    return P, T, U


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Running Both 1D and 2D Simulations

    We'll run the **same underlying simulation** but extract data in both formats:

    1. **1D Output**: Traditional column counts (compressed spatial data)
    2. **2D Output**: Full spatial grid (complete spatial information)

    This allows us to directly compare information content and visualization capabilities.
    """
    )
    return


@app.cell
def _(P, T, U, np, simulator):
    """
    Run simulation in both 1D and 2D modes
    """
    print("🏃 Running simulation in BOTH modes...")

    # Run simulation in 1D mode (traditional)
    print("\n📊 1D Mode (Traditional):")
    observation_1d, initial_positions, final_positions = simulator.simulate(
        U=U, P=P, T=T, random_seed=123, use_2d_output=False
    )

    # Run simulation in 2D mode (new) - same seed for comparison
    print("📊 2D Mode (Enhanced):")
    observation_2d, _, _ = simulator.simulate(
        U=U, P=P, T=T, random_seed=123, use_2d_output=True
    )

    print(f"\n✅ Simulations completed!")
    print(f"🔢 Agent count: {len(final_positions)} agents")
    print(f"📏 1D output shape: {observation_1d.shape} (column counts)")
    print(f"📐 2D output shape: {observation_2d.shape} (spatial grid)")

    # Verify consistency
    column_sums_from_2d = np.sum(observation_2d, axis=0)
    consistency_check = np.array_equal(observation_1d, column_sums_from_2d)
    print(f"✅ Data consistency: {'PASSED' if consistency_check else 'FAILED'}")

    return observation_1d, observation_2d, initial_positions, final_positions, column_sums_from_2d


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Visualization: 1D vs 2D Data Comparison

    Now let's visualize the difference between 1D and 2D data representations:

    - **Traditional 1D**: Column counts only - spatial structure within columns is lost
    - **Enhanced 2D**: Full spatial grid - complete spatial information preserved

    The 2D visualization reveals spatial patterns invisible in the 1D representation!
    """
    )
    return


@app.cell
def _(Lx, Ly, observation_1d, observation_2d, plot_column_counts, plot_2d_grid, plt):
    """
    Create side-by-side comparison of 1D vs 2D visualization
    """
    print("📈 Creating 1D vs 2D comparison visualization...")

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1D visualization (traditional)
    ax1.bar(range(-(Lx//2), Lx//2 + 1), observation_1d, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Column (x, centered coordinates)')
    ax1.set_ylabel('Number of agents')
    ax1.set_title(f'1D Traditional: Column Counts Only\n(Total: {np.sum(observation_1d)} agents)')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.grid(True, alpha=0.3)

    # 2D visualization (enhanced)
    x_min = -(Lx // 2)
    x_max = Lx // 2 if Lx % 2 == 1 else (Lx // 2) - 1
    im = ax2.imshow(observation_2d, cmap='Blues', origin='lower', aspect='equal',
                   extent=[x_min, x_max+1, 0, Ly], interpolation='nearest')
    ax2.set_xlabel('x (centered coordinates)')
    ax2.set_ylabel('y (row)')
    ax2.set_title(f'2D Enhanced: Full Spatial Grid\n(Total: {np.sum(observation_2d)} agents)')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.colorbar(im, ax=ax2, label='Agents per site')

    plt.tight_layout()
    plt.savefig('notebooks/images/1d_vs_2d_comparison.png', dpi=150, bbox_inches='tight')

    print("✅ Comparison plot created!")
    print("💡 Notice how 2D preserves spatial structure lost in 1D compression")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2D Spatial Evolution Visualization

    Let's create a comprehensive before/after comparison showing the **full spatial evolution**:

    - **Initial State**: Agents concentrated in central region (high spatial structure)
    - **Final State**: Agents spread by random walk (evolved spatial patterns)
    - **Spatial Patterns**: Gradients and correlations visible only in 2D
    """
    )
    return


@app.cell
def _(Lx, Ly, P, T, U, initial_positions, observation_2d, plot_2d_comparison, simulator):
    """
    Create 2D spatial evolution visualization
    """
    print("🎬 Creating 2D spatial evolution visualization...")

    # Generate 2D grid for initial state
    initial_grid = simulator.get_2d_grid(initial_positions)
    final_grid = observation_2d

    # Create comprehensive 2D comparison
    fig_2d = plot_2d_comparison(
        initial_grid=initial_grid,
        final_grid=final_grid,
        Lx=Lx,
        Ly=Ly,
        U=U,
        P=P,
        T=T,
        figsize=(16, 6)
    )

    fig_2d.suptitle('2D Spatial Evolution: Full Information Preserved', fontsize=16, y=1.02)
    plt.savefig('notebooks/images/2d_spatial_evolution.png', dpi=150, bbox_inches='tight')

    print("✅ 2D evolution visualization created!")
    print(f"📊 Spatial information: {initial_grid.shape[0] * initial_grid.shape[1]} data points vs {len(observation_1d)} in 1D")
    return initial_grid, final_grid


@app.cell
def _(mo):
    mo.md(
        r"""
    ## CNN Architecture for 2D Processing

    The **key innovation** of our 2D approach is using **Convolutional Neural Networks** to process spatial data:

    ### CNN Architecture Features:
    - **Spatial Convolutions**: Extract local spatial patterns and correlations
    - **Adaptive Pooling**: Handle variable lattice sizes gracefully
    - **Feature Learning**: Automatically discover relevant spatial features
    - **Translation Invariance**: Robust to spatial shifts in patterns

    ### Integration with NPE:
    1. **2D Input**: Spatial grid (Ly × Lx) instead of column vector (Lx)
    2. **CNN Embedding**: Extract spatial features → dense representation
    3. **Neural Posterior**: Standard normalizing flow on CNN features
    4. **Enhanced Inference**: More informative features → better parameter estimation
    """
    )
    return


@app.cell
def _():
    """
    Demonstrate CNN architecture (without training)
    """
    try:
        from cnn_utils import SpatialCNN, create_spatial_embedding_net

        print("🧠 CNN Architecture for 2D Spatial Processing:")
        print("=" * 50)

        # Create CNN for our lattice size
        cnn = SpatialCNN(input_height=Ly, input_width=Lx, output_dim=128)

        print(f"📥 Input: 2D spatial grid ({Ly} × {Lx})")
        print(f"🔄 Processing:")
        print(f"   - Conv2D layers: Extract spatial features")
        print(f"   - BatchNorm: Stabilize training")
        print(f"   - Adaptive pooling: Handle size variations")
        print(f"   - Dense layers: Feature compression")
        print(f"📤 Output: 128-dimensional feature vector")
        print(f"")
        print(f"🔗 Integration: CNN features → Normalizing Flow → Posterior")
        print(f"🎯 Advantage: Spatial patterns → Better parameter inference")

        # Show embedding network
        embedding_net = create_spatial_embedding_net(Ly, Lx, output_dim=128)
        print(f"✅ Spatial embedding network created successfully!")

    except ImportError as e:
        print(f"⚠️  CNN modules not available (expected without PyTorch): {e}")
        print(f"💡 In full environment: CNN processes {Ly}×{Lx} spatial grids → 128D features")

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2D NPE Workflow Demonstration

    ### Complete 2D Workflow:

    1. **Data Generation**: Simulate with `use_2d_output=True` → full spatial grids
    2. **CNN Processing**: Extract spatial features from 2D grids
    3. **Neural Training**: Train normalizing flow on spatial features
    4. **Inference**: Estimate parameters from 2D spatial observations
    5. **Validation**: Posterior predictive checks with 2D outputs

    ### Command Line Usage:
    ```bash
    # Standard 2D NPE run
    python src/main.py --use_2d_data --use_snpe \\
        --Lx 80 --Ly 40 --T 100 \\
        --snpe_rounds 8 --samples_per_round 5000

    # Or use Slurm script
    sbatch slurm/run_main_2d.sh
    ```

    ### Expected Benefits:
    - **Higher Accuracy**: More informative spatial data
    - **Better Calibration**: CNN learns relevant spatial patterns
    - **Richer Analysis**: Spatial uncertainty quantification
    - **Scientific Insight**: Spatial pattern interpretation
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Information Content Analysis

    Let's quantify the **information difference** between 1D and 2D approaches:
    """
    )
    return


@app.cell
def _(Lx, Ly, np, observation_1d, observation_2d):
    """
    Analyze information content: 1D vs 2D
    """
    print("📊 Information Content Analysis:")
    print("=" * 40)

    # Data size comparison
    size_1d = observation_1d.size
    size_2d = observation_2d.size
    size_ratio = size_2d / size_1d

    print(f"📏 Data Dimensions:")
    print(f"   1D (column counts): {observation_1d.shape} = {size_1d} values")
    print(f"   2D (spatial grid):  {observation_2d.shape} = {size_2d} values")
    print(f"   Information ratio:  {size_ratio:.1f}x more data in 2D")
    print()

    # Information loss in 1D compression
    total_agents = np.sum(observation_2d)
    spatial_variance = np.var(observation_2d)
    column_variance = np.var(observation_1d)

    print(f"📈 Spatial Structure:")
    print(f"   Total agents: {total_agents}")
    print(f"   2D spatial variance: {spatial_variance:.2f}")
    print(f"   1D column variance:  {column_variance:.2f}")
    print(f"   Spatial detail lost: {((spatial_variance - column_variance) / spatial_variance * 100):.1f}%")
    print()

    # Spatial correlations (example)
    if observation_2d.shape[0] > 1:
        row_correlations = []
        for i in range(observation_2d.shape[0] - 1):
            corr = np.corrcoef(observation_2d[i], observation_2d[i+1])[0,1]
            if not np.isnan(corr):
                row_correlations.append(corr)

        if row_correlations:
            avg_row_correlation = np.mean(row_correlations)
            print(f"🔗 Spatial Correlations:")
            print(f"   Average row-to-row correlation: {avg_row_correlation:.3f}")
            print(f"   This spatial structure is LOST in 1D compression!")
        else:
            print(f"🔗 Spatial Correlations: Insufficient variation for analysis")

    print()
    print(f"💡 Key Insight: 2D preserves {size_ratio:.1f}x more spatial information")
    print(f"   CNN can learn from spatial patterns invisible to 1D methods")

    return size_1d, size_2d, size_ratio, spatial_variance, column_variance


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary: 1D vs 2D NPE Comparison

    | Aspect | 1D NPE (Traditional) | 2D NPE (Enhanced) |
    |--------|---------------------|-------------------|
    | **Data Format** | Column counts (Lx values) | Spatial grid (Ly × Lx values) |
    | **Information** | Compressed, spatial loss | Complete spatial structure |
    | **Neural Network** | Standard dense layers | CNN + dense layers |
    | **Spatial Patterns** | ❌ Lost in compression | ✅ Preserved and learned |
    | **Parameter Inference** | Good for simple cases | Enhanced for complex patterns |
    | **Computational Cost** | Lower memory/time | Higher memory/time |
    | **Scientific Insight** | Limited spatial understanding | Full spatial analysis |

    ### When to Use 2D NPE:
    - **Spatial patterns matter** for parameter inference
    - **Computational resources available** for CNN processing
    - **Scientific insight** into spatial processes desired
    - **High accuracy requirements** justify additional complexity

    ### Implementation:
    ✅ **Ready to use**: Add `--use_2d_data` flag to existing workflows
    ✅ **Backward compatible**: All 1D functionality preserved
    ✅ **Scalable**: CNN handles various lattice sizes
    ✅ **Well-tested**: Comprehensive validation and comparison tools
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Next Steps

    🚀 **Try the 2D approach yourself:**

    1. **Quick test**: `sbatch slurm/run_main_2d_light.sh`
    2. **Full comparison**: `sbatch slurm/run_comparison_1d_vs_2d.sh`
    3. **Production run**: `sbatch slurm/run_main_2d.sh`

    📊 **Analyze results:**
    - Compare posterior accuracy (credible intervals)
    - Examine spatial prediction intervals
    - Visualize learned spatial patterns
    - Quantify inference improvements

    📝 **Scientific applications:**
    - Biological pattern formation
    - Spatial epidemiology
    - Ecological dispersal models
    - Materials science diffusion

    The 2D NPE approach opens new possibilities for **spatially-aware parameter inference**! 🎉
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()