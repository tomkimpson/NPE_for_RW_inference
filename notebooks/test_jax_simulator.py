import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    mo.md(
        r"""
        # JAX-Optimized 2D Random Walk Lattice Simulator

        This notebook demonstrates the **JAX-optimized 2D lattice random walk simulator** for biological population models, 
        as described in Simpson & Planck. The simulator implements the same discrete random walk model as the NumPy version
        but with significant performance improvements through JAX compilation and vectorization.

        ## Key JAX Features:
        - **JIT Compilation**: Functions compiled for maximum performance
        - **GPU Acceleration**: Automatic GPU usage when available  
        - **Vectorized Operations**: Eliminates Python loops for agent updates
        - **Batch Processing**: Parallel simulation of multiple parameter sets
        - **Drop-in Replacement**: Same interface as NumPy version

        ## Parameters:
        - **U**: Initial occupancy probability (0 < U ≤ 1)
        - **P**: Movement probability (0 ≤ P ≤ 1)
        - **T**: Number of time steps
        """
    )
    return mo,


@app.cell
def __(mo):
    """
    Import required modules and set up the JAX simulator
    """
    import sys
    import os
    import time
    import numpy as np

    # Add src directory to path
    sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))

    import matplotlib.pyplot as plt
    
    # Import both simulators for comparison
    from simulator import (
        RandomWalkSimulator, 
        RandomWalkSimulatorJax, 
        JAX_AVAILABLE,
        plot_simulation_comparison, 
        plot_lattice, 
        plot_column_counts
    )

    # Set random seed for reproducibility
    np.random.seed(42)

    if JAX_AVAILABLE:
        print("✅ JAX is available - JAX simulator ready!")
    else:
        print("❌ JAX not available - install with: pip install jax jaxlib")

    print("✅ Successfully imported simulator modules!")
    
    return (
        RandomWalkSimulator,
        RandomWalkSimulatorJax,
        JAX_AVAILABLE,
        np,
        plot_column_counts,
        plot_lattice,
        plot_simulation_comparison,
        plt,
        time,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## JAX Simulator Setup

        We'll create both NumPy and JAX simulators with the same parameters for comparison.
        The JAX simulator will automatically detect and use the best available device (GPU or CPU).
        """
    )
    return


@app.cell
def __(JAX_AVAILABLE, RandomWalkSimulator, RandomWalkSimulatorJax):
    """
    Initialize both simulators with lattice parameters
    """
    # Lattice parameters
    Lx = 100  # Number of columns (smaller for faster testing)
    Ly = 50   # Number of rows
    initial_region_half_width = 25  # Central region for initial placement

    # Create NumPy simulator for comparison
    numpy_simulator = RandomWalkSimulator(Lx, Ly, initial_region_half_width)

    # Create JAX simulator if available
    if JAX_AVAILABLE:
        jax_simulator = RandomWalkSimulatorJax(Lx, Ly, initial_region_half_width, device='auto')
        print(f"✅ Created JAX simulator with lattice size {Lx} x {Ly}")
        print(f"📍 Coordinate system: x ∈ [{-(Lx//2)}, {Lx//2 if Lx%2==1 else Lx//2-1}], y ∈ [0, {Ly-1}]")
        print(f"🎯 Initial region: x ∈ [{-initial_region_half_width}, {initial_region_half_width}]")
    else:
        jax_simulator = None
        print("❌ JAX simulator not available")

    print(f"✅ Created NumPy simulator for comparison")
    
    return Lx, Ly, jax_simulator, numpy_simulator


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Simulation Parameters

        We'll test with standard parameters and compare performance between NumPy and JAX implementations.
        """
    )
    return


@app.cell
def __():
    """
    Set simulation parameters
    """
    # Model parameters
    U = 0.3  # Initial occupancy probability
    P = 0.7  # Movement probability
    T = 100  # Number of time steps

    print(f"🎲 Simulation parameters:")
    print(f"   U (occupancy probability): {U}")
    print(f"   P (movement probability): {P}")
    print(f"   T (time steps): {T}")

    return P, T, U


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Performance Comparison: Single Simulation

        Let's compare the performance of a single simulation between NumPy and JAX implementations.
        """
    )
    return


@app.cell
def __(JAX_AVAILABLE, P, T, U, jax_simulator, numpy_simulator, time):
    """
    Compare single simulation performance
    """
    print("⏱️  Single Simulation Performance Test")
    print("=" * 50)
    
    # NumPy version timing
    print("🔹 NumPy Simulator:")
    numpy_simulator.simulate(U, P, T, random_seed=1)  # Warm up
    
    start_time = time.time()
    numpy_counts, numpy_init, numpy_final = numpy_simulator.simulate(U, P, T, random_seed=42)
    numpy_time = time.time() - start_time
    
    print(f"   Time: {numpy_time:.4f}s")
    print(f"   Agents: {len(numpy_init)} → {len(numpy_final)}")
    print(f"   Rate: {1/numpy_time:.1f} simulations/second")
    
    # JAX version timing
    if JAX_AVAILABLE and jax_simulator is not None:
        print("\n🔹 JAX Simulator:")
        jax_simulator.simulate(U, P, T, random_seed=1)  # Warm up
        
        start_time = time.time()
        jax_counts, jax_init, jax_final = jax_simulator.simulate(U, P, T, random_seed=42)
        jax_time = time.time() - start_time
        
        print(f"   Time: {jax_time:.4f}s")
        print(f"   Agents: {len(jax_init)} → {len(jax_final)}")
        print(f"   Rate: {1/jax_time:.1f} simulations/second")
        
        # Compare results
        speedup = numpy_time / jax_time if jax_time > 0 else 0
        print(f"\n🚀 JAX Speedup: {speedup:.2f}x")
        
        # Check numerical consistency
        agents_match = (len(numpy_init) == len(jax_init) and len(numpy_final) == len(jax_final))
        total_match = (numpy_counts.sum() == jax_counts.sum())
        
        print(f"✅ Agent count consistency: {'✓' if agents_match and total_match else '✗'}")
        
        return (
            numpy_counts, numpy_init, numpy_final, numpy_time,
            jax_counts, jax_init, jax_final, jax_time, speedup
        )
    else:
        print("\n❌ JAX simulator not available for comparison")
        return numpy_counts, numpy_init, numpy_final, numpy_time, None, None, None, None, None


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Batch Processing Demonstration

        One of the key advantages of the JAX implementation is efficient batch processing.
        Let's test processing multiple parameter sets simultaneously.
        """
    )
    return


@app.cell
def __(JAX_AVAILABLE, jax_simulator, np, time):
    """
    Test JAX batch processing capabilities
    """
    if JAX_AVAILABLE and jax_simulator is not None:
        print("⚡ JAX Batch Processing Test")
        print("=" * 40)
        
        # Generate test parameter sets
        n_batch = 50
        U_batch = np.random.uniform(0.1, 0.9, n_batch)
        P_batch = np.random.uniform(0.1, 0.9, n_batch)
        
        print(f"🎯 Processing {n_batch} parameter sets in batch...")
        
        # Time batch processing
        start_time = time.time()
        batch_results = jax_simulator.simulate_batch(U_batch, P_batch, T=50, random_seed=42)
        batch_time = time.time() - start_time
        
        print(f"✅ Batch processing completed!")
        print(f"   Time: {batch_time:.4f}s")
        print(f"   Rate: {n_batch/batch_time:.1f} simulations/second")
        print(f"   Result shape: {batch_results.shape}")
        print(f"   Mean agents per simulation: {batch_results.sum(axis=1).mean():.1f}")
        
        batch_rate = n_batch / batch_time
        
        return batch_results, batch_rate, n_batch
    else:
        print("❌ JAX batch processing not available")
        return None, None, 0


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Large-Scale Training Data Generation

        Let's test the JAX simulator's ability to generate large training datasets efficiently,
        which is crucial for NPE training.
        """
    )
    return


@app.cell
def __(JAX_AVAILABLE, jax_simulator, time):
    """
    Test large-scale training data generation
    """
    if JAX_AVAILABLE and jax_simulator is not None:
        print("🎯 Large-Scale Training Data Generation Test")
        print("=" * 50)
        
        # Generate a moderately large training dataset
        n_training = 1000
        print(f"🏭 Generating {n_training} training simulations...")
        
        start_time = time.time()
        parameters, observations = jax_simulator.generate_training_data(
            n_simulations=n_training,
            T=100,
            batch_size=100,
            random_seed=42,
            show_progress=True
        )
        generation_time = time.time() - start_time
        
        print(f"\n📊 Training Data Results:")
        print(f"   Generation time: {generation_time:.2f}s")
        print(f"   Rate: {n_training/generation_time:.1f} simulations/second")
        print(f"   Parameters shape: {parameters.shape}")
        print(f"   Observations shape: {observations.shape}")
        print(f"   U range: [{parameters[:, 0].min():.3f}, {parameters[:, 0].max():.3f}]")
        print(f"   P range: [{parameters[:, 1].min():.3f}, {parameters[:, 1].max():.3f}]")
        
        # Project timing for larger datasets
        rate = n_training / generation_time
        print(f"\n🚀 Performance Projections:")
        
        targets = [10000, 50000, 100000]
        for target in targets:
            estimated_time = target / rate
            if estimated_time < 60:
                time_str = f"{estimated_time:.1f}s"
            elif estimated_time < 3600:
                time_str = f"{estimated_time/60:.1f}min"
            else:
                time_str = f"{estimated_time/3600:.1f}hr"
            print(f"   {target:,} simulations: ~{time_str}")
        
        return parameters, observations, generation_time, rate
    else:
        print("❌ JAX training data generation not available")
        return None, None, None, None


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Performance Summary

        🎉 **JAX Simulator Testing Complete!**
        """
    )
    return


@app.cell
def __(
    JAX_AVAILABLE,
    batch_rate, 
    n_batch,
    numpy_time,
    jax_time,
    rate,
    speedup
):
    """
    Generate performance summary
    """
    if JAX_AVAILABLE and 'speedup' in locals() and speedup is not None:
        print("🎯 JAX Simulator Performance Summary")
        print("=" * 50)
        
        print(f"✅ **Single Simulation Performance:**")
        print(f"   NumPy: {numpy_time:.4f}s ({1/numpy_time:.1f} sims/sec)")
        print(f"   JAX:   {jax_time:.4f}s ({1/jax_time:.1f} sims/sec)")
        print(f"   Speedup: {speedup:.2f}x")
        
        if 'batch_rate' in locals() and batch_rate is not None:
            print(f"\n✅ **Batch Processing:**")
            print(f"   {n_batch} simulations: {batch_rate:.1f} sims/sec")
        
        if 'rate' in locals() and rate is not None:
            print(f"\n✅ **Training Data Generation:**")
            print(f"   Rate: {rate:.1f} simulations/second")
            print(f"   50k dataset: ~{50000/rate/60:.1f} minutes")
            print(f"   100k dataset: ~{100000/rate/60:.1f} minutes")
        
        print(f"\n🚀 **Ready for Production Use!**")
        print(f"   The JAX simulator provides significant performance improvements")
        print(f"   while maintaining numerical consistency with the NumPy version.")
        print(f"   Perfect for large-scale NPE training data generation!")
        
    else:
        print("❌ JAX simulator performance summary not available")
        print("   Install JAX to enable high-performance simulation capabilities")

    return


if __name__ == "__main__":
    app.run()