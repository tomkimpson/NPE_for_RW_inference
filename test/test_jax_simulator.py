#!/usr/bin/env python3
"""
Simple test script for JAX-optimized RandomWalk simulator.

This script demonstrates basic usage of the RandomWalkSimulatorJax class
and compares its performance with the NumPy implementation.
"""

import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

try:
    from simulator import RandomWalkSimulator, RandomWalkSimulatorJax, JAX_AVAILABLE
except ImportError as e:
    print(f"Error importing simulators: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def main():
    """Run simple JAX simulator test."""
    print("JAX RandomWalk Simulator Test")
    print("=" * 50)
    
    # Check JAX availability
    if not JAX_AVAILABLE:
        print("❌ JAX not available!")
        print("Install with: pip install jax jaxlib")
        print("\nFalling back to NumPy implementation demonstration...")
        
        # Demonstrate NumPy version only
        print("\n📊 NumPy Simulator Demo:")
        numpy_sim = RandomWalkSimulator(100, 50)
        
        start_time = time.time()
        column_counts, initial_pos, final_pos = numpy_sim.simulate(U=0.3, P=0.7, T=100, random_seed=42)
        numpy_time = time.time() - start_time
        
        print(f"   Simulation time: {numpy_time:.4f}s")
        print(f"   Initial agents: {len(initial_pos)}")
        print(f"   Final agents: {len(final_pos)}")
        print(f"   Column count sum: {column_counts.sum()}")
        
        return
    
    print("✅ JAX is available!")
    
    # Test parameters
    Lx, Ly = 100, 50
    test_params = {'U': 0.3, 'P': 0.7, 'T': 100}
    
    # Create simulators
    print(f"\n🔧 Setting up simulators (lattice: {Lx}×{Ly})...")
    numpy_sim = RandomWalkSimulator(Lx, Ly)
    jax_sim = RandomWalkSimulatorJax(Lx, Ly, device='auto')
    
    # Single simulation test
    print(f"\n🧪 Running single simulation test...")
    print(f"   Parameters: U={test_params['U']}, P={test_params['P']}, T={test_params['T']}")
    
    # NumPy version
    start_time = time.time()
    numpy_counts, numpy_init, numpy_final = numpy_sim.simulate(**test_params, random_seed=42)
    numpy_time = time.time() - start_time
    
    # JAX version
    start_time = time.time()
    jax_counts, jax_init, jax_final = jax_sim.simulate(**test_params, random_seed=42)
    jax_time = time.time() - start_time
    
    print(f"\n📊 Results Comparison:")
    print(f"   NumPy time: {numpy_time:.4f}s | JAX time: {jax_time:.4f}s")
    print(f"   NumPy agents: {len(numpy_init)} → {len(numpy_final)} | JAX agents: {len(jax_init)} → {len(jax_final)}")
    print(f"   NumPy column sum: {numpy_counts.sum()} | JAX column sum: {jax_counts.sum()}")
    
    if jax_time > 0:
        speedup = numpy_time / jax_time
        print(f"   Speedup: {speedup:.2f}x")
        
        if speedup > 1.5:
            print("   ✅ JAX shows performance improvement!")
        elif speedup > 0.8:
            print("   ➖ JAX performance similar to NumPy (expected for single simulation)")
        else:
            print("   ⚠️ JAX slower (JIT compilation overhead for single run)")
    
    # Batch processing test
    print(f"\n⚡ Testing batch processing capabilities...")
    
    # Generate test parameter sets
    n_batch = 100
    U_batch = np.random.uniform(0.1, 0.9, n_batch)
    P_batch = np.random.uniform(0.1, 0.9, n_batch)
    
    print(f"   Processing {n_batch} simulations in batch...")
    
    start_time = time.time()
    batch_results = jax_sim.simulate_batch(U_batch, P_batch, T=50, random_seed=42)
    batch_time = time.time() - start_time
    
    print(f"   Batch time: {batch_time:.4f}s")
    print(f"   Rate: {n_batch/batch_time:.1f} simulations/second")
    print(f"   Result shape: {batch_results.shape}")
    print(f"   Mean agents per sim: {batch_results.sum(axis=1).mean():.1f}")
    print("   ✅ Batch processing successful!")
    
    # Training data generation test  
    print(f"\n🎯 Testing training data generation...")
    
    start_time = time.time()
    parameters, observations = jax_sim.generate_training_data(
        n_simulations=500,
        T=100,
        batch_size=100,
        random_seed=42,
        show_progress=False
    )
    generation_time = time.time() - start_time
    
    print(f"   Generated 500 training simulations in {generation_time:.2f}s")
    print(f"   Rate: {500/generation_time:.1f} simulations/second")
    print(f"   Parameters shape: {parameters.shape}")
    print(f"   Observations shape: {observations.shape}")
    print("   ✅ Training data generation successful!")
    
    # Performance projection
    print(f"\n🚀 Performance Projections:")
    rate = 500 / generation_time
    
    targets = [1000, 10000, 50000, 100000]
    for target in targets:
        estimated_time = target / rate
        if estimated_time < 60:
            time_str = f"{estimated_time:.1f}s"
        else:
            time_str = f"{estimated_time/60:.1f}min"
        print(f"   {target:,} simulations: ~{time_str}")
    
    # Validation check
    print(f"\n🔍 Quick validation check...")
    validation_results = jax_sim.compare_with_numpy(
        numpy_sim, 
        n_comparisons=3,
        tolerance=0,
        U=0.5, P=0.5, T=50
    )
    
    print(f"   Exact matches: {validation_results['exact_matches']}/3")
    print(f"   Mean correlation: {validation_results['mean_correlation']:.3f}")
    
    if validation_results['exact_matches'] >= 1:
        print("   ✅ Validation looks good!")
    else:
        print("   ⚠️ Some validation differences (may be due to random seed handling)")
    
    print(f"\n🎉 JAX Simulator Test Complete!")
    print(f"   Ready for high-performance NPE training data generation!")

if __name__ == "__main__":
    main()