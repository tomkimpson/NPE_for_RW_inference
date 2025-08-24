#!/usr/bin/env python3
"""
Validation script for JAX-optimized RandomWalk simulator.

This script compares the JAX implementation against the original NumPy
implementation to ensure correctness and measures performance improvements.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import time
from typing import Dict, Any

try:
    from simulator import RandomWalkSimulator, RandomWalkSimulatorJax, JAX_AVAILABLE
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality and interface compatibility."""
    print("=" * 60)
    print("BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    
    if not JAX_AVAILABLE:
        print("❌ JAX not available - cannot test JAX implementation")
        return False
    
    # Create simulators with same parameters
    Lx, Ly = 50, 25
    numpy_sim = RandomWalkSimulator(Lx, Ly)
    jax_sim = RandomWalkSimulatorJax(Lx, Ly, device='cpu')  # Force CPU for comparison
    
    print(f"✅ Created simulators with lattice size {Lx} × {Ly}")
    
    # Test parameter validation
    test_params = [
        {'U': 0.5, 'P': 0.7, 'T': 10},
        {'U': 0.1, 'P': 1.0, 'T': 0},
        {'U': 1.0, 'P': 0.0, 'T': 50}
    ]
    
    for i, params in enumerate(test_params):
        try:
            # Test NumPy version
            numpy_result = numpy_sim.simulate(**params, random_seed=42)
            
            # Test JAX version  
            jax_result = jax_sim.simulate(**params, random_seed=42)
            
            # Verify return format
            assert len(numpy_result) == 3, "NumPy simulator should return 3 elements"
            assert len(jax_result) == 3, "JAX simulator should return 3 elements"
            
            numpy_counts, numpy_init, numpy_final = numpy_result
            jax_counts, jax_init, jax_final = jax_result
            
            # Verify shapes and types
            assert numpy_counts.shape == jax_counts.shape, f"Column counts shape mismatch: {numpy_counts.shape} vs {jax_counts.shape}"
            assert isinstance(jax_init, list), "JAX initial positions should be converted to list"
            assert isinstance(jax_final, list), "JAX final positions should be converted to list"
            
            print(f"✅ Test case {i+1}: U={params['U']}, P={params['P']}, T={params['T']} - Format OK")
            
        except Exception as e:
            print(f"❌ Test case {i+1} failed: {e}")
            return False
    
    print("✅ All basic functionality tests passed!")
    return True

def test_numerical_validation():
    """Test numerical accuracy by comparing results with same random seeds."""
    print("\n" + "=" * 60)
    print("NUMERICAL VALIDATION TEST")
    print("=" * 60)
    
    if not JAX_AVAILABLE:
        print("❌ JAX not available - skipping numerical validation")
        return False
    
    # Create simulators
    Lx, Ly = 100, 50
    numpy_sim = RandomWalkSimulator(Lx, Ly)
    jax_sim = RandomWalkSimulatorJax(Lx, Ly, device='cpu')
    
    # Run comparison using built-in method
    print("Running numerical comparison...")
    comparison_results = jax_sim.compare_with_numpy(
        numpy_sim, 
        n_comparisons=10,
        tolerance=0.0,
        U=0.3, P=0.7, T=100
    )
    
    print(f"📊 Comparison Results:")
    print(f"   Exact matches: {comparison_results['exact_matches']}/10")
    print(f"   Success rate: {comparison_results['success_rate']:.1%}")
    print(f"   Mean correlation: {comparison_results['mean_correlation']:.4f}")
    print(f"   Mean difference: {comparison_results['mean_difference']:.4f}")
    print(f"   Max difference: {comparison_results['max_difference']:.1f}")
    print(f"   Agent conservation error: {comparison_results['mean_agent_conservation_error']:.1f}")
    
    # Assess results
    if comparison_results['success_rate'] >= 0.5:  # Allow some randomness differences
        print("✅ Numerical validation passed!")
        if comparison_results['mean_correlation'] > 0.95:
            print("✅ High correlation indicates excellent agreement!")
        return True
    else:
        print("⚠️ Numerical validation shows differences - this may be due to random number generation differences")
        return False

def test_performance_benchmark():
    """Test performance improvements."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    if not JAX_AVAILABLE:
        print("❌ JAX not available - skipping performance test")
        return False
    
    # Standard benchmark parameters
    Lx, Ly = 100, 50
    params = {'U': 0.3, 'P': 0.7, 'T': 100}
    
    # Create simulators
    print("Setting up simulators...")
    numpy_sim = RandomWalkSimulator(Lx, Ly)
    jax_sim = RandomWalkSimulatorJax(Lx, Ly, device='auto')  # Use best available device
    
    # Benchmark NumPy version
    print("Benchmarking NumPy implementation...")
    numpy_times = []
    for i in range(10):
        start_time = time.time()
        numpy_sim.simulate(**params, random_seed=i)
        numpy_times.append(time.time() - start_time)
    
    numpy_mean_time = np.mean(numpy_times)
    numpy_rate = 1.0 / numpy_mean_time
    
    # Benchmark JAX version
    print("Benchmarking JAX implementation...")
    jax_results = jax_sim.benchmark_performance(n_trials=10, **params)
    
    jax_mean_time = jax_results['mean_time']
    jax_rate = jax_results['simulations_per_second']
    
    # Calculate speedup
    speedup = jax_rate / numpy_rate
    
    print(f"📊 Performance Results:")
    print(f"   NumPy: {numpy_mean_time:.4f}s per simulation ({numpy_rate:.1f} sims/sec)")
    print(f"   JAX:   {jax_mean_time:.4f}s per simulation ({jax_rate:.1f} sims/sec)")
    print(f"   Speedup: {speedup:.1f}x")
    print(f"   JAX Device: {jax_results['device']}")
    
    if speedup > 2.0:
        print("✅ Significant performance improvement achieved!")
        return True
    elif speedup > 1.0:
        print("✅ Performance improvement achieved!")
        return True
    else:
        print("⚠️ No significant performance improvement - check JAX installation")
        return False

def test_batch_processing():
    """Test batch processing capabilities."""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING TEST")
    print("=" * 60)
    
    if not JAX_AVAILABLE:
        print("❌ JAX not available - skipping batch test")
        return False
    
    # Create JAX simulator
    Lx, Ly = 100, 50
    jax_sim = RandomWalkSimulatorJax(Lx, Ly)
    
    # Test small batch
    print("Testing batch simulation...")
    n_sims = 100
    U_batch = np.random.uniform(0.1, 0.9, n_sims)
    P_batch = np.random.uniform(0.1, 0.9, n_sims)
    
    start_time = time.time()
    batch_results = jax_sim.simulate_batch(U_batch, P_batch, T=50, random_seed=42)
    batch_time = time.time() - start_time
    
    print(f"📊 Batch Results:")
    print(f"   Processed {n_sims} simulations in {batch_time:.4f}s")
    print(f"   Rate: {n_sims/batch_time:.1f} simulations/second")
    print(f"   Result shape: {batch_results.shape}")
    print(f"   Mean agents per simulation: {batch_results.sum(axis=1).mean():.1f}")
    
    # Verify results
    assert batch_results.shape == (n_sims, Lx), f"Unexpected batch result shape: {batch_results.shape}"
    assert np.all(batch_results >= 0), "All column counts should be non-negative"
    
    print("✅ Batch processing test passed!")
    return True

def test_training_data_generation():
    """Test large-scale training data generation."""
    print("\n" + "=" * 60)
    print("TRAINING DATA GENERATION TEST")
    print("=" * 60)
    
    if not JAX_AVAILABLE:
        print("❌ JAX not available - skipping training data test")
        return False
    
    # Create JAX simulator
    jax_sim = RandomWalkSimulatorJax(100, 50)
    
    # Generate moderate-sized training dataset
    print("Generating training data...")
    n_sims = 1000
    
    start_time = time.time()
    parameters, observations = jax_sim.generate_training_data(
        n_simulations=n_sims,
        T=100,
        batch_size=250,
        random_seed=42,
        show_progress=True
    )
    generation_time = time.time() - start_time
    
    print(f"📊 Training Data Results:")
    print(f"   Generated {n_sims} simulations in {generation_time:.2f}s")
    print(f"   Rate: {n_sims/generation_time:.1f} simulations/second")
    print(f"   Parameters shape: {parameters.shape}")
    print(f"   Observations shape: {observations.shape}")
    print(f"   U range: [{parameters[:, 0].min():.3f}, {parameters[:, 0].max():.3f}]")
    print(f"   P range: [{parameters[:, 1].min():.3f}, {parameters[:, 1].max():.3f}]")
    
    # Verify results
    assert parameters.shape == (n_sims, 2), f"Unexpected parameters shape: {parameters.shape}"
    assert observations.shape == (n_sims, 100), f"Unexpected observations shape: {observations.shape}"
    
    print("✅ Training data generation test passed!")
    
    # Estimate time for larger datasets
    rate = n_sims / generation_time
    for target in [10000, 50000, 100000]:
        estimated_time = target / rate / 60  # Convert to minutes
        print(f"   Estimated time for {target} simulations: {estimated_time:.1f} minutes")
    
    return True

def main():
    """Run all validation tests."""
    print("JAX RandomWalk Simulator Validation")
    print("=" * 60)
    
    # Check JAX availability
    if not JAX_AVAILABLE:
        print("❌ JAX is not available!")
        print("Install with: pip install jax jaxlib")
        print("For GPU support: pip install jax[cuda12_pip] (or appropriate CUDA version)")
        return False
    
    print(f"✅ JAX is available and ready for testing")
    
    # Run all tests
    tests = [
        test_basic_functionality,
        test_numerical_validation, 
        test_performance_benchmark,
        test_batch_processing,
        test_training_data_generation
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_func.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 All validation tests passed! JAX simulator is ready for production use.")
    elif passed >= total * 0.8:
        print("⚠️ Most tests passed. JAX simulator is likely working correctly.")
    else:
        print("❌ Multiple test failures. Please check JAX installation and implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)