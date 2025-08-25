#!/usr/bin/env python3
"""
Simulator Comparison Test Script

This script compares the two simulator classes (RandomWalkSimulator vs RandomWalkSimulatorNumpy)
and generates comprehensive comparison plots and statistics.

The script validates that both simulators:
- Produce identical results with same random seeds
- Maintain agent conservation 
- Have consistent statistical properties
- Show expected performance differences

Results are saved to test/results/ directory.
"""

import sys
import os
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from simulator import (
    RandomWalkSimulator, 
    RandomWalkSimulatorNumpy,
    plot_simulation_comparison,
    plot_column_counts,
    plot_lattice
)


class SimulatorComparison:
    """Class to handle comprehensive comparison of the two simulator implementations."""
    
    def __init__(self, results_dir: str = "test/results"):
        """Initialize the comparison with results directory."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Test parameters
        self.test_params = {
            'Lx': 100,
            'Ly': 50,  
            'initial_region_half_width': 25,
            'U': 0.3,
            'P': 0.7,
            'T': 100,
            'random_seed': 42
        }
        
        # Initialize simulators
        self.sim_original = RandomWalkSimulator(
            self.test_params['Lx'], 
            self.test_params['Ly'],
            self.test_params['initial_region_half_width']
        )
        
        self.sim_numpy = RandomWalkSimulatorNumpy(
            self.test_params['Lx'], 
            self.test_params['Ly'],
            self.test_params['initial_region_half_width']
        )
        
        # Storage for results
        self.results = {}
        
    def run_single_comparison(self, label: str, **params) -> Dict[str, Any]:
        """Run a single comparison between the two simulators."""
        print(f"\n{'='*60}")
        print(f"Running comparison: {label}")
        print(f"Parameters: {params}")
        print(f"{'='*60}")
        
        # Merge params with defaults
        test_params = {**self.test_params, **params}
        
        # Run original simulator
        print("Running original simulator...")
        start_time = time.time()
        col_counts_orig, init_pos_orig, final_pos_orig = self.sim_original.simulate(
            U=test_params['U'],
            P=test_params['P'], 
            T=test_params['T'],
            random_seed=test_params['random_seed']
        )
        time_orig = time.time() - start_time
        
        # Run NumPy simulator  
        print("Running NumPy simulator...")
        start_time = time.time()
        col_counts_numpy, init_pos_numpy, final_pos_numpy = self.sim_numpy.simulate(
            U=test_params['U'],
            P=test_params['P'],
            T=test_params['T'], 
            random_seed=test_params['random_seed']
        )
        time_numpy = time.time() - start_time
        
        # Convert numpy arrays to lists for comparison (original returns lists)
        init_pos_numpy = [(int(x), int(y)) for x, y in init_pos_numpy]
        final_pos_numpy = [(int(x), int(y)) for x, y in final_pos_numpy]
        
        # Analyze results
        results = self._analyze_comparison(
            label, test_params, time_orig, time_numpy,
            col_counts_orig, col_counts_numpy,
            init_pos_orig, init_pos_numpy,
            final_pos_orig, final_pos_numpy
        )
        
        return results
        
    def _analyze_comparison(self, label: str, params: Dict, time_orig: float, time_numpy: float,
                          col_counts_orig: np.ndarray, col_counts_numpy: np.ndarray,
                          init_pos_orig: List, init_pos_numpy: List,
                          final_pos_orig: List, final_pos_numpy: List) -> Dict[str, Any]:
        """Analyze and compare simulation results."""
        
        # Check exact equality
        counts_equal = np.array_equal(col_counts_orig, col_counts_numpy)
        init_equal = init_pos_orig == init_pos_numpy  
        final_equal = final_pos_orig == final_pos_numpy
        
        # Agent conservation check
        n_agents_orig = len(init_pos_orig)
        n_agents_numpy = len(init_pos_numpy)
        n_agents_final_orig = len(final_pos_orig)
        n_agents_final_numpy = len(final_pos_numpy)
        
        conservation_orig = (n_agents_orig == n_agents_final_orig == np.sum(col_counts_orig))
        conservation_numpy = (n_agents_numpy == n_agents_final_numpy == np.sum(col_counts_numpy))
        
        # Statistical comparisons
        stats_orig = self._compute_stats(col_counts_orig)
        stats_numpy = self._compute_stats(col_counts_numpy)
        
        # Performance metrics
        speedup = time_orig / time_numpy if time_numpy > 0 else float('inf')
        
        results = {
            'label': label,
            'params': params,
            'timing': {
                'original': time_orig,
                'numpy': time_numpy,
                'speedup': speedup
            },
            'validation': {
                'counts_equal': counts_equal,
                'initial_equal': init_equal,
                'final_equal': final_equal,
                'conservation_orig': conservation_orig,
                'conservation_numpy': conservation_numpy
            },
            'agent_counts': {
                'original': n_agents_orig,
                'numpy': n_agents_numpy,
                'final_original': n_agents_final_orig,
                'final_numpy': n_agents_final_numpy
            },
            'statistics': {
                'original': stats_orig,
                'numpy': stats_numpy
            },
            'data': {
                'col_counts_orig': col_counts_orig,
                'col_counts_numpy': col_counts_numpy,
                'init_pos_orig': init_pos_orig,
                'init_pos_numpy': init_pos_numpy,
                'final_pos_orig': final_pos_orig,
                'final_pos_numpy': final_pos_numpy
            }
        }
        
        # Print results
        self._print_comparison_results(results)
        
        return results
    
    def _compute_stats(self, col_counts: np.ndarray) -> Dict[str, float]:
        """Compute statistical measures of column count distribution."""
        if len(col_counts) == 0 or np.sum(col_counts) == 0:
            return {'mean': 0, 'std': 0, 'skew': 0, 'center_of_mass': 0}
            
        # Center of mass calculation
        x_min = -(self.test_params['Lx'] // 2)
        x_positions = np.arange(x_min, x_min + len(col_counts))
        center_of_mass = np.average(x_positions, weights=col_counts)
        
        # Basic statistics
        mean = np.mean(col_counts)
        std = np.std(col_counts)
        
        # Skewness
        if std > 0:
            skew = np.mean(((col_counts - mean) / std) ** 3)
        else:
            skew = 0
            
        return {
            'mean': mean,
            'std': std, 
            'skew': skew,
            'center_of_mass': center_of_mass
        }
    
    def _print_comparison_results(self, results: Dict[str, Any]):
        """Print detailed comparison results."""
        print(f"\n📊 Comparison Results for {results['label']}")
        print("-" * 50)
        
        # Validation results
        val = results['validation']
        print(f"✅ Validation Status:")
        print(f"   Column counts identical: {'✅' if val['counts_equal'] else '❌'}")
        print(f"   Initial positions identical: {'✅' if val['initial_equal'] else '❌'}")
        print(f"   Final positions identical: {'✅' if val['final_equal'] else '❌'}")
        print(f"   Agent conservation (original): {'✅' if val['conservation_orig'] else '❌'}")
        print(f"   Agent conservation (numpy): {'✅' if val['conservation_numpy'] else '❌'}")
        
        # Agent counts
        counts = results['agent_counts']
        print(f"\n🔢 Agent Counts:")
        print(f"   Initial agents - Original: {counts['original']}, NumPy: {counts['numpy']}")
        print(f"   Final agents - Original: {counts['final_original']}, NumPy: {counts['final_numpy']}")
        
        # Performance
        timing = results['timing']
        print(f"\n⏱️ Performance:")
        print(f"   Original simulator: {timing['original']:.4f} seconds")
        print(f"   NumPy simulator: {timing['numpy']:.4f} seconds")
        print(f"   Speedup: {timing['speedup']:.2f}x")
        
        # Statistics comparison
        stats_orig = results['statistics']['original']
        stats_numpy = results['statistics']['numpy']
        print(f"\n📈 Statistical Comparison:")
        print(f"   Mean - Original: {stats_orig['mean']:.3f}, NumPy: {stats_numpy['mean']:.3f}")
        print(f"   Std Dev - Original: {stats_orig['std']:.3f}, NumPy: {stats_numpy['std']:.3f}")
        print(f"   Center of Mass - Original: {stats_orig['center_of_mass']:.3f}, NumPy: {stats_numpy['center_of_mass']:.3f}")
        
    def run_comprehensive_test(self):
        """Run comprehensive test suite with multiple parameter sets."""
        print("🚀 Starting Comprehensive Simulator Comparison Test")
        print(f"Results will be saved to: {self.results_dir}")
        
        # Test scenarios
        scenarios = [
            ("Standard Parameters", {}),
            ("High Movement Probability", {"P": 1.0}),
            ("Low Movement Probability", {"P": 0.1}),
            ("High Occupancy", {"U": 0.8}),
            ("Low Occupancy", {"U": 0.1}),
            ("Long Simulation", {"T": 200}),
            ("Short Simulation", {"T": 50}),
        ]
        
        # Run all scenarios
        all_results = []
        for label, params in scenarios:
            # Use different seed for each test to avoid correlation
            seed = self.test_params['random_seed'] + len(all_results)
            params['random_seed'] = seed
            
            result = self.run_single_comparison(label, **params)
            all_results.append(result)
            self.results[label] = result
        
        # Generate summary
        self._generate_summary(all_results)
        
        # Create visualizations
        self._create_comparison_plots(all_results)
        
        return all_results
    
    def _generate_summary(self, results: List[Dict[str, Any]]):
        """Generate and save summary statistics."""
        summary_file = self.results_dir / "comparison_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SIMULATOR COMPARISON SUMMARY REPORT\n") 
            f.write("="*80 + "\n\n")
            
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Scenarios Tested: {len(results)}\n\n")
            
            # Overall validation status
            all_valid = all(
                r['validation']['counts_equal'] and 
                r['validation']['initial_equal'] and
                r['validation']['final_equal'] and
                r['validation']['conservation_orig'] and
                r['validation']['conservation_numpy']
                for r in results
            )
            
            f.write(f"Overall Validation Status: {'✅ PASSED' if all_valid else '❌ FAILED'}\n\n")
            
            # Performance summary
            speedups = [r['timing']['speedup'] for r in results]
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Speedup: {np.mean(speedups):.2f}x\n")
            f.write(f"Median Speedup: {np.median(speedups):.2f}x\n")
            f.write(f"Min/Max Speedup: {np.min(speedups):.2f}x / {np.max(speedups):.2f}x\n\n")
            
            # Detailed results for each scenario
            f.write("DETAILED RESULTS BY SCENARIO:\n")
            f.write("="*60 + "\n")
            
            for result in results:
                f.write(f"\nScenario: {result['label']}\n")
                f.write(f"Parameters: {result['params']}\n")
                
                val = result['validation']
                f.write(f"Validation - Counts Equal: {val['counts_equal']}\n")
                f.write(f"Validation - Positions Equal: {val['initial_equal']} / {val['final_equal']}\n")
                f.write(f"Agent Conservation: {val['conservation_orig']} / {val['conservation_numpy']}\n")
                
                timing = result['timing']
                f.write(f"Timing - Original: {timing['original']:.4f}s, NumPy: {timing['numpy']:.4f}s\n")
                f.write(f"Speedup: {timing['speedup']:.2f}x\n")
                
                f.write("-" * 30 + "\n")
        
        print(f"📝 Summary saved to: {summary_file}")
    
    def _create_comparison_plots(self, results: List[Dict[str, Any]]):
        """Create comprehensive comparison plots."""
        # 1. Main simulation comparison plot for standard parameters
        standard_result = results[0]  # First result should be standard parameters
        self._plot_simulation_comparison(standard_result)
        
        # 2. Performance comparison plot
        self._plot_performance_comparison(results)
        
        # 3. Statistical validation plot
        self._plot_statistical_validation(results)
        
        print(f"🎨 All plots saved to: {self.results_dir}")
    
    def _plot_simulation_comparison(self, result: Dict[str, Any]):
        """Create detailed simulation comparison plot."""
        data = result['data']
        params = result['params']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Plot original simulator results
        ax1 = plt.subplot(2, 3, 1)
        self._plot_lattice_state(data['init_pos_orig'], params['Lx'], params['Ly'], 
                                "Initial State - Original", ax1)
        
        ax2 = plt.subplot(2, 3, 2)  
        self._plot_lattice_state(data['final_pos_orig'], params['Lx'], params['Ly'],
                                "Final State - Original", ax2)
        
        ax3 = plt.subplot(2, 3, 3)
        self._plot_column_distribution(data['col_counts_orig'], params['Lx'],
                                     "Column Counts - Original", ax3)
        
        # Plot NumPy simulator results
        ax4 = plt.subplot(2, 3, 4)
        self._plot_lattice_state(data['init_pos_numpy'], params['Lx'], params['Ly'],
                                "Initial State - NumPy", ax4)
        
        ax5 = plt.subplot(2, 3, 5)
        self._plot_lattice_state(data['final_pos_numpy'], params['Lx'], params['Ly'],
                                "Final State - NumPy", ax5)
        
        ax6 = plt.subplot(2, 3, 6)
        self._plot_column_distribution(data['col_counts_numpy'], params['Lx'],
                                     "Column Counts - NumPy", ax6)
        
        # Add overall title with validation info
        validation = result['validation']
        identical = validation['counts_equal'] and validation['initial_equal'] and validation['final_equal']
        status_text = "✅ IDENTICAL RESULTS" if identical else "❌ DIFFERENT RESULTS"
        
        plt.suptitle(f'Simulator Comparison - {result["label"]}\n{status_text}\n'
                    f'Parameters: U={params["U"]}, P={params["P"]}, T={params["T"]}', 
                    fontsize=16, y=0.95)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / "simulator_comparison_plot.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Main comparison plot saved to: {plot_file}")
        
    def _plot_lattice_state(self, positions: List[Tuple[int, int]], Lx: int, Ly: int, 
                           title: str, ax: plt.Axes):
        """Plot lattice state on given axes."""
        # Calculate x boundaries (centered around 0)
        x_min = -(Lx // 2)
        x_max = Lx // 2 if Lx % 2 == 1 else (Lx // 2) - 1
        
        # Create lattice grid
        lattice = np.zeros((Ly, Lx))
        
        # Mark agent positions
        for x, y in positions:
            if x_min <= x <= x_max and 0 <= y < Ly:
                array_index = x - x_min
                lattice[y, array_index] += 1
        
        # Plot
        im = ax.imshow(lattice, cmap='Blues', origin='lower', aspect='equal', 
                      extent=[x_min, x_max+1, 0, Ly])
        ax.set_xlabel('x (centered coordinates)')
        ax.set_ylabel('y (row)')
        ax.set_title(f'{title} ({len(positions)} agents)')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Number of agents')
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
    def _plot_column_distribution(self, column_counts: np.ndarray, Lx: int, 
                                 title: str, ax: plt.Axes):
        """Plot column distribution on given axes."""
        # Calculate centered x positions
        x_min = -(Lx // 2)
        x_max = Lx // 2 if Lx % 2 == 1 else (Lx // 2) - 1
        x_positions = np.arange(x_min, x_max + 1)
        
        ax.bar(x_positions, column_counts, alpha=0.7, color='skyblue', edgecolor='navy')
        
        ax.set_xlabel('Column (x, centered coordinates)')
        ax.set_ylabel('Number of agents')
        ax.set_title(f'{title} (Total: {np.sum(column_counts)} agents)')
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
    def _plot_performance_comparison(self, results: List[Dict[str, Any]]):
        """Create performance comparison plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data
        labels = [r['label'] for r in results]
        times_orig = [r['timing']['original'] for r in results]
        times_numpy = [r['timing']['numpy'] for r in results]
        speedups = [r['timing']['speedup'] for r in results]
        
        # Plot 1: Execution times
        x_pos = np.arange(len(labels))
        width = 0.35
        
        ax1.bar(x_pos - width/2, times_orig, width, label='Original', alpha=0.8, color='orange')
        ax1.bar(x_pos + width/2, times_numpy, width, label='NumPy', alpha=0.8, color='blue')
        
        ax1.set_xlabel('Test Scenario')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup factors
        bars = ax2.bar(x_pos, speedups, alpha=0.8, color='green')
        ax2.set_xlabel('Test Scenario')
        ax2.set_ylabel('Speedup Factor (x)')
        ax2.set_title('NumPy Simulator Speedup')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add speedup values on bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{speedup:.1f}x', ha='center', va='bottom')
        
        plt.suptitle('Performance Comparison: Original vs NumPy Simulator', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / "performance_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"⚡ Performance comparison plot saved to: {plot_file}")
        
    def _plot_statistical_validation(self, results: List[Dict[str, Any]]):
        """Create statistical validation plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract statistical data
        stats_orig = [r['statistics']['original'] for r in results]
        stats_numpy = [r['statistics']['numpy'] for r in results]
        labels = [r['label'] for r in results]
        
        metrics = ['mean', 'std', 'skew', 'center_of_mass']
        metric_titles = ['Mean', 'Standard Deviation', 'Skewness', 'Center of Mass']
        
        for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[i//2, i%2]
            
            values_orig = [s[metric] for s in stats_orig]
            values_numpy = [s[metric] for s in stats_numpy]
            
            # Create scatter plot comparing the two implementations
            ax.scatter(values_orig, values_numpy, alpha=0.7, s=100)
            
            # Add perfect correlation line
            min_val = min(min(values_orig), min(values_numpy))
            max_val = max(max(values_orig), max(values_numpy))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, 
                   label='Perfect Correlation')
            
            ax.set_xlabel(f'{title} - Original')
            ax.set_ylabel(f'{title} - NumPy')
            ax.set_title(f'{title} Comparison')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add correlation coefficient
            if len(values_orig) > 1 and np.std(values_orig) > 0 and np.std(values_numpy) > 0:
                corr = np.corrcoef(values_orig, values_numpy)[0, 1]
                ax.text(0.05, 0.95, f'r = {corr:.4f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.suptitle('Statistical Validation: Original vs NumPy Results', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / "statistical_validation.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Statistical validation plot saved to: {plot_file}")


def main():
    """Main function to run the comprehensive comparison test."""
    print("🧪 Random Walk Simulator Comparison Test")
    print("="*80)
    
    # Create comparison instance
    comparison = SimulatorComparison()
    
    # Run comprehensive test
    results = comparison.run_comprehensive_test()
    
    # Final summary
    print("\n🎉 Comprehensive Comparison Complete!")
    print(f"📁 All results saved to: {comparison.results_dir}")
    
    # Check overall success
    all_passed = all(
        r['validation']['counts_equal'] and 
        r['validation']['initial_equal'] and
        r['validation']['final_equal']
        for r in results
    )
    
    if all_passed:
        print("✅ All tests PASSED - Both simulators produce identical results!")
    else:
        print("❌ Some tests FAILED - Simulators produce different results!")
        
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)