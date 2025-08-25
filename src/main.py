#!/usr/bin/env python3
"""
Complete end-to-end workflow for NPE on Random Walk model.
Generates data, trains model, and runs inference in a single script.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

# Project imports
from simulator import RandomWalkSimulator
from inference import RandomWalkNPE
from utils import check_device_availability, print_device_info, configure_warnings

# Configure warning filters
configure_warnings()


def main():
    parser = argparse.ArgumentParser(
        description='Complete NPE Random Walk workflow',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Simulation parameters
    parser.add_argument('--Lx', type=int, default=21,
                       help='Lattice width (columns)')
    parser.add_argument('--Ly', type=int, default=21,
                       help='Lattice height (rows)')
    parser.add_argument('--T', type=int, default=100,
                       help='Number of simulation time steps')
    parser.add_argument('--initial_region_half_width', type=int, default=None,
                       help='Half-width of initial region (default: Lx//4)')
    
    # Training data parameters
    parser.add_argument('--n_samples', type=int, default=10000,
                       help='Number of training samples')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to existing training data (skip generation if provided)')
    parser.add_argument('--save_data', type=str, default=None,
                       help='Path to save generated training data')
    
    # NPE Training parameters
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--hidden_features', type=int, default=128,
                       help='Hidden layer size')
    parser.add_argument('--num_transforms', type=int, default=5,
                       help='Number of coupling transforms')
    parser.add_argument('--validation_fraction', type=float, default=0.1,
                       help='Validation fraction')
    parser.add_argument('--stop_after_epochs', type=int, default=20,
                       help='Early stopping patience')
    
    # Inference parameters
    parser.add_argument('--theta_true', type=float, nargs=2, 
                       default=[0.3, 0.7],
                       help='True parameters for inference test [U P]')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of posterior samples')
    
    # Sequential NPE parameters
    parser.add_argument('--use_snpe', action='store_true',
                       help='Use Sequential Neural Posterior Estimation (SNPE)')
    parser.add_argument('--snpe_rounds', type=int, default=3,
                       help='Number of sequential rounds for SNPE')
    parser.add_argument('--samples_per_round', type=int, default=None,
                       help='Number of simulations per SNPE round (default: n_samples // snpe_rounds)')
    parser.add_argument('--convergence_threshold', type=float, default=0.01,
                       help='Convergence threshold for early stopping in SNPE')
    
    # General parameters
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device for training (auto: use CUDA if available, else CPU)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: results/workflow_TIMESTAMP)')
    
    # Skip steps (for partial runs)
    parser.add_argument('--skip_data', action='store_true',
                       help='Skip data generation (use existing data)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training (use existing model)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to existing model (if skipping training)')
    
    args = parser.parse_args()
    
    # Check device availability and set device
    if args.device == 'auto':
        recommended_device, device_info = check_device_availability()
        device = recommended_device
    else:
        device = args.device
        _, device_info = check_device_availability()
        
        # Handle device fallbacks
        if device == 'cuda' and not device_info['cuda_available']:
            print("⚠️  WARNING: CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/workflow_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    data_path = args.data_path or str(output_dir / "training_data.pkl")
    model_path = args.model_path or str(output_dir / "npe_model.pkl")
    results_dir = output_dir / "inference_results"
    results_dir.mkdir(exist_ok=True)
    
    # Determine training approach
    training_method = "Sequential NPE (SNPE)" if args.use_snpe else "Standard NPE"
    
    print(f"🚀 Starting {training_method} Random Walk Workflow")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Target parameters: U={args.theta_true[0]}, P={args.theta_true[1]}")
    
    if args.use_snpe:
        samples_per_round = args.samples_per_round or (args.n_samples // args.snpe_rounds)
        print(f"📊 SNPE Configuration:")
        print(f"   Rounds: {args.snpe_rounds}")
        print(f"   Samples per round: {samples_per_round}")
        print(f"   Convergence threshold: {args.convergence_threshold}")
    else:
        print(f"📊 Training samples: {args.n_samples}")
    
    print(f"🌱 Random seed: {args.seed}")
    print(f"📝 Note: PyTorch deprecation warnings are suppressed for cleaner output")
    print()
    
    # Print device information
    print_device_info(device, device_info)
    
    # Save configuration
    config_path = output_dir / "config.txt"
    with open(config_path, 'w') as f:
        f.write("NPE Random Walk Workflow Configuration\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        f.write("Arguments:\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")
        
        f.write(f"\nDevice Information:\n")
        f.write(f"  device_used: {device}\n")
        f.write(f"  cuda_available: {device_info['cuda_available']}\n")
        if device_info['cuda_available']:
            f.write(f"  cuda_device_name: {device_info['cuda_device_name']}\n")
            f.write(f"  cuda_memory_total_gb: {device_info['cuda_memory_total']:.1f}\n")
    
    total_start = time.time()
    
    # Initialize simulator
    print(f"\n📐 Setting up simulator (Lx={args.Lx}, Ly={args.Ly}, T={args.T})")
    simulator = RandomWalkSimulator(
        Lx=args.Lx,
        Ly=args.Ly,
        initial_region_half_width=args.initial_region_half_width
    )
    
    # Initialize NPE
    npe = RandomWalkNPE(device=device, seed=args.seed)
    
    # Step 0: Generate test observation (needed for SNPE)
    print(f"\n🎯 Generating test observation with true parameters U={args.theta_true[0]}, P={args.theta_true[1]}")
    true_theta = torch.tensor(args.theta_true, dtype=torch.float32)
    
    # Generate observed data using true parameters
    column_counts, initial_positions, final_positions = simulator.simulate(
        U=args.theta_true[0], 
        P=args.theta_true[1], 
        T=args.T,
        random_seed=args.seed + 1000
    )
    
    x_obs = torch.tensor(column_counts, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    print(f"   Observed data: {len(column_counts)} columns, {column_counts.sum()} total agents")
    
    # Step 1: Training workflow - different for NPE vs SNPE
    if not args.skip_training:
        start_time = time.time()
        
        # Setup neural network configuration
        neural_net_kwargs = {
            'hidden_features': args.hidden_features,
            'num_transforms': args.num_transforms
        }
        
        if args.use_snpe:
            # Sequential NPE workflow
            samples_per_round = args.samples_per_round or (args.n_samples // args.snpe_rounds)
            
            training_info = npe.train_sequential(
                simulator=simulator,
                n_rounds=args.snpe_rounds,
                n_simulations_per_round=samples_per_round,
                T=args.T,
                x_obs=x_obs,
                training_batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_num_epochs=args.max_epochs,
                validation_fraction=args.validation_fraction,
                stop_after_epochs=args.stop_after_epochs,
                neural_net_kwargs=neural_net_kwargs,
                convergence_threshold=args.convergence_threshold,
                random_seed=args.seed,
                output_dir=str(output_dir)
            )
            
        else:
            # Standard NPE workflow
            if not args.skip_data and args.data_path is None:
                print(f"\n📊 Generating {args.n_samples} training samples...")
                
                save_path = args.save_data or data_path
                theta, x = npe.generate_training_data(
                    simulator=simulator,
                    n_simulations=args.n_samples,
                    T=args.T,
                    output_path=save_path,
                    random_seed=args.seed
                )
                
                print(f"✅ Data generation completed")
                print(f"   Parameter range: U ∈ [{theta[:, 0].min():.3f}, {theta[:, 0].max():.3f}], "
                      f"P ∈ [{theta[:, 1].min():.3f}, {theta[:, 1].max():.3f}]")
                print(f"   Observation range: [{x.min():.0f}, {x.max():.0f}] agents per column")
                
            else:
                print(f"\n📂 Loading training data from {data_path}")
                theta, x, metadata = RandomWalkNPE.load_training_data(data_path)
                print(f"✅ Loaded {len(theta)} training samples")
                print(f"   Simulation metadata: {metadata}")
            
            # Train standard NPE
            print(f"\n🧠 Training NPE model...")
            training_info = npe.train(
                theta=theta,
                x=x,
                training_batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_num_epochs=args.max_epochs,
                validation_fraction=args.validation_fraction,
                stop_after_epochs=args.stop_after_epochs,
                neural_net_kwargs=neural_net_kwargs
            )
        
        elapsed = time.time() - start_time
        print(f"✅ Training completed in {elapsed:.1f} seconds")
        
        # Save model
        metadata = {
            'training_approach': 'SNPE' if args.use_snpe else 'NPE',
            'lattice_size': (args.Lx, args.Ly),
            'time_steps': args.T,
            'training_time': elapsed,
            'training_epochs': args.max_epochs
        }
        
        if args.use_snpe:
            metadata.update({
                'snpe_rounds': args.snpe_rounds,
                'samples_per_round': samples_per_round,
                'convergence_threshold': args.convergence_threshold,
                'rounds_completed': training_info['total_rounds_completed'],
                'converged': training_info['converged']
            })
        else:
            metadata['training_samples'] = len(theta)
            
        npe.save_model(model_path, metadata)
        
    else:
        print(f"\n📂 Loading trained model from {model_path}")
        npe = RandomWalkNPE.load_model(model_path, device=device)
        print("✅ Model loaded successfully")
    
    # Step 2: Perform inference
    print(f"\n🔍 Sampling {args.num_samples} posterior samples...")
    start_time = time.time()
    
    posterior_samples = npe.sample_posterior(x_obs, num_samples=args.num_samples)
    
    elapsed = time.time() - start_time
    print(f"✅ Posterior sampling completed in {elapsed:.1f} seconds")
    
    # Compute summary statistics
    samples_np = posterior_samples.numpy()
    U_mean, U_std = samples_np[:, 0].mean(), samples_np[:, 0].std()
    P_mean, P_std = samples_np[:, 1].mean(), samples_np[:, 1].std()
    
    print(f"\n📈 Posterior Summary:")
    print(f"   U: {U_mean:.3f} ± {U_std:.3f} (true: {args.theta_true[0]})")
    print(f"   P: {P_mean:.3f} ± {P_std:.3f} (true: {args.theta_true[1]})")
    
    # Compute credible intervals
    U_ci = np.percentile(samples_np[:, 0], [2.5, 97.5])
    P_ci = np.percentile(samples_np[:, 1], [2.5, 97.5])
    
    print(f"   95% Credible Intervals:")
    print(f"   U: [{U_ci[0]:.3f}, {U_ci[1]:.3f}]")
    print(f"   P: [{P_ci[0]:.3f}, {P_ci[1]:.3f}]")
    
    # Check if true values are within credible intervals
    U_in_ci = U_ci[0] <= args.theta_true[0] <= U_ci[1]
    P_in_ci = P_ci[0] <= args.theta_true[1] <= P_ci[1]
    print(f"   True values in CI: U={U_in_ci}, P={P_in_ci}")
    
    # Step 5: Create visualizations
    print(f"\n📊 Creating visualizations...")
    
    # Plot posterior samples
    fig1 = npe.plot_posterior_samples(posterior_samples, true_theta)
    fig1.savefig(results_dir / "posterior_marginals.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # Plot pairwise relationships
    fig2 = npe.plot_pairwise(posterior_samples, true_theta)
    fig2.savefig(results_dir / "posterior_pairwise.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # Plot observed data
    from simulator import plot_column_counts, plot_simulation_comparison
    
    fig3 = plot_column_counts(column_counts, args.Lx, title="Observed Data")
    fig3.savefig(results_dir / "observed_data.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # Plot simulation comparison
    fig4 = plot_simulation_comparison(
        initial_positions, final_positions, column_counts,
        args.Lx, args.Ly, args.theta_true[0], args.theta_true[1], args.T
    )
    fig4.savefig(results_dir / "simulation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    # Save results
    results = {
        'posterior_samples': posterior_samples.numpy(),
        'true_parameters': args.theta_true,
        'observed_data': column_counts,
        'summary_statistics': {
            'U_mean': U_mean, 'U_std': U_std, 'U_ci': U_ci,
            'P_mean': P_mean, 'P_std': P_std, 'P_ci': P_ci
        },
        'metadata': {
            'n_samples': args.num_samples,
            'lattice_size': (args.Lx, args.Ly),
            'time_steps': args.T,
            'seed': args.seed,
            'training_approach': 'SNPE' if args.use_snpe else 'NPE'
        }
    }
    
    # Add SNPE-specific results if applicable
    if args.use_snpe and 'training_info' in locals():
        results['snpe_results'] = {
            'rounds_completed': training_info.get('total_rounds_completed', 0),
            'converged': training_info.get('converged', False),
            'final_convergence_metric': training_info.get('final_convergence_metric'),
            'round_results': npe.get_round_results()
        }
    
    import pickle
    with open(results_dir / "results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    # Workflow completed
    total_elapsed = time.time() - total_start
    
    print(f"\n🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
    print(f"⏱️  Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"📁 All results saved in: {output_dir}")
    print(f"\n📊 Files created:")
    print(f"   📈 Training data: {data_path}")
    print(f"   🧠 Trained model: {model_path}")
    print(f"   📊 Inference results: {results_dir}")
    print(f"   ⚙️  Configuration: {config_path}")
    print(f"\n🖼️  Visualization files:")
    print(f"   - posterior_marginals.png")
    print(f"   - posterior_pairwise.png") 
    print(f"   - observed_data.png")
    print(f"   - simulation_comparison.png")
    
    # Print SNPE-specific summary if applicable
    if args.use_snpe and 'training_info' in locals():
        print(f"\n🔄 SNPE Summary:")
        print(f"   Rounds completed: {training_info.get('total_rounds_completed', 0)}")
        print(f"   Converged: {'Yes' if training_info.get('converged', False) else 'No'}")
        if training_info.get('final_convergence_metric'):
            print(f"   Final convergence metric: {training_info['final_convergence_metric']:.6f}")
        print(f"   Round results saved in: {output_dir}/round_*")
    
    print(f"\n🔍 Next steps:")
    print(f"   - Examine posterior plots for parameter estimates")
    if args.use_snpe:
        print(f"   - Review round-by-round convergence in {output_dir}/round_* directories")
        print(f"   - Try different convergence thresholds or number of rounds")
    else:
        print(f"   - Try SNPE with --use_snpe for potentially better inference")
    print(f"   - Try different lattice sizes or time steps")
    print(f"   - Test with different true parameter values")
    print(f"   - Analyze model performance with validation data")


if __name__ == '__main__':
    main()