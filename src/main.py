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
from simulator import RandomWalkSimulator, ExclusionRandomWalkSimulator
from inference import RandomWalkNPE
from models import get_model_config
from utils import check_device_availability, print_device_info, configure_warnings
from predict import posterior_predictive_sample, compute_prediction_intervals, plot_prediction_intervals

# Configure warning filters
configure_warnings()

# Default true parameters per model (used when --theta_true is not given)
DEFAULT_THETA_TRUE = {
    'original': [0.3, 0.7],          # U, P
    'A':        [0.5, 1.0, 0.5],     # U, P, rho
    'B':        [1.0, 0.001],         # P, R
    'C':        [1.0, 0.5, 0.001],   # P, rho, R
}


def main():
    parser = argparse.ArgumentParser(
        description='Complete NPE Random Walk workflow',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model selection
    parser.add_argument('--model', type=str, default='original',
                       choices=['original', 'A', 'B', 'C'],
                       help='Random walk model variant')
    parser.add_argument('--fixed_U', type=float, default=None,
                       help='Override fixed U value for growth models B/C (default from model config)')

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
    parser.add_argument('--theta_true', type=float, nargs='+',
                       default=None,
                       help='True parameters for inference test (order matches model param_names)')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of posterior samples')
    parser.add_argument('--n_pred_samples', type=int, default=None,
                       help='Number of posterior predictive samples (default: same as num_samples)')

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

    # Parallelism
    parser.add_argument('--n_workers', type=int, default=1,
                       help='Number of parallel workers for simulation (1 = sequential)')

    # Skip steps (for partial runs)
    parser.add_argument('--skip_data', action='store_true',
                       help='Skip data generation (use existing data)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training (use existing model)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to existing model (if skipping training)')

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Model configuration
    # ------------------------------------------------------------------
    model_name = args.model
    cfg = get_model_config(model_name) if model_name != 'original' else None

    # Override fixed_U if requested
    if args.fixed_U is not None and cfg is not None and 'U' in cfg.fixed_params:
        cfg.fixed_params['U'] = args.fixed_U

    # Resolve true parameters
    if args.theta_true is not None:
        theta_true_list = args.theta_true
    else:
        theta_true_list = DEFAULT_THETA_TRUE[model_name]

    param_names = cfg.param_names if cfg is not None else ['U', 'P']
    n_params = len(param_names)

    if len(theta_true_list) != n_params:
        parser.error(
            f"Model '{model_name}' expects {n_params} parameters "
            f"({', '.join(param_names)}), got {len(theta_true_list)}"
        )

    # Check device availability and set device
    if args.device == 'auto':
        recommended_device, device_info = check_device_availability()
        device = recommended_device
    else:
        device = args.device
        _, device_info = check_device_availability()

        # Handle device fallbacks
        if device == 'cuda' and not device_info['cuda_available']:
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/workflow_{model_name}_{timestamp}")
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

    print(f"Starting {training_method} Random Walk Workflow  [model={model_name}]")
    print(f"Output directory: {output_dir}")
    theta_str = ', '.join(f'{n}={v}' for n, v in zip(param_names, theta_true_list))
    print(f"Target parameters: {theta_str}")

    if cfg is not None and cfg.fixed_params:
        fixed_str = ', '.join(f'{k}={v}' for k, v in cfg.fixed_params.items())
        print(f"Fixed parameters: {fixed_str}")

    if args.use_snpe:
        samples_per_round = args.samples_per_round or (args.n_samples // args.snpe_rounds)
        print(f"SNPE Configuration:")
        print(f"   Rounds: {args.snpe_rounds}")
        print(f"   Samples per round: {samples_per_round}")
        print(f"   Convergence threshold: {args.convergence_threshold}")
    else:
        print(f"Training samples: {args.n_samples}")

    print(f"Random seed: {args.seed}")
    if args.n_workers > 1:
        print(f"Parallel workers: {args.n_workers}")
    print()

    # Print device information
    print_device_info(device, device_info)

    # Save configuration
    config_path = output_dir / "config.txt"
    with open(config_path, 'w') as f:
        f.write("NPE Random Walk Workflow Configuration\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Model: {model_name}\n\n")

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

    # ------------------------------------------------------------------
    # Initialize simulator
    # ------------------------------------------------------------------
    print(f"\nSetting up simulator (Lx={args.Lx}, Ly={args.Ly}, T={args.T})")

    if model_name == 'original':
        simulator = RandomWalkSimulator(
            Lx=args.Lx,
            Ly=args.Ly,
            initial_region_half_width=args.initial_region_half_width
        )
    else:
        simulator = ExclusionRandomWalkSimulator(
            Lx=args.Lx,
            Ly=args.Ly,
            initial_region_half_width=args.initial_region_half_width,
            has_bias=cfg.has_bias,
            has_growth=cfg.has_growth,
        )

    # ------------------------------------------------------------------
    # Initialize NPE
    # ------------------------------------------------------------------
    npe = RandomWalkNPE(device=device, seed=args.seed, model_config=cfg)

    # ------------------------------------------------------------------
    # Generate test observation
    # ------------------------------------------------------------------
    print(f"\nGenerating test observation with true parameters: {theta_str}")
    true_theta = torch.tensor(theta_true_list, dtype=torch.float32)

    if model_name == 'original':
        column_counts, initial_positions, final_positions = simulator.simulate(
            U=theta_true_list[0],
            P=theta_true_list[1],
            T=args.T,
            random_seed=args.seed + 1000
        )
    else:
        # Build full theta_dict including fixed params
        theta_dict_obs = dict(zip(param_names, theta_true_list))
        theta_dict_obs.update(cfg.fixed_params)
        column_counts, _, _ = simulator.simulate(
            theta_dict_obs, T=args.T, random_seed=args.seed + 1000
        )

    x_obs = torch.tensor(column_counts, dtype=torch.float32).unsqueeze(0)
    print(f"   Observed data: {len(column_counts)} columns, {column_counts.sum()} total agents")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    if not args.skip_training:
        start_time = time.time()

        neural_net_kwargs = {
            'hidden_features': args.hidden_features,
            'num_transforms': args.num_transforms
        }

        if args.use_snpe:
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
                output_dir=str(output_dir),
                n_workers=args.n_workers
            )

        else:
            # Standard NPE workflow
            if not args.skip_data and args.data_path is None:
                print(f"\nGenerating {args.n_samples} training samples...")

                save_path = args.save_data or data_path
                theta, x = npe.generate_training_data(
                    simulator=simulator,
                    n_simulations=args.n_samples,
                    T=args.T,
                    output_path=save_path,
                    random_seed=args.seed,
                    n_workers=args.n_workers
                )

                print(f"Data generation completed")
                for pidx, pname in enumerate(param_names):
                    lo, hi = theta[:, pidx].min(), theta[:, pidx].max()
                    print(f"   {pname} range: [{lo:.3f}, {hi:.3f}]")
                print(f"   Observation range: [{x.min():.0f}, {x.max():.0f}] agents per column")

            else:
                print(f"\nLoading training data from {data_path}")
                theta, x, metadata = RandomWalkNPE.load_training_data(data_path)
                print(f"Loaded {len(theta)} training samples")

            # Train standard NPE
            print(f"\nTraining NPE model...")
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
        print(f"Training completed in {elapsed:.1f} seconds")

        # Save model
        metadata = {
            'model_name': model_name,
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
        print(f"\nLoading trained model from {model_path}")
        npe = RandomWalkNPE.load_model(model_path, device=device)
        print("Model loaded successfully")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    print(f"\nSampling {args.num_samples} posterior samples...")
    start_time = time.time()

    posterior_samples = npe.sample_posterior(x_obs, num_samples=args.num_samples)

    elapsed = time.time() - start_time
    print(f"Posterior sampling completed in {elapsed:.1f} seconds")

    # Compute summary statistics
    samples_np = posterior_samples.cpu().numpy()

    print(f"\nPosterior Summary:")
    for pidx, pname in enumerate(param_names):
        pmean = samples_np[:, pidx].mean()
        pstd = samples_np[:, pidx].std()
        pci = np.percentile(samples_np[:, pidx], [2.5, 97.5])
        true_val = theta_true_list[pidx]
        in_ci = pci[0] <= true_val <= pci[1]
        print(f"   {pname}: {pmean:.3f} +/- {pstd:.3f} (true: {true_val})  "
              f"95% CI: [{pci[0]:.3f}, {pci[1]:.3f}]  in CI: {in_ci}")

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    print(f"\nCreating visualizations...")

    # Posterior marginals
    fig1 = npe.plot_posterior_samples(posterior_samples, true_theta)
    fig1.savefig(results_dir / "posterior_marginals.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # Pairwise relationships
    fig2 = npe.plot_pairwise(posterior_samples, true_theta)
    fig2.savefig(results_dir / "posterior_pairwise.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # Column counts
    from simulator import plot_column_counts
    fig3 = plot_column_counts(column_counts, args.Lx, title="Observed Data")
    fig3.savefig(results_dir / "observed_data.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)

    # Simulation comparison plot (only for original model which returns position lists)
    if model_name == 'original':
        from simulator import plot_simulation_comparison
        fig4 = plot_simulation_comparison(
            initial_positions, final_positions, column_counts,
            args.Lx, args.Ly, theta_true_list[0], theta_true_list[1], args.T
        )
        fig4.savefig(results_dir / "simulation_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig4)

    # ------------------------------------------------------------------
    # Save posterior samples
    # ------------------------------------------------------------------
    print(f"\nSaving posterior samples...")
    posterior_data = {
        'posterior_samples': posterior_samples.cpu().numpy(),
        'true_parameters': theta_true_list,
        'param_names': param_names,
        'metadata': {
            'model_name': model_name,
            'n_samples': args.num_samples,
            'lattice_size': (args.Lx, args.Ly),
            'time_steps': args.T,
            'seed': args.seed,
            'training_approach': 'SNPE' if args.use_snpe else 'NPE',
            'sampling_time': elapsed
        }
    }

    import pickle
    with open(results_dir / "posterior_samples.pkl", 'wb') as f:
        pickle.dump(posterior_data, f)

    # ------------------------------------------------------------------
    # Posterior predictive sampling
    # ------------------------------------------------------------------
    print(f"\nGenerating posterior predictive samples...")
    pred_start_time = time.time()

    n_pred = args.n_pred_samples or args.num_samples
    predictions = posterior_predictive_sample(
        posterior_samples=posterior_samples.cpu().numpy(),
        simulator=simulator,
        T=args.T,
        n_pred_samples=n_pred,
        random_seed=args.seed + 2000,
        param_names=param_names,
        fixed_params=cfg.fixed_params if cfg is not None else {},
        n_workers=args.n_workers,
    )

    prediction_results = compute_prediction_intervals(predictions)

    fig5 = plot_prediction_intervals(
        prediction_results=prediction_results,
        observed_data=column_counts,
        Lx=args.Lx,
        title_suffix=f" (T={args.T})"
    )
    fig5.savefig(results_dir / "prediction_intervals.png", dpi=150, bbox_inches='tight')
    plt.close(fig5)

    # Save prediction results
    predictive_data = {
        'prediction_results': prediction_results,
        'input_metadata': {
            'posterior_samples_shape': posterior_samples.shape,
            'true_parameters': theta_true_list,
            'param_names': param_names,
            'simulation_params': {
                'Lx': args.Lx, 'Ly': args.Ly, 'T': args.T,
                'initial_region_half_width': args.initial_region_half_width
            },
            'prediction_params': {
                'n_pred_samples': n_pred,
                'seed': args.seed + 2000
            }
        },
        'observed_data': column_counts
    }

    with open(results_dir / "predictive_results.pkl", 'wb') as f:
        pickle.dump(predictive_data, f)

    pred_elapsed = time.time() - pred_start_time
    print(f"Posterior predictive sampling completed in {pred_elapsed:.1f} seconds")

    # ------------------------------------------------------------------
    # Save full results
    # ------------------------------------------------------------------
    summary_stats = {}
    for pidx, pname in enumerate(param_names):
        summary_stats[f'{pname}_mean'] = float(samples_np[:, pidx].mean())
        summary_stats[f'{pname}_std'] = float(samples_np[:, pidx].std())
        summary_stats[f'{pname}_ci'] = np.percentile(samples_np[:, pidx], [2.5, 97.5]).tolist()

    results = {
        'posterior_samples': posterior_samples.cpu().numpy(),
        'true_parameters': theta_true_list,
        'param_names': param_names,
        'observed_data': column_counts,
        'summary_statistics': summary_stats,
        'metadata': {
            'model_name': model_name,
            'n_samples': args.num_samples,
            'lattice_size': (args.Lx, args.Ly),
            'time_steps': args.T,
            'seed': args.seed,
            'training_approach': 'SNPE' if args.use_snpe else 'NPE'
        }
    }

    if args.use_snpe and 'training_info' in locals():
        results['snpe_results'] = {
            'rounds_completed': training_info.get('total_rounds_completed', 0),
            'converged': training_info.get('converged', False),
            'final_convergence_metric': training_info.get('final_convergence_metric'),
            'round_results': npe.get_round_results()
        }

    with open(results_dir / "results.pkl", 'wb') as f:
        pickle.dump(results, f)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - total_start

    print(f"\nWORKFLOW COMPLETED SUCCESSFULLY!")
    print(f"Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"All results saved in: {output_dir}")

    if args.use_snpe and 'training_info' in locals():
        print(f"\nSNPE Summary:")
        print(f"   Rounds completed: {training_info.get('total_rounds_completed', 0)}")
        print(f"   Converged: {'Yes' if training_info.get('converged', False) else 'No'}")

    print(f"\nPosterior Predictive Results:")
    global_stats = prediction_results['global_stats']
    print(f"   Total agents: {global_stats['total_agents_mean']:.1f} +/- {global_stats['total_agents_std']:.1f}")
    observed_total = np.sum(column_counts)
    predictions_total = np.sum(predictions, axis=1)
    p2_5 = np.percentile(predictions_total, 2.5)
    p97_5 = np.percentile(predictions_total, 97.5)
    in_interval = p2_5 <= observed_total <= p97_5
    print(f"   Observed total: {observed_total}")
    print(f"   95% interval: [{p2_5:.1f}, {p97_5:.1f}]")
    print(f"   Observed in interval: {'Yes' if in_interval else 'No'}")


if __name__ == '__main__':
    main()
