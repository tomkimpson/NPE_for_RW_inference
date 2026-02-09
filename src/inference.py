"""
Inference module for Neural Posterior Estimation (NPE).

This module contains the NPE training pipeline, neural network
architectures, and posterior prediction functionality for the
random walk parameter inference problem.
"""

import numpy as np
import torch
import pickle
import warnings
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import time
import matplotlib.pyplot as plt

# Import centralized warning configuration
from utils import configure_warnings
configure_warnings()

# SBI imports
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

from simulator import RandomWalkSimulator, ExclusionRandomWalkSimulator
from models import ModelConfig
from cnn_utils import create_spatial_embedding_net, create_dual_branch_embedding_net


# --- Pool-initializer pattern for parallel simulation ---
_worker_simulator = None


def _init_worker(sim_class_name, sim_kwargs, cpu_affinity=None):
    """Initializer called once per worker process to create the simulator."""
    # Pin each worker to 1 BLAS/OpenMP thread
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = '1'

    # Restore CPU affinity (torch/CUDA can pin forked children to 1 core)
    if cpu_affinity is not None:
        try:
            os.sched_setaffinity(0, cpu_affinity)
        except (OSError, AttributeError):
            pass

    global _worker_simulator
    if sim_class_name == 'ExclusionRandomWalkSimulator':
        _worker_simulator = ExclusionRandomWalkSimulator(**sim_kwargs)
    else:
        _worker_simulator = RandomWalkSimulator(**sim_kwargs)


def _run_single_sim(sim_args):
    """Worker for parallel simulation — reads simulator from process global."""
    idx, param_values, param_names, fixed_params, use_exclusion, T, random_seed, use_2d_output = sim_args
    theta_dict = dict(zip(param_names, param_values))
    if fixed_params:
        theta_dict.update(fixed_params)
    if use_exclusion:
        obs, _, _ = _worker_simulator.simulate(theta_dict, T, random_seed=random_seed, use_2d_output=use_2d_output)
    else:
        obs, _, _ = _worker_simulator.simulate(theta_dict['U'], theta_dict['P'], T, random_seed=random_seed, use_2d_output=use_2d_output)
    return (idx, obs)


def _run_single_sim_sequential(sim_args):
    """Worker for sequential simulation — receives simulator explicitly."""
    idx, param_values, simulator, param_names, fixed_params, use_exclusion, T, random_seed, use_2d_output = sim_args
    theta_dict = dict(zip(param_names, param_values))
    if fixed_params:
        theta_dict.update(fixed_params)
    if use_exclusion:
        obs, _, _ = simulator.simulate(theta_dict, T, random_seed=random_seed, use_2d_output=use_2d_output)
    else:
        obs, _, _ = simulator.simulate(theta_dict['U'], theta_dict['P'], T, random_seed=random_seed, use_2d_output=use_2d_output)
    return (idx, obs)


class RandomWalkNPE:
    """Sequential Neural Posterior Estimation for Random Walk parameter inference."""
    
    def __init__(self,
                 device: str = 'cpu',
                 seed: Optional[int] = None,
                 model_config: Optional[ModelConfig] = None,
                 use_2d_data: bool = False,
                 spatial_dims: Optional[Tuple[int, int]] = None):
        """
        Initialize SNPE inference.

        Parameters:
        -----------
        device : str
            Device for training ('cpu' or 'cuda')
        seed : int, optional
            Random seed
        model_config : ModelConfig, optional
            Model configuration.  If None, uses hardcoded [U, P] prior
            for backward compatibility with the original model.
        use_2d_data : bool
            If True, use 2D spatial grids (Ly, Lx) with CNN embedding.
        spatial_dims : tuple of (int, int), optional
            (Ly, Lx) for 2D data.  Required when use_2d_data=True.
        """
        self.device = device
        self.model_config = model_config
        self.use_2d_data = use_2d_data
        self.spatial_dims = spatial_dims

        if use_2d_data and spatial_dims is None:
            raise ValueError("spatial_dims must be provided when use_2d_data=True")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if model_config is not None:
            self.param_names = model_config.param_names
            self.prior = BoxUniform(
                low=torch.tensor(model_config.prior_low, dtype=torch.float32, device=device),
                high=torch.tensor(model_config.prior_high, dtype=torch.float32, device=device),
            )
        else:
            # Backward-compatible default: original model [U, P]
            self.param_names = ['U', 'P']
            self.prior = BoxUniform(
                low=torch.tensor([0.01, 0.0], device=device),
                high=torch.tensor([1.0, 1.0], device=device)
            )

        self.inference = None
        self.posterior = None

        # Sequential training attributes
        self.posteriors_by_round = []  # Store posterior from each round
        self.training_history = []  # Store training info from each round
        self.current_round = 0
        
    def setup_inference(self, x_dim: int = None, **kwargs):
        """
        Set up SBI inference object.

        Parameters:
        -----------
        x_dim : int, optional
            Dimension of 1D observations (number of columns).
            Ignored when use_2d_data=True.
        **kwargs : additional arguments for SNPE
            - disable_sbi_standardization : bool
                If True, disable sbi's z-score standardization of observations.
                This allows the CNN to see raw density information, which is
                critical for inferring parameters like P (proliferation rate).
        """
        # Default neural network configuration
        neural_net_kwargs = {
            'hidden_features': 256 if self.use_2d_data else 128,
            'num_transforms': 5,
        }

        if self.use_2d_data:
            Ly, Lx = self.spatial_dims
            use_dual_branch = kwargs.get('cnn_dual_branch', False)

            if use_dual_branch:
                # Dual-branch architecture: 1D for P/density, 2D for rho/U spatial patterns
                spatial_embedding = create_dual_branch_embedding_net(
                    input_height=Ly,
                    input_width=Lx,
                    output_dim=neural_net_kwargs['hidden_features'],
                    dropout=kwargs.get('cnn_dropout', 0.05),
                    use_spatial_pyramid=True,
                )
                print("   Note: Using dual-branch CNN (1D for P, 2D for rho/U)")
            else:
                # Standard single-branch CNN
                spatial_embedding = create_spatial_embedding_net(
                    input_height=Ly,
                    input_width=Lx,
                    output_dim=neural_net_kwargs['hidden_features'],
                    dropout=kwargs.get('cnn_dropout', 0.05),
                    normalize_input=kwargs.get('cnn_normalize_input', True),
                    use_auxiliary_features=kwargs.get('cnn_use_auxiliary_features', False),
                    use_density_channels=kwargs.get('cnn_use_density_channels', False),
                    use_spatial_pyramid=kwargs.get('cnn_use_spatial_pyramid', False),
                )
            neural_net_kwargs['embedding_net'] = spatial_embedding
        else:
            neural_net_kwargs['embedding_net'] = torch.nn.Identity()

        neural_net_kwargs.update(kwargs.get('neural_net_kwargs', {}))

        # Check if sbi standardization should be disabled
        # When True, the CNN sees raw (unstandardized) observations, allowing
        # density-preserving normalization to work correctly for P inference.
        disable_sbi_standardization = kwargs.get('disable_sbi_standardization', False)
        if disable_sbi_standardization:
            neural_net_kwargs['z_score_x'] = "none"
            print("   Note: sbi z-score standardization disabled (CNN will see raw observations)")

        # Create neural posterior estimator
        neural_posterior = posterior_nn(
            model='nsf',  # Neural Spline Flow
            **neural_net_kwargs
        )

        # Filter out CNN-specific and neural_net kwargs before passing to SNPE
        excluded_kwargs = {'neural_net_kwargs', 'cnn_dropout', 'cnn_normalize_input', 'cnn_use_auxiliary_features',
                          'cnn_use_density_channels', 'cnn_use_spatial_pyramid', 'disable_sbi_standardization',
                          'cnn_dual_branch'}
        self.inference = SNPE(
            prior=self.prior,
            density_estimator=neural_posterior,
            device=self.device,
            **{k: v for k, v in kwargs.items() if k not in excluded_kwargs}
        )
        
    def generate_training_data(
        self,
        simulator,
        n_simulations: int,
        T: int,
        output_path: Optional[str] = None,
        prior_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        random_seed: Optional[int] = None,
        proposal_distribution = None,
        n_workers: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training data for SNPE by running simulations.

        Parameters:
        -----------
        simulator : RandomWalkSimulator or ExclusionRandomWalkSimulator
            Configured simulator instance
        n_simulations : int
            Number of parameter-observation pairs to generate
        T : int
            Number of time steps for each simulation
        output_path : str, optional
            Path to save training data
        prior_bounds : Dict[str, Tuple[float, float]], optional
            Prior bounds for parameters (used only if proposal_distribution is None
            and no model_config is set)
        random_seed : int, optional
            Random seed for reproducibility
        proposal_distribution : optional
            Proposal distribution for sequential rounds (if None, uses priors)

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            - parameters: Tensor of shape (n_simulations, n_params)
            - observations: Tensor of shape (n_simulations, Lx) containing column counts
        """
        if n_simulations <= 0:
            raise ValueError("Number of simulations must be positive")
        if T < 0:
            raise ValueError("Number of time steps must be non-negative")

        use_exclusion = isinstance(simulator, ExclusionRandomWalkSimulator)
        n_params = len(self.param_names)
        cfg = self.model_config

        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        # Build prior_bounds from model_config if available
        if prior_bounds is None:
            if cfg is not None:
                prior_bounds = {
                    name: (lo, hi)
                    for name, lo, hi in zip(cfg.param_names, cfg.prior_low, cfg.prior_high)
                }
            else:
                prior_bounds = {
                    'U': (0.01, 1.0),
                    'P': (0.0, 1.0)
                }

        if n_workers > 1:
            print(f"Generating {n_simulations} training simulations using {n_workers} workers...")
        else:
            print(f"Generating {n_simulations} training simulations...")

        # Initialize storage
        parameters = np.zeros((n_simulations, n_params))
        if self.use_2d_data:
            Ly, Lx = self.spatial_dims
            observations = np.zeros((n_simulations, Ly, Lx))
        else:
            observations = np.zeros((n_simulations, simulator.Lx))

        fixed_params = cfg.fixed_params if cfg is not None else {}

        # Pre-generate all parameter samples
        if proposal_distribution is not None:
            print(f"   Drawing {n_simulations} parameter samples from proposal distribution...")
            theta_samples = proposal_distribution.sample((n_simulations,))
            theta_samples_np = theta_samples.cpu().numpy()
            all_param_values = [
                [float(theta_samples_np[i, k]) for k in range(n_params)]
                for i in range(n_simulations)
            ]
        else:
            all_param_values = [
                [np.random.uniform(*prior_bounds[name]) for name in self.param_names]
                for i in range(n_simulations)
            ]

        # Dispatch simulations
        use_2d = self.use_2d_data
        if n_workers > 1:
            # Build argument tuples WITHOUT simulator (workers get it from global)
            sim_args_list = []
            for i in range(n_simulations):
                seed_i = random_seed + i if random_seed is not None else None
                sim_args_list.append((
                    i, all_param_values[i], self.param_names,
                    fixed_params, use_exclusion, T, seed_i, use_2d
                ))

            # Build kwargs to reconstruct simulator in each worker
            sim_class_name = type(simulator).__name__
            sim_kwargs = {
                'Lx': simulator.Lx,
                'Ly': simulator.Ly,
                'initial_region_half_width': simulator.initial_region_half_width,
            }
            if sim_class_name == 'ExclusionRandomWalkSimulator':
                sim_kwargs.update({
                    'has_bias': simulator.has_bias,
                    'has_growth': simulator.has_growth,
                    'Delta': simulator.Delta,
                    'tau': simulator.tau,
                })

            ctx = mp.get_context('fork')
            try:
                parent_cpus = set(os.sched_getaffinity(0))
            except (OSError, AttributeError):
                parent_cpus = None
            t0 = time.time()
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(sim_class_name, sim_kwargs, parent_cpus),
            ) as executor:
                futures = {executor.submit(_run_single_sim, args): args[0] for args in sim_args_list}
                done_count = 0
                log_interval = max(1, n_simulations // 10)
                for future in as_completed(futures):
                    idx, column_counts = future.result()
                    observations[idx] = column_counts
                    parameters[idx] = all_param_values[idx]
                    done_count += 1
                    if done_count % log_interval == 0 or done_count == n_simulations:
                        elapsed = time.time() - t0
                        rate = done_count / elapsed if elapsed > 0 else 0
                        print(f"   [{done_count}/{n_simulations}] {elapsed:.1f}s elapsed ({rate:.1f} sims/s)")
        else:
            # Sequential path — pass simulator explicitly, no pool overhead
            sim_args_list = []
            for i in range(n_simulations):
                seed_i = random_seed + i if random_seed is not None else None
                sim_args_list.append((
                    i, all_param_values[i], simulator, self.param_names,
                    fixed_params, use_exclusion, T, seed_i, use_2d
                ))

            t0 = time.time()
            log_interval = max(1, n_simulations // 10)
            for done_count, args in enumerate(sim_args_list, 1):
                idx, column_counts = _run_single_sim_sequential(args)
                observations[idx] = column_counts
                parameters[idx] = all_param_values[idx]
                if done_count % log_interval == 0 or done_count == n_simulations:
                    elapsed = time.time() - t0
                    rate = done_count / elapsed if elapsed > 0 else 0
                    print(f"   [{done_count}/{n_simulations}] {elapsed:.1f}s elapsed ({rate:.1f} sims/s)")

        # Convert to tensors
        theta = torch.tensor(parameters, dtype=torch.float32)
        x = torch.tensor(observations, dtype=torch.float32)

        # Save data if requested
        if output_path is not None:
            data = {
                'parameters': theta,
                'observations': x,
                'metadata': {
                    'n_simulations': n_simulations,
                    'T': T,
                    'prior_bounds': prior_bounds,
                    'Lx': simulator.Lx,
                    'Ly': simulator.Ly,
                    'initial_region_half_width': simulator.initial_region_half_width,
                    'param_names': self.param_names,
                }
            }

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            if not output_path.endswith('.pkl'):
                output_path = output_path + '.pkl'

            with open(output_path, 'wb') as f:
                pickle.dump(data, f)

            print(f"Training data saved to {output_path}")

        return theta, x
    
    @staticmethod
    def load_training_data(data_path: str) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Load training data from pickle file.
        
        Parameters:
        -----------
        data_path : str
            Path to training data pickle file
            
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor, Dict]
            Parameters, observations, and metadata
        """
        # Ensure .pkl extension
        if not data_path.endswith('.pkl'):
            data_path = data_path + '.pkl'
            
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        return data['parameters'], data['observations'], data['metadata']
        
    def train(self, 
              theta: torch.Tensor, 
              x: torch.Tensor,
              training_batch_size: int = 512,
              learning_rate: float = 1e-4,
              max_num_epochs: int = 100,
              validation_fraction: float = 0.1,
              stop_after_epochs: int = 20,
              neural_net_kwargs: Optional[Dict[str, Any]] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train neural posterior estimator.
        
        Parameters:
        -----------
        theta : torch.Tensor of shape (n_samples, 2)
            Parameter vectors [U, P]
        x : torch.Tensor of shape (n_samples, x_dim)
            Observations (column counts)
        training_batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate
        max_num_epochs : int
            Maximum training epochs
        validation_fraction : float
            Fraction of data for validation
        stop_after_epochs : int
            Early stopping patience
        neural_net_kwargs : Dict[str, Any], optional
            Neural network configuration arguments
        **kwargs : additional training arguments
            
        Returns:
        --------
        training_info : dict
            Training information and losses
        """
        if self.inference is None:
            # Pass neural network configuration to setup
            setup_kwargs = {}
            if neural_net_kwargs is not None:
                setup_kwargs['neural_net_kwargs'] = neural_net_kwargs
            # Pass CNN-specific kwargs through to setup_inference
            for cnn_kwarg in ['cnn_dropout', 'cnn_normalize_input', 'cnn_use_auxiliary_features',
                              'cnn_use_density_channels', 'cnn_use_spatial_pyramid', 'disable_sbi_standardization',
                              'cnn_dual_branch']:
                if cnn_kwarg in kwargs:
                    setup_kwargs[cnn_kwarg] = kwargs.pop(cnn_kwarg)
            self.setup_inference(x_dim=x.shape[1], **setup_kwargs)

        print(f"Training NPE with {len(theta)} samples...")
        print(f"Parameter shape: {theta.shape}")
        print(f"Observation shape: {x.shape}")
        print(f"Note: Negative validation performance is normal (log-probability values)")
        
        # Move tensors to the correct device
        theta = theta.to(self.device)
        x = x.to(self.device)
        
        # Add training data (no proposal for standard NPE)
        self.inference = self.inference.append_simulations(theta, x)
        
        # Train (remove neural_net_kwargs from training arguments)
        density_estimator = self.inference.train(
            training_batch_size=training_batch_size,
            learning_rate=learning_rate,
            max_num_epochs=max_num_epochs,
            validation_fraction=validation_fraction,
            stop_after_epochs=stop_after_epochs,
            show_train_summary=True,
            **kwargs
        )
        
        # Build posterior
        self.posterior = self.inference.build_posterior()
        
        # Return training info as a dictionary
        training_info = {
            'density_estimator': density_estimator,
            'training_completed': True,
            'max_epochs': max_num_epochs,
            'batch_size': training_batch_size,
            'learning_rate': learning_rate,
            'validation_fraction': validation_fraction
        }
        
        return training_info
    
    def train_round(self, 
                   theta: torch.Tensor, 
                   x: torch.Tensor,
                   proposal_distribution = None,
                   training_batch_size: int = 512,
                   learning_rate: float = 1e-4,
                   max_num_epochs: int = 100,
                   validation_fraction: float = 0.1,
                   stop_after_epochs: int = 20,
                   neural_net_kwargs: Optional[Dict[str, Any]] = None,
                   **kwargs) -> Dict[str, Any]:
        """
        Train neural posterior estimator for a single SNPE round.
        
        Parameters:
        -----------
        theta : torch.Tensor of shape (n_samples, 2)
            Parameter vectors [U, P]
        x : torch.Tensor of shape (n_samples, x_dim)
            Observations (column counts)
        proposal_distribution : optional
            Proposal distribution for this round (None for first round)
        training_batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate
        max_num_epochs : int
            Maximum training epochs
        validation_fraction : float
            Fraction of data for validation
        stop_after_epochs : int
            Early stopping patience
        neural_net_kwargs : Dict[str, Any], optional
            Neural network configuration arguments
        **kwargs : additional training arguments
            
        Returns:
        --------
        training_info : dict
            Training information and losses
        """
        if self.inference is None:
            # Pass neural network configuration to setup
            setup_kwargs = {}
            if neural_net_kwargs is not None:
                setup_kwargs['neural_net_kwargs'] = neural_net_kwargs
            # Pass CNN-specific kwargs through to setup_inference
            for cnn_kwarg in ['cnn_dropout', 'cnn_normalize_input', 'cnn_use_auxiliary_features',
                              'cnn_use_density_channels', 'cnn_use_spatial_pyramid', 'disable_sbi_standardization',
                              'cnn_dual_branch']:
                if cnn_kwarg in kwargs:
                    setup_kwargs[cnn_kwarg] = kwargs.pop(cnn_kwarg)
            self.setup_inference(x_dim=x.shape[1], **setup_kwargs)

        print(f"Training SNPE round with {len(theta)} samples...")
        print(f"Parameter shape: {theta.shape}")
        print(f"Observation shape: {x.shape}")
        print(f"Note: Negative validation performance is normal (log-probability values)")
        
        # Move tensors to the correct device
        theta = theta.to(self.device)
        x = x.to(self.device)
        
        # Add training data with proposal if provided
        if proposal_distribution is not None:
            # proposal_distribution is now the posterior with default_x set
            self.inference = self.inference.append_simulations(theta, x, proposal=proposal_distribution)
        else:
            # First round: no proposal
            self.inference = self.inference.append_simulations(theta, x)
        
        # Train
        density_estimator = self.inference.train(
            training_batch_size=training_batch_size,
            learning_rate=learning_rate,
            max_num_epochs=max_num_epochs,
            validation_fraction=validation_fraction,
            stop_after_epochs=stop_after_epochs,
            show_train_summary=True,
            **kwargs
        )
        
        # Build posterior
        self.posterior = self.inference.build_posterior()
        
        # Return training info as a dictionary
        training_info = {
            'density_estimator': density_estimator,
            'training_completed': True,
            'max_epochs': max_num_epochs,
            'batch_size': training_batch_size,
            'learning_rate': learning_rate,
            'validation_fraction': validation_fraction
        }
        
        return training_info
    
    def train_sequential(self,
                        simulator: RandomWalkSimulator,
                        n_rounds: int,
                        n_simulations_per_round: int,
                        T: int,
                        x_obs: torch.Tensor,
                        training_batch_size: int = 512,
                        learning_rate: float = 1e-4,
                        max_num_epochs: int = 100,
                        validation_fraction: float = 0.1,
                        stop_after_epochs: int = 20,
                        neural_net_kwargs: Optional[Dict[str, Any]] = None,
                        convergence_threshold: float = 0.01,
                        random_seed: Optional[int] = None,
                        output_dir: Optional[str] = None,
                        n_workers: int = 1,
                        **kwargs) -> Dict[str, Any]:
        """
        Train SNPE model through multiple sequential rounds.
        
        Parameters:
        -----------
        simulator : RandomWalkSimulator
            Configured simulator instance
        n_rounds : int
            Number of sequential training rounds
        n_simulations_per_round : int
            Number of simulations to generate per round
        T : int
            Number of time steps for simulations
        x_obs : torch.Tensor
            Observed data for inference
        training_batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate
        max_num_epochs : int
            Maximum training epochs per round
        validation_fraction : float
            Fraction of data for validation
        stop_after_epochs : int
            Early stopping patience
        neural_net_kwargs : Dict[str, Any], optional
            Neural network configuration arguments
        convergence_threshold : float
            Threshold for early stopping based on posterior change
        random_seed : int, optional
            Random seed for reproducibility
        output_dir : str, optional
            Directory to save intermediate results
        **kwargs : additional training arguments
            
        Returns:
        --------
        sequential_training_info : dict
            Comprehensive training information across all rounds
        """
        print(f"\n🔄 Starting Sequential NPE with {n_rounds} rounds")
        print(f"   Simulations per round: {n_simulations_per_round}")
        print(f"   Convergence threshold: {convergence_threshold}")
        
        self.posteriors_by_round = []
        self.training_history = []
        
        for round_idx in range(n_rounds):
            self.current_round = round_idx + 1
            print(f"\n📊 ROUND {self.current_round}/{n_rounds}")
            
            # Determine proposal distribution for this round
            if round_idx == 0:
                proposal_distribution = None  # Use priors for first round
                print("   Using prior distributions for parameter sampling")
            else:
                # Set default x for the posterior and pass it as proposal
                self.posterior.set_default_x(x_obs)
                proposal_distribution = self.posterior
                print("   Using posterior from previous round as proposal")
            
            # Generate training data for this round
            print(f"   Generating {n_simulations_per_round} training samples...")
            theta, x = self.generate_training_data(
                simulator=simulator,
                n_simulations=n_simulations_per_round,
                T=T,
                proposal_distribution=proposal_distribution,
                random_seed=random_seed + round_idx * 1000 if random_seed else None,
                n_workers=n_workers
            )
            
            # Train model for this round
            print(f"   Training neural network...")
            training_info = self.train_round(
                theta=theta,
                x=x,
                proposal_distribution=proposal_distribution,
                training_batch_size=training_batch_size,
                learning_rate=learning_rate,
                max_num_epochs=max_num_epochs,
                validation_fraction=validation_fraction,
                stop_after_epochs=stop_after_epochs,
                neural_net_kwargs=neural_net_kwargs,
                **kwargs
            )
            
            # Store results for this round (ensure posterior has default_x set)
            self.posterior.set_default_x(x_obs)
            self.posteriors_by_round.append(self.posterior)
            self.training_history.append({
                'round': self.current_round,
                'n_simulations': n_simulations_per_round,
                'training_info': training_info
            })
            
            # Evaluate posterior on observed data
            posterior_samples = self.sample_posterior(x_obs, num_samples=1000)
            samples_np = posterior_samples.cpu().numpy()

            print(f"   Round {self.current_round} Results:")
            for pidx, pname in enumerate(self.param_names):
                pmean = samples_np[:, pidx].mean()
                pstd = samples_np[:, pidx].std()
                print(f"     {pname}: {pmean:.4f} +/- {pstd:.4f}")
            
            # Check for convergence (if not the first round)
            if round_idx > 0:
                convergence_metric = self._compute_convergence_metric(round_idx)
                print(f"     Convergence metric: {convergence_metric:.6f}")
                
                if convergence_metric < convergence_threshold:
                    print(f"   ✅ Convergence achieved after {self.current_round} rounds!")
                    break
            
            # Save intermediate results if output directory provided
            if output_dir is not None:
                self._save_round_results(output_dir, round_idx, posterior_samples, training_info)
        
        # Prepare comprehensive results
        sequential_training_info = {
            'total_rounds_completed': self.current_round,
            'converged': round_idx > 0 and convergence_metric < convergence_threshold if 'convergence_metric' in locals() else False,
            'final_convergence_metric': convergence_metric if 'convergence_metric' in locals() else None,
            'training_history': self.training_history,
            'posteriors_by_round': len(self.posteriors_by_round)
        }
        
        print(f"\n🎉 Sequential training completed after {self.current_round} rounds!")
        return sequential_training_info
    
    def _compute_convergence_metric(self, round_idx: int) -> float:
        """
        Compute convergence metric between consecutive rounds.
        
        Parameters:
        -----------
        round_idx : int
            Current round index
            
        Returns:
        --------
        float
            Convergence metric (lower values indicate better convergence)
        """
        if round_idx == 0:
            return float('inf')
        
        # Sample from previous and current posterior
        prev_samples = self.posteriors_by_round[round_idx - 1].sample((1000,))
        curr_samples = self.posteriors_by_round[round_idx].sample((1000,))
        
        # Compute simple metric based on sample statistics difference
        prev_mean = prev_samples.mean(dim=0)
        curr_mean = curr_samples.mean(dim=0)
        prev_std = prev_samples.std(dim=0)
        curr_std = curr_samples.std(dim=0)
        
        # Normalized difference in means and stds
        mean_diff = torch.norm(curr_mean - prev_mean)
        std_diff = torch.norm(curr_std - prev_std)
        
        # Combined metric
        metric = float(mean_diff + std_diff)
        return metric
    
    def _save_round_results(self, output_dir: str, round_idx: int, 
                           posterior_samples: torch.Tensor, 
                           training_info: Dict[str, Any]) -> None:
        """Save results for a specific round."""
        from pathlib import Path
        import pickle
        
        round_dir = Path(output_dir) / f"round_{round_idx + 1}"
        round_dir.mkdir(parents=True, exist_ok=True)
        
        # Save posterior samples
        samples_np = posterior_samples.cpu().numpy()
        np.save(round_dir / "posterior_samples.npy", samples_np)
        
        # Save training info
        with open(round_dir / "training_info.pkl", 'wb') as f:
            pickle.dump(training_info, f)
        
        print(f"     Round {round_idx + 1} results saved to {round_dir}")
    
    def get_round_results(self) -> List[Dict[str, Any]]:
        """
        Get results from all completed rounds.
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of results for each round
        """
        return self.training_history.copy()
    
    def sample_posterior(self, 
                        x_obs: torch.Tensor,
                        num_samples: int = 1000,
                        **kwargs) -> torch.Tensor:
        """
        Sample from posterior given observed data.
        
        Parameters:
        -----------
        x_obs : torch.Tensor
            Observed data (column counts)
        num_samples : int
            Number of posterior samples
        **kwargs : additional sampling arguments
            
        Returns:
        --------
        samples : torch.Tensor of shape (num_samples, 2)
            Posterior samples [U, P]
        """
        if self.posterior is None:
            raise RuntimeError("Must train model before sampling")
        
        # Move observation to the correct device
        x_obs = x_obs.to(self.device)
            
        return self.posterior.sample((num_samples,), x=x_obs, **kwargs)
    
    def log_prob(self, theta: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of parameters given observations.
        
        Parameters:
        -----------
        theta : torch.Tensor
            Parameter vectors [U, P]
        x_obs : torch.Tensor
            Observed data
            
        Returns:
        --------
        log_prob : torch.Tensor
            Log probabilities
        """
        if self.posterior is None:
            raise RuntimeError("Must train model before computing log prob")
        
        # Move tensors to the correct device
        theta = theta.to(self.device)
        x_obs = x_obs.to(self.device)
            
        return self.posterior.log_prob(theta, x=x_obs)
    
    def save_model(self, filepath: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save trained model.
        
        Parameters:
        -----------
        filepath : str
            Output filepath
        metadata : dict, optional
            Additional metadata
        """
        if self.posterior is None:
            raise RuntimeError("No trained model to save")
            
        data = {
            'posterior': self.posterior,
            'inference': self.inference,
            'prior': self.prior,
            'device': self.device,
            'model_config': self.model_config,
            'param_names': self.param_names,
            'use_2d_data': self.use_2d_data,
            'spatial_dims': self.spatial_dims,
            'metadata': metadata or {}
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"Saved model to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: Optional[str] = None) -> 'RandomWalkNPE':
        """
        Load trained model.
        
        Parameters:
        -----------
        filepath : str
            Model filepath
        device : str, optional
            Device to load model on. If None, uses saved device.
            
        Returns:
        --------
        inference_obj : RandomWalkNPE
            Loaded inference object
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Use provided device or fall back to saved device
        target_device = device or data['device']

        saved_config = data.get('model_config', None)
        obj = cls(
            device=target_device,
            model_config=saved_config,
            use_2d_data=data.get('use_2d_data', False),
            spatial_dims=data.get('spatial_dims', None),
        )
        obj.posterior = data['posterior']
        obj.inference = data['inference']
        obj.prior = data['prior']
        # Restore param_names from save (covers old files without model_config)
        if 'param_names' in data:
            obj.param_names = data['param_names']

        # Move components to target device if different from saved device
        if target_device != data['device']:
            try:
                if hasattr(obj.posterior, 'to'):
                    obj.posterior = obj.posterior.to(target_device)
                # Reconstruct prior on the correct device
                if saved_config is not None:
                    obj.prior = BoxUniform(
                        low=torch.tensor(saved_config.prior_low, dtype=torch.float32, device=target_device),
                        high=torch.tensor(saved_config.prior_high, dtype=torch.float32, device=target_device),
                    )
                else:
                    obj.prior = BoxUniform(
                        low=torch.tensor([0.01, 0.0], device=target_device),
                        high=torch.tensor([1.0, 1.0], device=target_device)
                    )
            except Exception as e:
                print(f"Warning: Could not move model to {target_device}: {e}")
                print("Model will remain on original device")

        return obj
    
    def plot_posterior_samples(self,
                              samples: torch.Tensor,
                              true_theta: Optional[torch.Tensor] = None,
                              figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot marginal posterior histograms for each parameter.

        Parameters:
        -----------
        samples : torch.Tensor
            Posterior samples of shape (n_samples, n_params)
        true_theta : torch.Tensor, optional
            True parameter values (for validation)
        figsize : tuple, optional
            Figure size (defaults to (6*n_params, 5))

        Returns:
        --------
        fig : matplotlib.figure.Figure
        """
        n_params = len(self.param_names)
        if self.model_config is not None:
            labels = self.model_config.param_labels
        else:
            labels = ['U (initial occupancy)', 'P (movement probability)']

        if figsize is None:
            figsize = (6 * n_params, 5)

        fig, axes = plt.subplots(1, n_params, figsize=figsize)
        if n_params == 1:
            axes = [axes]

        samples_np = samples.cpu().numpy()

        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.hist(samples_np[:, i], bins=50, alpha=0.7, density=True, color='skyblue')

            if true_theta is not None:
                ax.axvline(true_theta[i].item(), color='red', linestyle='--',
                          linewidth=2, label='True value')
                ax.legend()

            ax.set_xlabel(label)
            ax.set_ylabel('Density')
            ax.set_title(f'Posterior: {label}')

        plt.tight_layout()
        return fig
    
    def plot_pairwise(self,
                     samples: torch.Tensor,
                     true_theta: Optional[torch.Tensor] = None,
                     figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot pairwise posterior relationships (corner-style).

        Parameters:
        -----------
        samples : torch.Tensor
            Posterior samples of shape (n_samples, n_params)
        true_theta : torch.Tensor, optional
            True parameter values
        figsize : tuple, optional
            Figure size (defaults to (4*n_params, 4*n_params))

        Returns:
        --------
        fig : matplotlib.figure.Figure
        """
        n_params = len(self.param_names)
        if self.model_config is not None:
            labels = self.model_config.param_labels
        else:
            labels = ['U (initial occupancy)', 'P (movement probability)']

        if figsize is None:
            figsize = (4 * n_params, 4 * n_params)

        samples_np = samples.cpu().numpy()
        fig, axes = plt.subplots(n_params, n_params, figsize=figsize)
        if n_params == 1:
            axes = np.array([[axes]])

        for row in range(n_params):
            for col in range(n_params):
                ax = axes[row, col]
                if col > row:
                    ax.axis('off')
                elif row == col:
                    # Marginal histogram on diagonal
                    ax.hist(samples_np[:, row], bins=30, alpha=0.7, color='skyblue')
                    if true_theta is not None:
                        ax.axvline(true_theta[row].item(), color='red', linestyle='--')
                    ax.set_title(labels[row])
                    if row == n_params - 1:
                        ax.set_xlabel(labels[col])
                    ax.set_ylabel('Count')
                else:
                    # Scatter of col (x) vs row (y)
                    ax.scatter(samples_np[:, col], samples_np[:, row],
                              alpha=0.3, s=1, color='skyblue')
                    if true_theta is not None:
                        ax.scatter(true_theta[col].item(), true_theta[row].item(),
                                 color='red', s=50, marker='x', linewidth=2)
                    if row == n_params - 1:
                        ax.set_xlabel(labels[col])
                    if col == 0:
                        ax.set_ylabel(labels[row])

        plt.tight_layout()
        return fig


