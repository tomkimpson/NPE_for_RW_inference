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
from typing import Tuple, Dict, Any, Optional, List, Union
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt

# Import centralized warning configuration
from utils import configure_warnings
configure_warnings()

# SBI imports
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

from simulator import RandomWalkSimulator
from cnn_utils import create_spatial_embedding_net, compute_2d_tensor_shape, validate_2d_input
from training_monitor import TrainingMonitor

import tqdm


class RandomWalkNPE:
    """Sequential Neural Posterior Estimation for Random Walk parameter inference."""
    
    def __init__(self,
                 device: str = 'cpu',
                 seed: Optional[int] = None,
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
        use_2d_data : bool
            Whether to use 2D spatial data (True) or 1D column counts (False)
        spatial_dims : Tuple[int, int], optional
            (Ly, Lx) dimensions for 2D data. Required if use_2d_data=True
        """
        self.device = device
        self.use_2d_data = use_2d_data
        self.spatial_dims = spatial_dims

        if use_2d_data and spatial_dims is None:
            raise ValueError("spatial_dims must be provided when use_2d_data=True")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Define prior for U and P parameters on the correct device
        # U: initial occupancy probability (0.01, 1.0)
        # P: movement probability (0.0, 1.0)
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
        
    def setup_inference(self, x_shape: Union[int, Tuple[int, ...]], **kwargs):
        """
        Set up SBI inference object.

        Parameters:
        -----------
        x_shape : int or Tuple[int, ...]
            Shape of observations - int for 1D (number of columns) or tuple for 2D (Ly, Lx)
        **kwargs : additional arguments for SNPE
        """
        # Prepare neural network configuration based on data type
        neural_net_kwargs = {
            'hidden_features': 256 if self.use_2d_data else 128,  # Larger hidden features for 2D
            'num_transforms': 5,
        }

        if self.use_2d_data:
            # Use CNN embedding for 2D spatial data
            if isinstance(x_shape, int):
                raise ValueError("x_shape must be tuple (Ly, Lx) for 2D data")

            Ly, Lx = x_shape
            spatial_embedding = create_spatial_embedding_net(
                input_height=Ly,
                input_width=Lx,
                output_dim=neural_net_kwargs['hidden_features'],
                dropout=kwargs.get('cnn_dropout', 0.05)  # Reduced dropout
            )
            neural_net_kwargs['embedding_net'] = spatial_embedding
        else:
            # Use identity embedding for 1D data
            neural_net_kwargs['embedding_net'] = torch.nn.Identity()

        # Update with user-provided settings
        neural_net_kwargs.update(kwargs.get('neural_net_kwargs', {}))

        # Create neural posterior estimator
        neural_posterior = posterior_nn(
            model='nsf',  # Neural Spline Flow
            **neural_net_kwargs
        )

        self.inference = SNPE(
            prior=self.prior,
            density_estimator=neural_posterior,
            device=self.device,
            **{k: v for k, v in kwargs.items() if k != 'neural_net_kwargs'}
        )
        
    def generate_training_data(
        self,
        simulator: RandomWalkSimulator,
        n_simulations: int,
        T: int,
        output_path: Optional[str] = None,
        prior_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        random_seed: Optional[int] = None,
        proposal_distribution = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training data for SNPE by running simulations.
        
        Parameters:
        -----------
        simulator : RandomWalkSimulator
            Configured simulator instance
        n_simulations : int
            Number of parameter-observation pairs to generate
        T : int
            Number of time steps for each simulation
        output_path : str, optional
            Path to save training data
        prior_bounds : Dict[str, Tuple[float, float]], optional
            Prior bounds for parameters (used only if proposal_distribution is None)
        random_seed : int, optional
            Random seed for reproducibility
        proposal_distribution : optional
            Proposal distribution for sequential rounds (if None, uses priors)
            
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            - parameters: Tensor of shape (n_simulations, 2) containing [U, P] values
            - observations: Tensor of shape (n_simulations, Lx) containing column counts
        """
        if n_simulations <= 0:
            raise ValueError("Number of simulations must be positive")
        if T < 0:
            raise ValueError("Number of time steps must be non-negative")
            
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        # Define default prior bounds
        if prior_bounds is None:
            prior_bounds = {
                'U': (0.01, 1.0),  # Avoid U=0 to ensure at least some agents
                'P': (0.0, 1.0)
            }
        
        # Validate prior bounds
        for param, (low, high) in prior_bounds.items():
            if low >= high:
                raise ValueError(f"Invalid prior bounds for {param}: low >= high")
            if param == 'U' and (low <= 0 or high > 1):
                raise ValueError("U prior bounds must be in (0, 1]")
            if param == 'P' and (low < 0 or high > 1):
                raise ValueError("P prior bounds must be in [0, 1]")
        
        print(f"Generating {n_simulations} training simulations...")
        
        # Initialize storage
        parameters = np.zeros((n_simulations, 2))

        # Determine observation shape and initialize storage
        if self.use_2d_data:
            if self.spatial_dims is None:
                raise ValueError("spatial_dims must be set for 2D data mode")
            Ly, Lx = self.spatial_dims
            observations = np.zeros((n_simulations, Ly, Lx))
        else:
            observations = np.zeros((n_simulations, simulator.Lx))

        # Sample all parameters at once if using proposal
        if proposal_distribution is not None:
            print(f"   Drawing {n_simulations} parameter samples from proposal distribution...")
            # Sample all parameters at once to avoid multiple progress bars
            theta_samples = proposal_distribution.sample((n_simulations,))
            theta_samples_np = theta_samples.cpu().numpy()

            # Generate training data
            for i in tqdm.tqdm(range(n_simulations), desc="Running simulations"):
                U = float(theta_samples_np[i, 0])
                P = float(theta_samples_np[i, 1])

                # Run simulation with appropriate output format
                observation, _, _ = simulator.simulate(U, P, T, use_2d_output=self.use_2d_data)

                # Store results
                parameters[i] = [U, P]
                observations[i] = observation
        else:
            # Generate training data with prior sampling
            for i in tqdm.tqdm(range(n_simulations), desc="Running simulations"):
                # Sample parameters from priors (uniform distributions)
                U = np.random.uniform(*prior_bounds['U'])
                P = np.random.uniform(*prior_bounds['P'])

                # Run simulation with appropriate output format
                observation, _, _ = simulator.simulate(U, P, T, use_2d_output=self.use_2d_data)

                # Store results
                parameters[i] = [U, P]
                observations[i] = observation
            
        
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
                    'initial_region_half_width': simulator.initial_region_half_width
                }
            }
            
            # Create directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure .pkl extension
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
            # Determine x_shape based on data type
            if self.use_2d_data:
                x_shape = x.shape[1:]  # (Ly, Lx) for 2D data
            else:
                x_shape = x.shape[1]   # int for 1D data
            self.setup_inference(x_shape=x_shape, **setup_kwargs)
            
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
                   monitor: Optional[TrainingMonitor] = None,
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
            # Determine x_shape based on data type
            if self.use_2d_data:
                x_shape = x.shape[1:]  # (Ly, Lx) for 2D data
            else:
                x_shape = x.shape[1]   # int for 1D data
            self.setup_inference(x_shape=x_shape, **setup_kwargs)
            
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
        
        # Train with optional monitoring
        training_kwargs = {
            'training_batch_size': training_batch_size,
            'learning_rate': learning_rate,
            'max_num_epochs': max_num_epochs,
            'validation_fraction': validation_fraction,
            'stop_after_epochs': stop_after_epochs,
            'show_train_summary': True,
        }

        # Add monitoring callback if provided
        if monitor is not None and self.use_2d_data:
            print("   Enabling CNN training monitoring...")

        training_kwargs.update(kwargs)
        density_estimator = self.inference.train(**training_kwargs)

        # Analyze training if monitor was used
        if monitor is not None:
            monitor.save_logs()
            monitor.plot_training_progress()

            # Check for convergence issues
            issues = monitor.check_convergence_issues()
            if issues:
                print("   ⚠️  Training issues detected:")
                for issue in issues:
                    print(f"      - {issue}")
            else:
                print("   ✅ No major training issues detected")
        
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
                        start_round: int = 1,
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
        start_round : int, optional
            Round number to start from (1-indexed). Use with checkpointing.
        **kwargs : additional training arguments

        Returns:
        --------
        sequential_training_info : dict
            Comprehensive training information across all rounds
        """
        if start_round == 1:
            print(f"\n🔄 Starting Sequential NPE with {n_rounds} rounds")
            self.posteriors_by_round = []
            self.training_history = []
        else:
            print(f"\n🔄 Resuming Sequential NPE from round {start_round} (target: {n_rounds} rounds)")

        print(f"   Simulations per round: {n_simulations_per_round}")
        print(f"   Convergence threshold: {convergence_threshold}")

        for round_idx in range(start_round - 1, n_rounds):
            self.current_round = round_idx + 1
            print(f"\n📊 ROUND {self.current_round}/{n_rounds}")
            
            # Determine proposal distribution for this round
            if self.current_round == 1:
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
                random_seed=random_seed + round_idx * 1000 if random_seed else None
            )
            
            # Train model for this round
            print(f"   Training neural network...")

            # Set up monitoring for 2D data
            monitor = None
            if self.use_2d_data and output_dir is not None:
                monitor_dir = Path(output_dir) / f"round_{self.current_round}" / "training_monitor"
                monitor = TrainingMonitor(str(monitor_dir))

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
                monitor=monitor,
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
            U_mean, U_std = samples_np[:, 0].mean(), samples_np[:, 0].std()
            P_mean, P_std = samples_np[:, 1].mean(), samples_np[:, 1].std()
            
            print(f"   Round {self.current_round} Results:")
            print(f"     U: {U_mean:.4f} ± {U_std:.4f}")
            print(f"     P: {P_mean:.4f} ± {P_std:.4f}")
            
            # Check for convergence (if we have previous rounds to compare against)
            if len(self.posteriors_by_round) > 1:
                convergence_metric = self._compute_convergence_metric(len(self.posteriors_by_round) - 1)
                print(f"     Convergence metric: {convergence_metric:.6f}")

                if convergence_metric < convergence_threshold:
                    print(f"   ✅ Convergence achieved after {self.current_round} rounds!")
                    break

            # Save intermediate results if output directory provided
            if output_dir is not None:
                self._save_round_results(output_dir, self.current_round - 1, posterior_samples, training_info)
        
        # Prepare comprehensive results
        sequential_training_info = {
            'total_rounds_completed': self.current_round,
            'converged': len(self.posteriors_by_round) > 1 and convergence_metric < convergence_threshold if 'convergence_metric' in locals() else False,
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

        # Add the current posterior to training info for resuming
        training_info_with_posterior = training_info.copy()
        training_info_with_posterior['posterior'] = self.posterior

        # Save training info with posterior
        with open(round_dir / "training_info.pkl", 'wb') as f:
            pickle.dump(training_info_with_posterior, f)

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
        
        # Recreate object with saved configuration
        use_2d_data = data.get('use_2d_data', False)
        spatial_dims = data.get('spatial_dims', None)

        obj = cls(device=target_device, use_2d_data=use_2d_data, spatial_dims=spatial_dims)
        obj.posterior = data['posterior']
        obj.inference = data['inference']
        obj.prior = data['prior']
        
        # Move components to target device if different from saved device
        if target_device != data['device']:
            try:
                # Move posterior to new device (if possible)
                if hasattr(obj.posterior, 'to'):
                    obj.posterior = obj.posterior.to(target_device)
                # Update prior device
                from sbi.utils import BoxUniform
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
                              figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        Plot posterior samples.
        
        Parameters:
        -----------
        samples : torch.Tensor
            Posterior samples [U, P]
        true_theta : torch.Tensor, optional
            True parameter values (for validation)
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        param_names = ['U (initial occupancy)', 'P (movement probability)']
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        samples_np = samples.cpu().numpy()
        
        for i, (ax, name) in enumerate(zip(axes, param_names)):
            # Histogram
            ax.hist(samples_np[:, i], bins=50, alpha=0.7, density=True, color='skyblue')
            
            # True value if provided
            if true_theta is not None:
                ax.axvline(true_theta[i].item(), color='red', linestyle='--', 
                          linewidth=2, label='True value')
                ax.legend()
                
            ax.set_xlabel(name)
            ax.set_ylabel('Density')
            ax.set_title(f'Posterior: {name}')
            
        plt.tight_layout()
        return fig
    
    def plot_pairwise(self, 
                     samples: torch.Tensor,
                     true_theta: Optional[torch.Tensor] = None,
                     figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
        """
        Plot pairwise posterior relationships.
        
        Parameters:
        -----------
        samples : torch.Tensor
            Posterior samples [U, P]
        true_theta : torch.Tensor, optional
            True parameter values
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        samples_np = samples.cpu().numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # U marginal
        axes[0, 0].hist(samples_np[:, 0], bins=30, alpha=0.7, color='skyblue')
        if true_theta is not None:
            axes[0, 0].axvline(true_theta[0].item(), color='red', linestyle='--')
        axes[0, 0].set_title('U (initial occupancy)')
        axes[0, 0].set_ylabel('Count')
        
        # P marginal  
        axes[1, 1].hist(samples_np[:, 1], bins=30, alpha=0.7, color='skyblue')
        if true_theta is not None:
            axes[1, 1].axvline(true_theta[1].item(), color='red', linestyle='--')
        axes[1, 1].set_title('P (movement probability)')
        axes[1, 1].set_xlabel('P')
        axes[1, 1].set_ylabel('Count')
        
        # Joint distribution (U vs P)
        axes[1, 0].scatter(samples_np[:, 0], samples_np[:, 1], 
                          alpha=0.3, s=1, color='skyblue')
        if true_theta is not None:
            axes[1, 0].scatter(true_theta[0].item(), true_theta[1].item(), 
                             color='red', s=50, marker='x', linewidth=2)
        axes[1, 0].set_xlabel('U (initial occupancy)')
        axes[1, 0].set_ylabel('P (movement probability)')
        axes[1, 0].set_title('Joint Posterior')
        
        # Turn off upper right
        axes[0, 1].axis('off')
        
        plt.tight_layout()
        return fig

    def load_checkpoint(self, output_dir: str, resume_from_round: int) -> Dict[str, Any]:
        """
        Load checkpoint from a previous SNPE run to resume training.

        Parameters:
        -----------
        output_dir : str
            Directory containing the saved SNPE rounds
        resume_from_round : int
            Round number to resume from (1-indexed)

        Returns:
        --------
        Dict[str, Any]
            Checkpoint information including loaded states
        """
        from pathlib import Path
        import pickle
        import numpy as np
        import torch

        output_path = Path(output_dir)

        if not output_path.exists():
            raise ValueError(f"Output directory {output_dir} does not exist")

        # Find the latest valid round by checking backwards
        valid_round = None
        for round_num in range(resume_from_round, 0, -1):
            round_dir = output_path / f"round_{round_num}"
            training_info_path = round_dir / "training_info.pkl"
            samples_path = round_dir / "posterior_samples.npy"

            if training_info_path.exists() and samples_path.exists():
                try:
                    # Try to load the training info with different methods
                    training_info = None

                    # Method 1: Try torch.load without weights_only (for older PyTorch versions)
                    try:
                        training_info = torch.load(training_info_path, map_location=self.device)
                    except Exception:
                        # Method 2: Try with weights_only=False explicitly
                        try:
                            training_info = torch.load(training_info_path, map_location=self.device, weights_only=False)
                        except Exception:
                            # Method 3: Try standard pickle
                            import pickle
                            with open(training_info_path, 'rb') as f:
                                training_info = pickle.load(f)

                    if training_info is None:
                        raise ValueError("Could not load training info with any method")

                    # Check if it has a density estimator (required for reconstruction)
                    if 'density_estimator' in training_info:
                        valid_round = round_num
                        print(f"🔄 Loading checkpoint from round {valid_round}")
                        break
                    else:
                        print(f"   ⚠️ Round {round_num} missing density_estimator")

                except Exception as e:
                    print(f"   ❌ Round {round_num} load error: {e}")
                    continue

        if valid_round is None:
            raise ValueError(f"No valid rounds found with density_estimator in {output_dir}")

        # Load the valid round's data
        round_dir = output_path / f"round_{valid_round}"
        training_info_path = round_dir / "training_info.pkl"
        samples_path = round_dir / "posterior_samples.npy"

        # Load the training info with robust method
        training_info = None
        try:
            training_info = torch.load(training_info_path, map_location=self.device)
        except Exception:
            try:
                training_info = torch.load(training_info_path, map_location=self.device, weights_only=False)
            except Exception:
                import pickle
                with open(training_info_path, 'rb') as f:
                    training_info = pickle.load(f)

        if training_info is None:
            raise ValueError(f"Could not load training info from {training_info_path}")

        posterior_samples = np.load(samples_path)

        # Reconstruct the inference object and posterior from density estimator
        print(f"   🔧 Reconstructing inference state from density_estimator...")

        # The density_estimator contains the trained neural network
        density_estimator = training_info['density_estimator']

        # Create a new SNPE instance with the loaded density estimator
        from sbi import inference as sbi_inference
        self.inference = sbi_inference.SNPE(
            prior=self.prior,
            density_estimator=density_estimator,
            device=self.device
        )

        # Build the posterior from the reconstructed inference
        self.posterior = self.inference.build_posterior()
        print(f"   ✅ Successfully reconstructed posterior from round {valid_round}")

        # Initialize tracking lists for resumed training
        self.posteriors_by_round = []
        self.training_history = []

        # Rebuild training history for all completed rounds up to valid_round
        for round_idx in range(1, valid_round + 1):
            prev_round_dir = output_path / f"round_{round_idx}"
            if prev_round_dir.exists():
                prev_training_path = prev_round_dir / "training_info.pkl"
                if prev_training_path.exists():
                    try:
                        # Load with robust method
                        prev_training_info = None
                        try:
                            prev_training_info = torch.load(prev_training_path, map_location=self.device)
                        except Exception:
                            try:
                                prev_training_info = torch.load(prev_training_path, map_location=self.device, weights_only=False)
                            except Exception:
                                import pickle
                                with open(prev_training_path, 'rb') as f:
                                    prev_training_info = pickle.load(f)

                        if prev_training_info is None:
                            raise ValueError("Could not load previous training info")

                        # Add to training history
                        self.training_history.append({
                            'round': round_idx,
                            'n_simulations': prev_training_info.get('n_simulations', None),
                            'training_info': prev_training_info
                        })

                        # For the final round, use the reconstructed posterior
                        if round_idx == valid_round:
                            self.posteriors_by_round.append(self.posterior)

                    except Exception as e:
                        print(f"   ⚠️ Could not load round {round_idx} history: {e}")

        # Set current round
        self.current_round = valid_round

        checkpoint_info = {
            'resumed_from_round': valid_round,
            'output_dir': str(output_path),
            'posterior_samples_shape': posterior_samples.shape,
            'training_history_length': len(self.training_history),
            'posteriors_loaded': len(self.posteriors_by_round),
            'density_estimator_loaded': True
        }

        print(f"   ✅ Loaded {len(self.training_history)} rounds of training history")
        print(f"   ✅ Ready to resume from round {valid_round + 1}")

        return checkpoint_info


