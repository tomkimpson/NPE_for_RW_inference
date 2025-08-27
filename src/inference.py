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
import matplotlib.pyplot as plt

# Import centralized warning configuration
from utils import configure_warnings
configure_warnings()

# SBI imports
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

from simulator import RandomWalkSimulator

import tqdm


class RandomWalkNPE:
    """Sequential Neural Posterior Estimation for Random Walk parameter inference."""
    
    def __init__(self, 
                 device: str = 'cpu',
                 seed: Optional[int] = None):
        """
        Initialize SNPE inference.
        
        Parameters:
        -----------
        device : str
            Device for training ('cpu' or 'cuda')
        seed : int, optional
            Random seed
        """
        self.device = device
        
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
        
    def setup_inference(self, x_dim: int, **kwargs):
        """
        Set up SBI inference object.
        
        Parameters:
        -----------
        x_dim : int
            Dimension of observations (number of columns)
        **kwargs : additional arguments for SNPE
        """
        # Default neural network configuration
        neural_net_kwargs = {
            'hidden_features': 128,
            'num_transforms': 5,
            'embedding_net': torch.nn.Identity(),
        }
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
                
                # Run simulation
                column_counts, _, _ = simulator.simulate(U, P, T)
                
                # Store results
                parameters[i] = [U, P]
                observations[i] = column_counts
        else:
            # Generate training data with prior sampling
            for i in tqdm.tqdm(range(n_simulations), desc="Running simulations"):
                # Sample parameters from priors (uniform distributions)
                U = np.random.uniform(*prior_bounds['U'])
                P = np.random.uniform(*prior_bounds['P'])
                
                # Run simulation
                column_counts, _, _ = simulator.simulate(U, P, T)
                
                # Store results
                parameters[i] = [U, P]
                observations[i] = column_counts
            
        
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
                random_seed=random_seed + round_idx * 1000 if random_seed else None
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
            U_mean, U_std = samples_np[:, 0].mean(), samples_np[:, 0].std()
            P_mean, P_std = samples_np[:, 1].mean(), samples_np[:, 1].std()
            
            print(f"   Round {self.current_round} Results:")
            print(f"     U: {U_mean:.4f} ± {U_std:.4f}")
            print(f"     P: {P_mean:.4f} ± {P_std:.4f}")
            
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
        
        obj = cls(device=target_device)
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
    
    def plot_snpe_evolution(self, 
                           true_theta: Optional[torch.Tensor] = None,
                           figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot the evolution of posterior estimates across SNPE rounds.
        
        Parameters:
        -----------
        true_theta : torch.Tensor, optional
            True parameter values [U, P]
        figsize : Tuple[int, int]
            Figure size
            
        Returns:
        --------
        plt.Figure
            Figure showing posterior evolution
        """
        if len(self.posteriors_by_round) < 2:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'Need at least 2 rounds\nfor evolution plot', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('SNPE Posterior Evolution')
            return fig
        
        n_rounds = len(self.posteriors_by_round)
        fig, axes = plt.subplots(2, n_rounds, figsize=figsize)
        
        if n_rounds == 1:
            axes = axes.reshape(2, 1)
        
        param_names = ['U (initial occupancy)', 'P (movement probability)']
        colors = plt.cm.viridis(np.linspace(0, 1, n_rounds))
        
        # Sample from each round's posterior
        all_samples = []
        for i, posterior in enumerate(self.posteriors_by_round):
            samples = posterior.sample((1000,)).cpu().numpy()
            all_samples.append(samples)
            
            # Plot marginals for each parameter
            for param_idx, param_name in enumerate(param_names):
                ax = axes[param_idx, i]
                ax.hist(samples[:, param_idx], bins=30, alpha=0.7, 
                       color=colors[i], density=True)
                
                if true_theta is not None:
                    ax.axvline(true_theta[param_idx].item(), color='red', 
                              linestyle='--', linewidth=2)
                
                ax.set_title(f'Round {i+1}\n{param_name}')
                ax.set_ylabel('Density' if i == 0 else '')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_round_comparison(self, 
                             round1: int, 
                             round2: int,
                             true_theta: Optional[torch.Tensor] = None,
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Compare posteriors between two specific rounds.
        
        Parameters:
        -----------
        round1, round2 : int
            Round indices to compare (1-based)
        true_theta : torch.Tensor, optional
            True parameter values
        figsize : Tuple[int, int]
            Figure size
            
        Returns:
        --------
        plt.Figure
            Comparison figure
        """
        if round1 < 1 or round1 > len(self.posteriors_by_round):
            raise ValueError(f"round1 must be between 1 and {len(self.posteriors_by_round)}")
        if round2 < 1 or round2 > len(self.posteriors_by_round):
            raise ValueError(f"round2 must be between 1 and {len(self.posteriors_by_round)}")
        
        # Get samples from both rounds
        samples1 = self.posteriors_by_round[round1-1].sample((1000,)).cpu().numpy()
        samples2 = self.posteriors_by_round[round2-1].sample((1000,)).cpu().numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        param_names = ['U (initial occupancy)', 'P (movement probability)']
        
        for i, param_name in enumerate(param_names):
            # Marginal comparison
            axes[i, 0].hist(samples1[:, i], bins=30, alpha=0.7, 
                           label=f'Round {round1}', color='lightblue', density=True)
            axes[i, 0].hist(samples2[:, i], bins=30, alpha=0.7, 
                           label=f'Round {round2}', color='lightcoral', density=True)
            
            if true_theta is not None:
                axes[i, 0].axvline(true_theta[i].item(), color='red', 
                                  linestyle='--', linewidth=2, label='True value')
            
            axes[i, 0].set_xlabel(param_name)
            axes[i, 0].set_ylabel('Density')
            axes[i, 0].set_title(f'{param_name} Comparison')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
        
        # Joint distribution comparison
        axes[0, 1].scatter(samples1[:, 0], samples1[:, 1], 
                          alpha=0.5, s=1, color='lightblue', label=f'Round {round1}')
        axes[0, 1].scatter(samples2[:, 0], samples2[:, 1], 
                          alpha=0.5, s=1, color='lightcoral', label=f'Round {round2}')
        
        if true_theta is not None:
            axes[0, 1].scatter(true_theta[0].item(), true_theta[1].item(), 
                              color='red', s=100, marker='x', linewidth=3, label='True values')
        
        axes[0, 1].set_xlabel('U (initial occupancy)')
        axes[0, 1].set_ylabel('P (movement probability)')
        axes[0, 1].set_title('Joint Posterior Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Summary statistics comparison
        stats1 = {
            'U_mean': samples1[:, 0].mean(), 'U_std': samples1[:, 0].std(),
            'P_mean': samples1[:, 1].mean(), 'P_std': samples1[:, 1].std()
        }
        stats2 = {
            'U_mean': samples2[:, 0].mean(), 'U_std': samples2[:, 0].std(),
            'P_mean': samples2[:, 1].mean(), 'P_std': samples2[:, 1].std()
        }
        
        stats_text = f"Round {round1}:\n"
        stats_text += f"U: {stats1['U_mean']:.3f} ± {stats1['U_std']:.3f}\n"
        stats_text += f"P: {stats1['P_mean']:.3f} ± {stats1['P_std']:.3f}\n\n"
        stats_text += f"Round {round2}:\n"
        stats_text += f"U: {stats2['U_mean']:.3f} ± {stats2['U_std']:.3f}\n"
        stats_text += f"P: {stats2['P_mean']:.3f} ± {stats2['P_std']:.3f}"
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def posterior_predictive_sample(self,
                                   simulator: "RandomWalkSimulator",
                                   x_obs: torch.Tensor,
                                   T: int,
                                   n_posterior_samples: int = 1000,
                                   n_simulations_per_sample: int = 1,
                                   random_seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate posterior predictive samples by running simulator with posterior samples.
        
        Parameters:
        -----------
        simulator : RandomWalkSimulator
            Simulator instance configured with appropriate lattice parameters
        x_obs : torch.Tensor
            Observed data that was used for posterior inference
        T : int
            Number of time steps for simulation
        n_posterior_samples : int
            Number of posterior samples to draw for prediction
        n_simulations_per_sample : int
            Number of forward simulations per posterior sample (for averaging)
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        Tuple containing:
        - posterior_samples : torch.Tensor of shape (n_posterior_samples, 2)
            Posterior parameter samples used for prediction [U, P]
        - predictions : torch.Tensor of shape (n_posterior_samples, n_columns)
            Predicted column counts for each posterior sample
            
        Notes:
        ------
        This method provides a convenient interface to posterior predictive sampling
        directly from the NPE object. For more advanced options and analysis,
        consider using the standalone predict.py script.
        """
        if self.posterior is None:
            raise RuntimeError("Must train model before generating predictions")
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        
        # Sample from posterior
        posterior_samples = self.sample_posterior(x_obs, num_samples=n_posterior_samples)
        posterior_samples_np = posterior_samples.cpu().numpy()
        
        # Initialize predictions array
        n_columns = simulator.Lx
        predictions = np.zeros((n_posterior_samples, n_columns))
        
        print(f"🔮 Generating {n_posterior_samples} posterior predictive samples...")
        if n_simulations_per_sample > 1:
            print(f"   Using {n_simulations_per_sample} simulations per posterior sample (averaged)")
        
        # Generate predictions for each posterior sample
        for i, (U, P) in enumerate(posterior_samples_np):
            if i % max(1, n_posterior_samples // 10) == 0:
                progress = 100 * (i + 1) / n_posterior_samples
                print(f"   Progress: {i+1}/{n_posterior_samples} ({progress:.1f}%)")
            
            # Run multiple simulations per posterior sample if requested
            sample_predictions = []
            for sim_idx in range(n_simulations_per_sample):
                seed_offset = i * n_simulations_per_sample + sim_idx if random_seed is not None else None
                seed = random_seed + seed_offset if seed_offset is not None else None
                
                column_counts, _, _ = simulator.simulate(
                    U=float(U), 
                    P=float(P), 
                    T=T,
                    random_seed=seed
                )
                sample_predictions.append(column_counts)
            
            # Average across simulations for this posterior sample
            predictions[i] = np.mean(sample_predictions, axis=0)
        
        print(f"✅ Posterior predictive sampling completed")
        
        # Convert predictions back to tensor
        predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
        
        return posterior_samples, predictions_tensor


# Legacy functions for backward compatibility
def generate_training_data(
    simulator: RandomWalkSimulator,
    n_simulations: int,
    T: int,
    prior_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy function - use RandomWalkNPE.generate_training_data instead."""
    npe = RandomWalkNPE()
    theta, x = npe.generate_training_data(
        simulator, n_simulations, T, None, prior_bounds, random_seed
    )
    return theta.cpu().numpy(), x.cpu().numpy()


def define_priors(prior_type: str = "uniform") -> Dict[str, Any]:
    """
    Define prior distributions for the parameters U and P.
    
    Parameters:
    -----------
    prior_type : str
        Type of prior distribution ("uniform", "beta", "custom")
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing prior distribution objects
    """
    if prior_type == "uniform":
        return {
            'U': stats.uniform(loc=0.01, scale=0.99),  # Uniform(0.01, 1.0)
            'P': stats.uniform(loc=0.0, scale=1.0)     # Uniform(0.0, 1.0)
        }
    elif prior_type == "beta":
        return {
            'U': stats.beta(a=2, b=2),  # Beta(2, 2) - symmetric around 0.5
            'P': stats.beta(a=1, b=1)   # Beta(1, 1) - equivalent to Uniform(0, 1)
        }
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")


def compute_summary_statistics(observations: np.ndarray) -> np.ndarray:
    """
    Compute summary statistics from raw observation data.
    
    This function can be used to reduce the dimensionality of observations
    or extract relevant features for NPE training.
    
    Parameters:
    -----------
    observations : np.ndarray
        Raw observation data of shape (n_samples, n_columns)
        
    Returns:
    --------
    np.ndarray
        Summary statistics
    """
    # For now, return raw observations
    # This can be extended to compute mean, variance, moments, etc.
    return observations