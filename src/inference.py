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
from typing import Tuple, Dict, Any, Optional
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
    """Neural Posterior Estimation for Random Walk parameter inference."""
    
    def __init__(self, 
                 device: str = 'cpu',
                 seed: Optional[int] = None):
        """
        Initialize NPE inference.
        
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
        random_seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training data for NPE by running simulations.
        
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
            Prior bounds for parameters
        random_seed : int, optional
            Random seed for reproducibility
            
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
        
        # Generate training data
        for i in tqdm.tqdm(range(n_simulations)):
            # Sample parameters from priors (uniform distributions)
            U = np.random.uniform(*prior_bounds['U'])
            P = np.random.uniform(*prior_bounds['P'])
            
            # Run simulation
            column_counts, _, _ = simulator.simulate(U, P, T)
            
            # Store results
            parameters[i] = [U, P]
            observations[i] = column_counts
            
            # Progress indication
            # if n_simulations > 100 and (i + 1) % max(1, n_simulations // 10) == 0:
            #     print(f"Generated {i + 1}/{n_simulations} simulations ({100*(i+1)/n_simulations:.1f}%)")
        
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
        
        # Add training data
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
        
        samples_np = samples.numpy()
        
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
        samples_np = samples.numpy()
        
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
    return theta.numpy(), x.numpy()


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