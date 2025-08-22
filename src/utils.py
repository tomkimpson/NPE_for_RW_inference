"""
Utility functions for the NPE random walk inference project.

This module contains helper functions for data processing,
visualization, metrics, and other common functionality.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path
import json
import logging
from datetime import datetime


def configure_warnings() -> None:
    """
    Configure warning filters to suppress known deprecation warnings from third-party libraries.
    This helps keep the output clean while still showing important warnings.
    """
    # Suppress PyTorch deprecation warnings that come from SBI/third-party libraries
    warnings.filterwarnings("ignore", message=".*torch.triangular_solve is deprecated.*")
    warnings.filterwarnings("ignore", message=".*torch.solve is deprecated.*")
    warnings.filterwarnings("ignore", message=".*torch.qr is deprecated.*")
    
    # Suppress other common ML library warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.*")
    
    # Keep important warnings visible
    warnings.filterwarnings("default", category=RuntimeWarning)
    warnings.filterwarnings("default", category=DeprecationWarning, module="src.*")


def check_device_availability() -> Tuple[str, Dict[str, Any]]:
    """
    Check GPU/CUDA availability and return recommended device.
    
    Returns:
    --------
    Tuple[str, Dict[str, Any]]
        Recommended device string and device information
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_current_device': None,
        'cuda_device_name': None,
        'cuda_memory_total': None,
        'cuda_memory_free': None
    }
    
    # Get CUDA info if available
    if torch.cuda.is_available():
        device_info['cuda_current_device'] = torch.cuda.current_device()
        device_info['cuda_device_name'] = torch.cuda.get_device_name()
        
        # Get memory info (in GB)
        try:
            memory_total = torch.cuda.get_device_properties(0).total_memory
            memory_reserved = torch.cuda.memory_reserved(0)
            memory_allocated = torch.cuda.memory_allocated(0)
            
            device_info['cuda_memory_total'] = memory_total / 1e9  # Convert to GB
            device_info['cuda_memory_reserved'] = memory_reserved / 1e9
            device_info['cuda_memory_allocated'] = memory_allocated / 1e9
            device_info['cuda_memory_free'] = (memory_total - memory_reserved) / 1e9
        except:
            # Some CUDA setups might not support memory queries
            pass
    
    # Recommend device: CUDA if available, otherwise CPU
    if torch.cuda.is_available():
        recommended_device = 'cuda'
    else:
        recommended_device = 'cpu'
    
    return recommended_device, device_info


def print_device_info(device: str, device_info: Dict[str, Any]) -> None:
    """
    Print device information in a user-friendly format.
    
    Parameters:
    -----------
    device : str
        Device being used
    device_info : Dict[str, Any]
        Device information from check_device_availability()
    """
    print(f"🔧 Device Configuration:")
    print(f"   Using device: {device.upper()}")
    
    if device == 'cuda' and device_info['cuda_available']:
        print(f"   ✅ CUDA available: {device_info['cuda_device_count']} device(s)")
        print(f"   GPU: {device_info['cuda_device_name']}")
        if device_info['cuda_memory_total']:
            print(f"   GPU Memory: {device_info['cuda_memory_total']:.1f} GB total, "
                  f"{device_info['cuda_memory_free']:.1f} GB free")
    
    elif device == 'cpu':
        print(f"   ℹ️  Using CPU")
        # Show what acceleration was available but not used
        if device_info['cuda_available']:
            print(f"   📝 Note: CUDA is available but CPU was selected")
        else:
            print(f"   📝 No GPU acceleration available")
    
    else:
        # Handle fallback cases
        print(f"   ⚠️  Requested {device.upper()} not available")
        if device_info['cuda_available']:
            print(f"   📝 CUDA is available as alternative")
        print(f"   💡 Falling back to CPU (training will be slower)")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Parameters:
    -----------
    log_level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    log_file : str, optional
        Log file path
        
    Returns:
    --------
    logger : logging.Logger
        Configured logger
    """
    logger = logging.getLogger('NPE_RandomWalk')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_parameters(U: float, P: float) -> None:
    """
    Validate parameter values.
    
    Parameters:
    -----------
    U : float
        Initial occupancy probability
    P : float
        Movement probability
        
    Raises:
    -------
    ValueError
        If parameters are outside valid ranges
    """
    if not 0 < U <= 1:
        raise ValueError(f"U must be in (0, 1], got {U}")
    if not 0 <= P <= 1:
        raise ValueError(f"P must be in [0, 1], got {P}")


def compute_posterior_statistics(samples: np.ndarray) -> Dict[str, Any]:
    """
    Compute comprehensive statistics from posterior samples.
    
    Parameters:
    -----------
    samples : np.ndarray of shape (n_samples, 2)
        Posterior samples [U, P]
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary of statistics
    """
    stats = {}
    param_names = ['U', 'P']
    
    for i, name in enumerate(param_names):
        param_samples = samples[:, i]
        
        stats[name] = {
            'mean': float(param_samples.mean()),
            'std': float(param_samples.std()),
            'median': float(np.median(param_samples)),
            'mode': float(param_samples[np.argmax(np.histogram(param_samples, bins=100)[0])]),
            'ci_95': [float(np.percentile(param_samples, 2.5)), 
                     float(np.percentile(param_samples, 97.5))],
            'ci_68': [float(np.percentile(param_samples, 16)), 
                     float(np.percentile(param_samples, 84))],
            'min': float(param_samples.min()),
            'max': float(param_samples.max())
        }
    
    # Correlation between parameters
    stats['correlation'] = float(np.corrcoef(samples[:, 0], samples[:, 1])[0, 1])
    
    return stats


def assess_coverage(posterior_samples: np.ndarray, 
                   true_parameters: np.ndarray,
                   confidence_levels: List[float] = [0.68, 0.95]) -> Dict[str, Any]:
    """
    Assess coverage of credible intervals.
    
    Parameters:
    -----------
    posterior_samples : np.ndarray of shape (n_samples, 2)
        Posterior samples [U, P]
    true_parameters : np.ndarray of shape (2,)
        True parameter values [U, P]
    confidence_levels : List[float]
        Confidence levels to assess
        
    Returns:
    --------
    Dict[str, Any]
        Coverage assessment results
    """
    results = {}
    param_names = ['U', 'P']
    
    for i, name in enumerate(param_names):
        results[name] = {}
        param_samples = posterior_samples[:, i]
        true_value = true_parameters[i]
        
        for level in confidence_levels:
            alpha = 1 - level
            lower = np.percentile(param_samples, 100 * alpha / 2)
            upper = np.percentile(param_samples, 100 * (1 - alpha / 2))
            
            in_interval = lower <= true_value <= upper
            interval_width = upper - lower
            
            results[name][f'coverage_{int(level*100)}'] = {
                'covered': bool(in_interval),
                'interval': [float(lower), float(upper)],
                'width': float(interval_width),
                'true_value': float(true_value)
            }
    
    return results


def compare_posteriors(samples1: np.ndarray, samples2: np.ndarray, 
                      labels: List[str] = ['Model 1', 'Model 2']) -> Dict[str, Any]:
    """
    Compare two sets of posterior samples.
    
    Parameters:
    -----------
    samples1, samples2 : np.ndarray
        Posterior samples from different models
    labels : List[str]
        Labels for the two models
        
    Returns:
    --------
    Dict[str, Any]
        Comparison results
    """
    from scipy import stats as scipy_stats
    
    results = {}
    param_names = ['U', 'P']
    
    for i, name in enumerate(param_names):
        s1, s2 = samples1[:, i], samples2[:, i]
        
        # KS test for distribution comparison
        ks_stat, ks_pvalue = scipy_stats.ks_2samp(s1, s2)
        
        # Mean and std comparison
        mean_diff = s1.mean() - s2.mean()
        pooled_std = np.sqrt((s1.var() + s2.var()) / 2)
        
        results[name] = {
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'mean_difference': float(mean_diff),
            'pooled_std': float(pooled_std),
            'standardized_difference': float(mean_diff / pooled_std) if pooled_std > 0 else 0,
            f'{labels[0]}_stats': {
                'mean': float(s1.mean()),
                'std': float(s1.std())
            },
            f'{labels[1]}_stats': {
                'mean': float(s2.mean()),
                'std': float(s2.std())
            }
        }
    
    return results


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save results to JSON file with proper serialization.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Results dictionary
    filepath : str
        Output file path
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_arrays(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(item) for item in obj]
        else:
            return obj
    
    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Save with timestamp
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'results': convert_arrays(results)
    }
    
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Parameters:
    -----------
    filepath : str
        Input file path
        
    Returns:
    --------
    Dict[str, Any]
        Loaded results
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data['results']


def create_summary_report(posterior_samples: np.ndarray,
                         true_parameters: Optional[np.ndarray] = None,
                         observed_data: Optional[np.ndarray] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Create a comprehensive summary report.
    
    Parameters:
    -----------
    posterior_samples : np.ndarray
        Posterior samples [U, P]
    true_parameters : np.ndarray, optional
        True parameter values
    observed_data : np.ndarray, optional
        Observed column counts
    metadata : Dict[str, Any], optional
        Additional metadata
        
    Returns:
    --------
    str
        Formatted summary report
    """
    report = []
    report.append("="*60)
    report.append("NPE RANDOM WALK INFERENCE SUMMARY REPORT")
    report.append("="*60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Metadata section
    if metadata:
        report.append("CONFIGURATION")
        report.append("-" * 20)
        for key, value in metadata.items():
            report.append(f"{key}: {value}")
        report.append("")
    
    # Posterior statistics
    stats = compute_posterior_statistics(posterior_samples)
    
    report.append("POSTERIOR STATISTICS")
    report.append("-" * 20)
    for param in ['U', 'P']:
        param_stats = stats[param]
        report.append(f"{param} (N={len(posterior_samples)}):")
        report.append(f"  Mean ± Std: {param_stats['mean']:.4f} ± {param_stats['std']:.4f}")
        report.append(f"  Median: {param_stats['median']:.4f}")
        report.append(f"  95% CI: [{param_stats['ci_95'][0]:.4f}, {param_stats['ci_95'][1]:.4f}]")
        report.append(f"  68% CI: [{param_stats['ci_68'][0]:.4f}, {param_stats['ci_68'][1]:.4f}]")
        report.append("")
    
    report.append(f"Parameter Correlation: {stats['correlation']:.4f}")
    report.append("")
    
    # Validation if true parameters provided
    if true_parameters is not None:
        report.append("VALIDATION")
        report.append("-" * 20)
        coverage = assess_coverage(posterior_samples, true_parameters)
        
        param_names = ['U', 'P']
        for i, param in enumerate(param_names):
            true_val = true_parameters[i]
            param_stats = stats[param]
            
            report.append(f"{param}:")
            report.append(f"  True value: {true_val:.4f}")
            report.append(f"  Posterior mean: {param_stats['mean']:.4f}")
            report.append(f"  Bias: {param_stats['mean'] - true_val:.4f}")
            
            for level in [68, 95]:
                cov_info = coverage[param][f'coverage_{level}']
                status = "✓" if cov_info['covered'] else "✗"
                report.append(f"  {level}% CI {status}: [{cov_info['interval'][0]:.4f}, {cov_info['interval'][1]:.4f}]")
            
            report.append("")
    
    # Observed data summary
    if observed_data is not None:
        report.append("OBSERVED DATA")
        report.append("-" * 20)
        report.append(f"Total agents: {observed_data.sum()}")
        report.append(f"Columns with agents: {(observed_data > 0).sum()}/{len(observed_data)}")
        report.append(f"Max agents per column: {observed_data.max()}")
        report.append(f"Mean agents per column: {observed_data.mean():.2f}")
        report.append("")
    
    report.append("="*60)
    
    return "\n".join(report)


def plot_training_curves(training_info: Dict[str, Any], 
                        figsize: Tuple[int, int] = (10, 4)) -> plt.Figure:
    """
    Plot training curves if available.
    
    Parameters:
    -----------
    training_info : Dict[str, Any]
        Training information from NPE
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Training loss
    if 'training_log_probs' in training_info:
        epochs = range(len(training_info['training_log_probs']))
        axes[0].plot(epochs, training_info['training_log_probs'], 'b-', label='Training')
        
        if 'validation_log_probs' in training_info:
            axes[0].plot(epochs, training_info['validation_log_probs'], 'r-', label='Validation')
            
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Log Probability')
        axes[0].set_title('Training Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Learning rate schedule (if available)
    if 'learning_rates' in training_info:
        axes[1].plot(training_info['learning_rates'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Learning rate\nschedule not\navailable', 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    return fig