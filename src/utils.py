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
            return obj.cpu().numpy().tolist()
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












