"""
Simulator module for generating random walk data.

This module contains the RandomWalkSimulator class for simulating 
2D lattice random walks as described in Simpson & Planck for use 
in Neural Posterior Estimation (NPE) training and validation.
"""

import numpy as np
from typing import Tuple, List, Optional, Union
import matplotlib.pyplot as plt


class RandomWalkSimulator:
    """
    2D lattice random walk simulator for biological population models.
    
    This simulator implements the discrete random walk model from Simpson & Planck,
    where agents move on a 2D square lattice with zero-flux boundary conditions.
    """
    
    def __init__(self, Lx: int, Ly: int, initial_region_half_width: Optional[int] = None):
        """
        Initialize the random walk simulator.
        
        Parameters:
        -----------
        Lx : int
            Number of lattice sites in x-direction (columns)
        Ly : int
            Number of lattice sites in y-direction (rows)
        initial_region_half_width : int, optional
            Half-width of initial region for agent placement. If None, 
            defaults to Lx//4 (agents initially placed in central columns)
        """
        self.Lx = Lx
        self.Ly = Ly
        self.initial_region_half_width = initial_region_half_width or Lx // 4
        
        # Validate parameters
        if Lx <= 0 or Ly <= 0:
            raise ValueError("Lattice dimensions must be positive")
        if self.initial_region_half_width >= Lx // 2:
            raise ValueError("Initial region half-width too large for lattice")
    
    def initialize_lattice(self, U: float, random_seed: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Initialize agent positions based on occupancy probability U.
        
        Parameters:
        -----------
        U : float
            Initial occupancy probability (0 < U <= 1)
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        List[Tuple[int, int]]
            List of initial agent positions as (x, y) tuples with centered coordinates
        """
        if not 0 < U <= 1:
            raise ValueError("Occupancy probability U must be in (0, 1]")
            
        if random_seed is not None:
            np.random.seed(random_seed)
        
        positions = []
        
        # Define initial region around x=0 (centered coordinates)
        x_min = -self.initial_region_half_width
        x_max = self.initial_region_half_width
        
        # Place agents with probability U in initial region
        for x in range(x_min, x_max + 1):
            for y in range(self.Ly):
                if np.random.random() < U:
                    positions.append((x, y))
        
        return positions
    
    def _get_valid_moves(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Get valid neighboring positions (zero-flux boundary conditions).
        
        Parameters:
        -----------
        x, y : int
            Current position (centered coordinates: x ∈ [-Lx/2, Lx/2], y ∈ [0, Ly-1])
            
        Returns:
        --------
        List[Tuple[int, int]]
            List of valid neighboring positions
        """
        moves = []
        
        # Calculate x boundaries (centered around 0)
        x_min = -(self.Lx // 2)
        x_max = self.Lx // 2 if self.Lx % 2 == 1 else (self.Lx // 2) - 1
        
        # Check all four directions
        if x > x_min:       # Left
            moves.append((x - 1, y))
        if x < x_max:       # Right
            moves.append((x + 1, y))
        if y > 0:           # Down
            moves.append((x, y - 1))
        if y < self.Ly - 1: # Up
            moves.append((x, y + 1))
            
        return moves
    
    def simulate_step(self, positions: List[Tuple[int, int]], P: float) -> List[Tuple[int, int]]:
        """
        Execute one time step of the random sequential update.
        
        Parameters:
        -----------
        positions : List[Tuple[int, int]]
            Current agent positions
        P : float
            Movement probability (0 <= P <= 1)
            
        Returns:
        --------
        List[Tuple[int, int]]
            Updated agent positions
        """
        if not 0 <= P <= 1:
            raise ValueError("Movement probability P must be in [0, 1]")
        
        if not positions:
            return positions
            
        Q = len(positions)
        new_positions = positions.copy()
        
        # Random sequential update: select Q agents with replacement
        for _ in range(Q):
            # Select random agent
            idx = np.random.randint(0, Q)
            x, y = new_positions[idx]
            
            # Agent moves with probability P
            if np.random.random() < P:
                valid_moves = self._get_valid_moves(x, y)
                
                # Choose random direction from valid moves
                if valid_moves:
                    new_x, new_y = valid_moves[np.random.randint(len(valid_moves))]
                    new_positions[idx] = (new_x, new_y)
        
        return new_positions
    
    def simulate(self, U: float, P: float, T: int, random_seed: Optional[int] = None) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Run complete simulation from initialization to final time.
        
        Parameters:
        -----------
        U : float
            Initial occupancy probability
        P : float
            Movement probability
        T : int
            Number of time steps
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        Tuple containing:
        - column_counts : np.ndarray
            Final observation vector (agent counts per column)
        - initial_positions : List[Tuple[int, int]]
            Initial agent positions
        - final_positions : List[Tuple[int, int]]
            Final agent positions
        """
        if T < 0:
            raise ValueError("Number of time steps T must be non-negative")
            
        # Initialize agents
        positions = self.initialize_lattice(U, random_seed)
        initial_positions = positions.copy()
        
        # Run simulation
        for t in range(T):
            positions = self.simulate_step(positions, P)
        
        # Get final column counts
        column_counts = self.get_column_counts(positions)
        
        return column_counts, initial_positions, positions
    
    def get_column_counts(self, positions: List[Tuple[int, int]]) -> np.ndarray:
        """
        Count agents in each column (observation vector).
        
        Parameters:
        -----------
        positions : List[Tuple[int, int]]
            Agent positions with centered coordinates
            
        Returns:
        --------
        np.ndarray
            Count of agents in each column, indexed from left to right
        """
        counts = np.zeros(self.Lx, dtype=int)
        
        # Calculate x boundaries (centered around 0)
        x_min = -(self.Lx // 2)
        x_max = self.Lx // 2 if self.Lx % 2 == 1 else (self.Lx // 2) - 1
        
        for x, y in positions:
            if x_min <= x <= x_max:  # Safety check
                # Convert centered coordinate to array index
                array_index = x - x_min
                counts[array_index] += 1
                
        return counts


# Plotting functions


def plot_lattice(
    positions: List[Tuple[int, int]], 
    Lx: int, 
    Ly: int, 
    title: str = "Lattice Configuration",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Visualize agent positions on the lattice with centered coordinates.
    
    Parameters:
    -----------
    positions : List[Tuple[int, int]]
        Agent positions as (x, y) tuples with centered coordinates
    Lx, Ly : int
        Lattice dimensions
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate x boundaries (centered around 0)
    x_min = -(Lx // 2)
    x_max = Lx // 2 if Lx % 2 == 1 else (Lx // 2) - 1
    
    # Create lattice grid
    lattice = np.zeros((Ly, Lx))
    
    # Mark agent positions
    for x, y in positions:
        if x_min <= x <= x_max and 0 <= y < Ly:
            # Convert centered coordinate to array index
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
    
    # Set grid with centered ticks
    x_ticks = range(x_min, x_max+1, max(1, Lx//10))
    y_ticks = range(0, Ly, max(1, Ly//10))
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(True, alpha=0.3)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    return fig


def plot_column_counts(
    column_counts: np.ndarray, 
    Lx: int,
    title: str = "Column Counts",
    figsize: Tuple[int, int] = (10, 4)
) -> plt.Figure:
    """
    Plot the distribution of agents across columns with centered coordinates.
    
    Parameters:
    -----------
    column_counts : np.ndarray
        Number of agents in each column
    Lx : int
        Lattice width (for coordinate conversion)
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
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
    
    plt.tight_layout()
    return fig


def plot_simulation_comparison(
    initial_positions: List[Tuple[int, int]],
    final_positions: List[Tuple[int, int]],
    column_counts: np.ndarray,
    Lx: int,
    Ly: int,
    U: float,
    P: float,
    T: int,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Create a comprehensive visualization of simulation results with centered coordinates.
    
    Parameters:
    -----------
    initial_positions, final_positions : List[Tuple[int, int]]
        Agent positions at start and end (centered coordinates)
    column_counts : np.ndarray
        Final column counts
    Lx, Ly : int
        Lattice dimensions
    U, P : float
        Simulation parameters
    T : int
        Number of time steps
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.1, wspace=0.3)
    ax_initial = fig.add_subplot(gs[0, 0])
    ax_final = fig.add_subplot(gs[1, 0], sharex=ax_initial)
    ax_counts = fig.add_subplot(gs[:, 1])
    
    # Calculate x boundaries (centered around 0)
    x_min = -(Lx // 2)
    x_max = Lx // 2 if Lx % 2 == 1 else (Lx // 2) - 1
    
    # Initial state - scatter plot
    if initial_positions:
        x_coords, y_coords = zip(*initial_positions)
        ax_initial.scatter(x_coords, y_coords, c='blue', s=20, alpha=0.7, edgecolors='darkblue', linewidth=0.5)
    
    ax_initial.set_xlim(x_min - 0.5, x_max + 0.5)
    ax_initial.set_ylim(-0.5, Ly - 0.5)
    ax_initial.set_title(f'Initial State\n({len(initial_positions)} agents)')
    ax_initial.set_ylabel('y (row)')
    ax_initial.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax_initial.tick_params(labelbottom=False)
    ax_initial.grid(True, alpha=0.3)
    ax_initial.set_aspect('equal')
    
    # Final state - scatter plot
    if final_positions:
        x_coords, y_coords = zip(*final_positions)
        ax_final.scatter(x_coords, y_coords, c='blue', s=20, alpha=0.7, edgecolors='darkblue', linewidth=0.5)
    
    ax_final.set_xlim(x_min - 0.5, x_max + 0.5)
    ax_final.set_ylim(-0.5, Ly - 0.5)
    ax_final.set_title(f'Final State (T={T})\n({len(final_positions)} agents)')
    ax_final.set_xlabel('x (centered)')
    ax_final.set_ylabel('y (row)')
    ax_final.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax_final.grid(True, alpha=0.3)
    ax_final.set_aspect('equal')
    
    # Column counts
    x_positions = np.arange(x_min, x_max + 1)
    ax_counts.bar(x_positions, column_counts, alpha=0.7, color='skyblue', edgecolor='navy')
    ax_counts.set_title(f'Final Column Counts\nU={U:.3f}, P={P:.3f}')
    ax_counts.set_xlabel('Column (x, centered)')
    ax_counts.set_ylabel('Number of agents')
    ax_counts.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax_counts.grid(True, alpha=0.3)

    return fig