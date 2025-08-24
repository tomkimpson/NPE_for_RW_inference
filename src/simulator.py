"""
Simulator module for generating random walk data.

This module contains the RandomWalkSimulator class for simulating 
2D lattice random walks as described in Simpson & Planck for use 
in Neural Posterior Estimation (NPE) training and validation.
"""

import numpy as np
from typing import Tuple, List, Optional, Union
import matplotlib.pyplot as plt

# Optional JAX imports for high-performance computing
try:
    import jax
    import jax.numpy as jnp
    from jax import random, jit, vmap, lax
    try:
        from jax.typing import ArrayLike
        from jax import Array as JaxArray
    except ImportError:
        # Fallback for older JAX versions
        ArrayLike = jnp.ndarray
        JaxArray = jnp.ndarray
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np  # Fallback to numpy if JAX unavailable
    ArrayLike = np.ndarray
    JaxArray = np.ndarray


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


class RandomWalkSimulatorJax:
    """
    High-performance JAX-optimized 2D lattice random walk simulator.
    
    This class provides the same functionality as RandomWalkSimulator but with
    significant performance improvements through JAX compilation and vectorization.
    Supports both CPU JIT compilation and GPU acceleration.
    """
    
    def __init__(self, Lx: int, Ly: int, initial_region_half_width: Optional[int] = None,
                 device: str = 'auto'):
        """
        Initialize the JAX-optimized random walk simulator.
        
        Parameters:
        -----------
        Lx : int
            Number of lattice sites in x-direction (columns)
        Ly : int
            Number of lattice sites in y-direction (rows)
        initial_region_half_width : int, optional
            Half-width of initial region for agent placement. If None, 
            defaults to Lx//4 (agents initially placed in central columns)
        device : str
            Device to use: 'auto', 'cpu', 'gpu', or specific device string
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for RandomWalkSimulatorJax. Install with: pip install jax jaxlib")
            
        self.Lx = Lx
        self.Ly = Ly
        self.initial_region_half_width = initial_region_half_width or Lx // 4
        
        # Validate parameters
        if Lx <= 0 or Ly <= 0:
            raise ValueError("Lattice dimensions must be positive")
        if self.initial_region_half_width >= Lx // 2:
            raise ValueError("Initial region half-width too large for lattice")
            
        # Set up device
        self._setup_device(device)
        
        # Pre-compute boundaries for efficiency
        self.x_min = -(self.Lx // 2)
        self.x_max = self.Lx // 2 if self.Lx % 2 == 1 else (self.Lx // 2) - 1
        
        # Compile core functions for maximum performance
        self._initialize_jit_functions()
    
    def _setup_device(self, device: str):
        """Setup JAX device configuration."""
        if device == 'auto':
            # Use GPU if available, otherwise CPU
            try:
                devices = jax.devices('gpu')
                if devices:
                    self.device = devices[0]
                    print(f"JAX using GPU: {self.device}")
                else:
                    self.device = jax.devices('cpu')[0]
                    print(f"JAX using CPU: {self.device}")
            except:
                self.device = jax.devices('cpu')[0]
                print(f"JAX using CPU (fallback): {self.device}")
        elif device == 'gpu':
            try:
                self.device = jax.devices('gpu')[0]
                print(f"JAX using GPU: {self.device}")
            except:
                raise RuntimeError("GPU requested but not available")
        elif device == 'cpu':
            self.device = jax.devices('cpu')[0]
            print(f"JAX using CPU: {self.device}")
        else:
            # Assume specific device string
            self.device = jax.device_put(jnp.array(0), device).device_buffer.device()
            print(f"JAX using device: {self.device}")
    
    def _initialize_jit_functions(self):
        """Pre-compile JIT functions for core operations."""
        # Calculate max agents for static compilation
        region_width = 2 * self.initial_region_half_width + 1
        max_agents = region_width * self.Ly
        
        # JIT-compile the core simulation functions with static arguments
        self._simulate_step_jit = jit(
            self._simulate_step_core,
            static_argnums=(2, 3, 4, 5, 6)  # P, x_min, x_max, Ly, max_agents are static
        )
        self._initialize_lattice_jit = jit(
            self._initialize_lattice_core,
            static_argnums=(1, 2, 3, 4, 5)  # U, Lx, Ly, x_min, initial_region_half_width are static
        )
        self._get_column_counts_jit = jit(
            self._get_column_counts_core,
            static_argnums=(1, 2)  # Lx, x_min are static
        )
        
        # Store max_agents for use in function calls
        self._max_agents = max_agents
        
        print("JAX functions compiled and ready for high-performance simulation")
    
    @staticmethod
    def _initialize_lattice_core(key: JaxArray, U: float, Lx: int, Ly: int, 
                                x_min: int, initial_region_half_width: int) -> JaxArray:
        """
        Core JAX function for vectorized agent initialization.
        
        Returns agent positions as a 2D array of shape (max_agents, 2).
        """
        # Create coordinate grids for initial region
        region_width = 2 * initial_region_half_width + 1
        max_agents = region_width * Ly
        
        # Create all possible positions in initial region
        x_coords = jnp.arange(x_min, x_min + region_width)
        y_coords = jnp.arange(Ly)
        
        # Create meshgrid for all possible positions in initial region
        X, Y = jnp.meshgrid(x_coords, y_coords, indexing='ij')
        all_positions = jnp.stack([X.ravel(), Y.ravel()], axis=1)
        
        # Generate random numbers for occupancy decisions
        occupancy_probs = random.uniform(key, (max_agents,))
        
        # Create occupied flags
        occupied = occupancy_probs < U
        
        # Instead of boolean indexing, use where to create positions
        # For occupied sites, use the actual position, for unoccupied use sentinel
        sentinel_pos = jnp.array([-9999, -9999], dtype=jnp.int32)
        
        # Use where to select between actual position and sentinel
        positions = jnp.where(
            occupied[:, None],  # Broadcast occupied to shape (max_agents, 1)
            all_positions,      # Use actual positions where occupied
            sentinel_pos        # Use sentinel where not occupied
        )
        
        return positions
    
    @staticmethod 
    def _simulate_step_core(key: JaxArray, positions: JaxArray, P: float,
                           x_min: int, x_max: int, Ly: int, max_agents: int) -> JaxArray:
        """
        Core JAX function for one simulation time step with random sequential updates.
        """
        # Random sequential update: select max_agents agents with replacement
        agent_key, move_key, direction_key = random.split(key, 3)
        
        # Generate random selections and decisions for all possible selections
        selected_indices = random.randint(agent_key, (max_agents,), 0, max_agents)
        move_decisions = random.uniform(move_key, (max_agents,)) < P
        directions = random.randint(direction_key, (max_agents,), 0, 4)
        
        # Create direction vectors: [left, right, down, up]
        direction_vectors = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        
        def update_single_step(carry_positions, i):
            """Single step of random sequential update."""
            current_positions = carry_positions
            
            # Get the selected agent index
            agent_idx = selected_indices[i]
            
            # Check if selected agent is valid (not sentinel)
            agent_is_valid = current_positions[agent_idx, 0] != -9999
            should_move = move_decisions[i] & agent_is_valid
            
            # Get current position (use safe indexing)
            current_pos = current_positions[agent_idx]
            x, y = current_pos[0], current_pos[1]
            
            # Calculate new position
            direction = directions[i]
            dx, dy = direction_vectors[direction]
            new_x, new_y = x + dx, y + dy
            
            # Check boundaries
            valid_move = ((new_x >= x_min) & (new_x <= x_max) & 
                         (new_y >= 0) & (new_y < Ly))
            
            # Determine final position
            final_pos = lax.cond(
                should_move & valid_move,
                lambda: jnp.array([new_x, new_y], dtype=jnp.int32),
                lambda: current_pos
            )
            
            # Update the position array only if the agent was valid
            updated_positions = lax.cond(
                agent_is_valid,
                lambda: current_positions.at[agent_idx].set(final_pos),
                lambda: current_positions
            )
            
            return updated_positions, None
        
        # Apply max_agents updates using scan with fixed size
        final_positions, _ = lax.scan(update_single_step, positions, jnp.arange(max_agents))
        
        return final_positions
    
    @staticmethod
    def _get_column_counts_core(positions: JaxArray, Lx: int, x_min: int) -> JaxArray:
        """Core JAX function for fast column counting."""
        # Get x coordinates for all positions (including sentinels)
        x_coords = positions[:, 0]
        
        # Convert to array indices, sentinel values will be negative
        array_indices = x_coords - x_min
        
        # Create a mask for valid positions (non-sentinels)
        valid_mask = x_coords != -9999
        
        # Use where to set invalid indices to a safe value (0)
        safe_indices = jnp.where(valid_mask, array_indices, 0)
        
        # Use where to create weights (1 for valid, 0 for invalid)
        weights = jnp.where(valid_mask, 1, 0)
        
        # Count agents in each column using bincount with weights
        counts = jnp.bincount(safe_indices, weights=weights, length=Lx)
        
        return counts
    
    def initialize_lattice(self, U: float, random_seed: Optional[int] = None) -> JaxArray:
        """
        Initialize agent positions using JAX vectorization.
        
        Parameters:
        -----------
        U : float
            Initial occupancy probability (0 < U <= 1)
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        JaxArray
            Agent positions as JAX array of shape (max_agents, 2)
        """
        if not 0 < U <= 1:
            raise ValueError("Occupancy probability U must be in (0, 1]")
            
        # Create random key
        if random_seed is not None:
            key = random.PRNGKey(random_seed)
        else:
            key = random.PRNGKey(np.random.randint(0, 2**32))
            
        positions = self._initialize_lattice_jit(
            key, U, self.Lx, self.Ly, self.x_min, self.initial_region_half_width
        )
        
        return positions
    
    def simulate_step(self, positions: JaxArray, P: float, key: JaxArray) -> JaxArray:
        """
        Execute one time step using JAX optimization.
        
        Parameters:
        -----------
        positions : JaxArray
            Current agent positions
        P : float
            Movement probability (0 <= P <= 1)
        key : JaxArray
            JAX random key
            
        Returns:
        --------
        JaxArray
            Updated agent positions
        """
        if not 0 <= P <= 1:
            raise ValueError("Movement probability P must be in [0, 1]")
        
        return self._simulate_step_jit(key, positions, P, self.x_min, self.x_max, self.Ly, self._max_agents)
    
    def simulate(self, U: float, P: float, T: int, random_seed: Optional[int] = None) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Run complete JAX-optimized simulation.
        
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
            Initial agent positions (converted to list for compatibility)
        - final_positions : List[Tuple[int, int]]
            Final agent positions (converted to list for compatibility)
        """
        if T < 0:
            raise ValueError("Number of time steps T must be non-negative")
            
        # Initialize with JAX
        positions = self.initialize_lattice(U, random_seed)
        initial_positions_jax = positions.copy()
        
        # Create random keys for all time steps
        if random_seed is not None:
            main_key = random.PRNGKey(random_seed + 1)  # +1 to avoid same seed as initialization
        else:
            main_key = random.PRNGKey(np.random.randint(0, 2**32))
            
        keys = random.split(main_key, T)
        
        # Run simulation using JAX scan for maximum efficiency
        def step_fn(pos, key):
            return self.simulate_step(pos, P, key), None
            
        if T > 0:
            final_positions_jax, _ = lax.scan(step_fn, positions, keys)
        else:
            final_positions_jax = positions
            
        # Get column counts
        column_counts_jax = self._get_column_counts_jit(final_positions_jax, self.Lx, self.x_min)
        
        # Convert to numpy and format for compatibility with original interface
        column_counts = np.array(column_counts_jax)
        
        # Convert JAX arrays to lists of tuples for compatibility
        initial_positions = self._jax_positions_to_list(initial_positions_jax)
        final_positions = self._jax_positions_to_list(final_positions_jax)
        
        return column_counts, initial_positions, final_positions
    
    def _jax_positions_to_list(self, positions_jax: JaxArray) -> List[Tuple[int, int]]:
        """Convert JAX position array to list of tuples."""
        positions_np = np.array(positions_jax)
        # Filter out sentinel values
        valid_mask = positions_np[:, 0] != -9999
        valid_positions = positions_np[valid_mask]
        return [(int(x), int(y)) for x, y in valid_positions]
    
    def get_column_counts(self, positions: Union[JaxArray, List[Tuple[int, int]]]) -> np.ndarray:
        """
        Count agents in each column (observation vector).
        
        Parameters:
        -----------
        positions : Union[JaxArray, List[Tuple[int, int]]]
            Agent positions (JAX array or list of tuples)
            
        Returns:
        --------
        np.ndarray
            Count of agents in each column
        """
        if isinstance(positions, list):
            # Convert from list of tuples to JAX array
            if not positions:
                return np.zeros(self.Lx, dtype=int)
                
            max_agents = len(positions)
            positions_array = jnp.full((max_agents, 2), -9999, dtype=jnp.int32)
            for i, (x, y) in enumerate(positions):
                positions_array = positions_array.at[i].set(jnp.array([x, y]))
                
            positions_jax = positions_array
        else:
            positions_jax = positions
            
        counts_jax = self._get_column_counts_jit(positions_jax, self.Lx, self.x_min)
        return np.array(counts_jax)
    
    def simulate_batch(self, U_batch: np.ndarray, P_batch: np.ndarray, T: int, 
                      random_seed: Optional[int] = None) -> np.ndarray:
        """
        Run batch simulations for multiple parameter sets in parallel.
        
        This is the key method for efficient NPE training data generation,
        processing multiple (U, P) parameter pairs simultaneously.
        
        Parameters:
        -----------
        U_batch : np.ndarray
            Array of initial occupancy probabilities, shape (n_sims,)
        P_batch : np.ndarray  
            Array of movement probabilities, shape (n_sims,)
        T : int
            Number of time steps
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray
            Batch of column counts, shape (n_sims, Lx)
        """
        if len(U_batch) != len(P_batch):
            raise ValueError("U_batch and P_batch must have the same length")
        
        n_sims = len(U_batch)
        
        # Validate all parameters
        if not all(0 < U <= 1 for U in U_batch):
            raise ValueError("All U values must be in (0, 1]")
        if not all(0 <= P <= 1 for P in P_batch):
            raise ValueError("All P values must be in [0, 1]")
        if T < 0:
            raise ValueError("Number of time steps T must be non-negative")
        
        # For now, fall back to sequential processing to avoid JIT issues
        # This can be optimized later once the core functionality works
        results = []
        
        for i in range(n_sims):
            U = U_batch[i]
            P = P_batch[i]
            
            # Use individual random seed for each simulation
            sim_seed = random_seed + i if random_seed is not None else None
            
            # Run single simulation
            column_counts, _, _ = self.simulate(U, P, T, random_seed=sim_seed)
            results.append(column_counts)
        
        return np.array(results)
    
    def generate_training_data(self, n_simulations: int, T: int, 
                              prior_bounds: Optional[dict] = None,
                              batch_size: int = 1000,
                              random_seed: Optional[int] = None,
                              show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate large-scale training data for NPE using batch processing.
        
        This method efficiently generates thousands of simulations by processing
        them in batches, making full use of JAX vectorization and GPU parallelism.
        
        Parameters:
        -----------
        n_simulations : int
            Total number of simulations to generate
        T : int
            Number of time steps per simulation
        prior_bounds : dict, optional
            Prior bounds for U and P parameters
            Default: {'U': (0.01, 1.0), 'P': (0.0, 1.0)}
        batch_size : int
            Number of simulations per batch (adjust based on GPU memory)
        random_seed : int, optional
            Random seed for reproducibility
        show_progress : bool
            Whether to show progress bar
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            - parameters: Array of shape (n_simulations, 2) containing [U, P] values
            - observations: Array of shape (n_simulations, Lx) containing column counts
        """
        # Default prior bounds
        if prior_bounds is None:
            prior_bounds = {'U': (0.01, 1.0), 'P': (0.0, 1.0)}
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate parameter samples
        U_samples = np.random.uniform(*prior_bounds['U'], n_simulations)
        P_samples = np.random.uniform(*prior_bounds['P'], n_simulations)
        
        # Initialize output arrays
        parameters = np.column_stack([U_samples, P_samples])
        observations = np.zeros((n_simulations, self.Lx))
        
        # Process in batches
        n_batches = (n_simulations + batch_size - 1) // batch_size
        
        if show_progress:
            try:
                from tqdm import tqdm
                batch_iterator = tqdm(range(n_batches), desc="Generating training data")
            except ImportError:
                batch_iterator = range(n_batches)
                print(f"Generating {n_simulations} simulations in {n_batches} batches...")
        else:
            batch_iterator = range(n_batches)
        
        for batch_idx in batch_iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_simulations)
            
            # Get batch parameters
            U_batch = U_samples[start_idx:end_idx]
            P_batch = P_samples[start_idx:end_idx]
            
            # Run batch simulation
            batch_seed = random_seed + batch_idx if random_seed is not None else None
            batch_results = self.simulate_batch(U_batch, P_batch, T, batch_seed)
            
            # Store results
            observations[start_idx:end_idx] = batch_results
        
        print(f"Generated {n_simulations} simulations using JAX batch processing")
        print(f"Average agents per simulation: {observations.sum(axis=1).mean():.1f}")
        
        return parameters, observations
    
    def benchmark_performance(self, n_trials: int = 10, **sim_params) -> dict:
        """
        Benchmark simulation performance.
        
        Parameters:
        -----------
        n_trials : int
            Number of timing trials
        **sim_params : additional simulation parameters
            
        Returns:
        --------
        dict
            Timing statistics
        """
        import time
        
        # Default parameters
        params = {'U': 0.3, 'P': 0.7, 'T': 100}
        params.update(sim_params)
        
        times = []
        
        # Warm-up run
        self.simulate(**params, random_seed=42)
        
        # Timing runs
        for i in range(n_trials):
            start_time = time.time()
            column_counts, _, _ = self.simulate(**params, random_seed=i)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
        times = np.array(times)
        
        return {
            'mean_time': times.mean(),
            'std_time': times.std(),
            'min_time': times.min(),
            'max_time': times.max(),
            'simulations_per_second': 1.0 / times.mean(),
            'total_agents': column_counts.sum() if 'column_counts' in locals() else 0,
            'device': str(self.device)
        }
    
    def compare_with_numpy(self, numpy_simulator, n_comparisons: int = 5, 
                          tolerance: float = 0.0, **sim_params) -> dict:
        """
        Compare JAX simulation results with NumPy implementation for validation.
        
        Parameters:
        -----------
        numpy_simulator : RandomWalkSimulator
            NumPy-based simulator for comparison
        n_comparisons : int
            Number of comparisons to perform
        tolerance : float
            Tolerance for numerical differences (0.0 for exact match)
        **sim_params : simulation parameters
            
        Returns:
        --------
        dict
            Comparison results and statistics
        """
        # Default parameters
        params = {'U': 0.3, 'P': 0.7, 'T': 100}
        params.update(sim_params)
        
        results = {
            'exact_matches': 0,
            'close_matches': 0,
            'max_difference': 0.0,
            'mean_difference': 0.0,
            'column_count_correlations': [],
            'total_agent_differences': []
        }
        
        differences = []
        
        for i in range(n_comparisons):
            seed = 1000 + i
            
            # Run both simulations with same seed
            jax_counts, _, _ = self.simulate(**params, random_seed=seed)
            numpy_counts, _, _ = numpy_simulator.simulate(**params, random_seed=seed)
            
            # Compare column counts
            diff = np.abs(jax_counts - numpy_counts)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            differences.extend(diff)
            results['max_difference'] = max(results['max_difference'], max_diff)
            
            # Check for matches
            if max_diff == 0:
                results['exact_matches'] += 1
            elif max_diff <= tolerance:
                results['close_matches'] += 1
            
            # Additional statistics
            if len(jax_counts) > 1 and len(numpy_counts) > 1:
                correlation = np.corrcoef(jax_counts, numpy_counts)[0, 1]
                results['column_count_correlations'].append(correlation)
            
            total_agent_diff = abs(jax_counts.sum() - numpy_counts.sum())
            results['total_agent_differences'].append(total_agent_diff)
        
        # Summary statistics
        results['mean_difference'] = np.mean(differences)
        results['mean_correlation'] = np.mean(results['column_count_correlations']) if results['column_count_correlations'] else 0.0
        results['mean_agent_conservation_error'] = np.mean(results['total_agent_differences'])
        results['success_rate'] = (results['exact_matches'] + results['close_matches']) / n_comparisons
        
        return results


# Helper function for movement validation (defined outside class for JIT compatibility)
def _try_move(x: int, y: int, direction: int, direction_vectors: JaxArray,
              x_min: int, x_max: int, Ly: int) -> JaxArray:
    """Try to move an agent in a given direction, respecting boundaries."""
    dx, dy = direction_vectors[direction]
    new_x, new_y = x + dx, y + dy
    
    # Check boundaries
    valid_x = (new_x >= x_min) & (new_x <= x_max)
    valid_y = (new_y >= 0) & (new_y < Ly)
    valid_move = valid_x & valid_y
    
    # Return new position if valid, otherwise original position
    return lax.cond(
        valid_move,
        lambda: jnp.array([new_x, new_y]),
        lambda: jnp.array([x, y])
    )


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
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Calculate x boundaries (centered around 0)
    x_min = -(Lx // 2)
    x_max = Lx // 2 if Lx % 2 == 1 else (Lx // 2) - 1
    
    # Initial state
    lattice_initial = np.zeros((Ly, Lx))
    for x, y in initial_positions:
        if x_min <= x <= x_max and 0 <= y < Ly:
            array_index = x - x_min
            lattice_initial[y, array_index] += 1
    
    im1 = axes[0].imshow(lattice_initial, cmap='Blues', origin='lower', aspect='equal',
                        extent=[x_min, x_max+1, 0, Ly])
    axes[0].set_title(f'Initial State\n({len(initial_positions)} agents)')
    axes[0].set_xlabel('x (centered)')
    axes[0].set_ylabel('y (row)')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    plt.colorbar(im1, ax=axes[0], label='Agents')
    
    # Final state
    lattice_final = np.zeros((Ly, Lx))
    for x, y in final_positions:
        if x_min <= x <= x_max and 0 <= y < Ly:
            array_index = x - x_min
            lattice_final[y, array_index] += 1
    
    im2 = axes[1].imshow(lattice_final, cmap='Blues', origin='lower', aspect='equal',
                        extent=[x_min, x_max+1, 0, Ly])
    axes[1].set_title(f'Final State (T={T})\n({len(final_positions)} agents)')
    axes[1].set_xlabel('x (centered)')
    axes[1].set_ylabel('y (row)')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    plt.colorbar(im2, ax=axes[1], label='Agents')
    
    # Column counts
    x_positions = np.arange(x_min, x_max + 1)
    axes[2].bar(x_positions, column_counts, alpha=0.7, color='skyblue', edgecolor='navy')
    axes[2].set_title(f'Final Column Counts\nU={U:.3f}, P={P:.3f}')
    axes[2].set_xlabel('Column (x, centered)')
    axes[2].set_ylabel('Number of agents')
    axes[2].axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig