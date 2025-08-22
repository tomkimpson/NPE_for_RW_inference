# Problem Summary: Inferring Parameters for a Stochastic Random Walk Model with NPE

This document outlines the problem of inferring microscopic parameters for a stochastic random walk model of a biological population, as described in [Simpson \& Planck](https://www.biorxiv.org/content/10.1101/2025.05.25.656057v4). The goal is to use **Neural Posterior Estimation (NPE)** for parameter inference. 

---
## ## Goal

The primary objective is to infer the key microscopic parameters of the discrete random walk model from a single vector of observation data, $y^{obs}$. The parameters of interest are:
* **U**: The **initial occupancy probability**. This is the probability that a site within the starting region contains an agent at $t=0$.
* **P**: The **movement probability**. This is the probability that a selected agent will move during a given time step.

NPE will be used to learn the posterior distribution $p(U, P | y^{obs})$.

---
## ## The Generative Model (Forward Process)

To train an NPE model, we need to be able to generate training data. This involves running the forward simulation, which maps a set of parameters $(U, P)$ to an observation vector $y^{obs}$. This process is based on the discrete model from the paper.

### ### Step 1: Model Setup
* The simulation takes place on a two-dimensional square lattice with dimensions $L_x \times L_y$ and lattice spacing $\Delta$.
* Time progresses in discrete steps of duration $\tau$.
* For simplicity, we use a dimensionless setup where $\Delta=1$ and $\tau=1$.

### ### Step 2: Initialization (The Role of U)
* At time $t=0$, the simulation is initialized based on the parameter **U**.
* For each lattice site $(i, j)$ within a specified initial region (e.g., $|x| \le h$):
    * The site is occupied by one agent with probability **U**.
    * The site is left empty with probability **1 - U**.
* The total initial number of agents, **Q**, is the outcome of these random placements.

### ### Step 3: Simulation/Evolution (The Role of P)
* The system evolves for a set number of time steps, `T`.
* Within each time step, a "random sequential update" method is used: **Q** agents are selected randomly, with replacement, to be considered for movement.
* For each selected agent, a two-step random event occurs:
    1.  The agent moves with probability **P** and stays stationary with probability **1 - P**.
    2.  *If* the agent moves, it chooses one of the four nearest neighboring sites with uniform probability (1/4 for each direction).
* The simulation uses **zero-flux boundary conditions**, meaning any move that would take an agent off the lattice is aborted.

### ### Step 4: Observation
* After the final time step `T`, an observation is made.
* The number of agents in each vertical column, $N_i$, is counted. The formula is:
    $$N_{i}(t)=\sum_{j=1}^{J}n_{i,j}(t)$$
    where $n_{i,j}(t)$ is the number of agents at site $(i, j)$ at time $t$.
* The final observation is a vector of these counts, $y^{obs} = (N_1, N_2, \dots, N_I)$.

---
## ## The Inference Task (Inverse Problem with NPE)

The inverse problem is to take a single observation, $y^{obs}_{real}$, and determine the parameters $(U, P)$ that likely produced it.

### ### Step 1: Define Priors
* Define a prior distribution for each parameter. This represents our initial belief about their possible values before seeing the data. For example:
    * $U \sim \text{Uniform}(0, 1)$
    * $P \sim \text{Uniform}(0, 1)$

### ### Step 2: Generate Training Data
* This is the core simulation loop for training the NPE model.
* Repeat for a large number of simulations, `M`:
    1.  Draw a parameter sample $(U_i, P_i)$ from their prior distributions.
    2.  Run the full **Generative Model** (Steps 1-4 above) using these parameters to produce an observation vector, $y^{obs}_i$.
    3.  Store the pair: $((U_i, P_i), y^{obs}_i)$.

### ### Step 3: Train the Neural Network
* Use the generated dataset of `M` parameter-observation pairs to train a neural density estimator (e.g., a normalizing flow).
* The network learns a universal mapping that can approximate the posterior distribution $p(U, P | y^{obs})$ for *any* observation $y^{obs}$.

### ### Step 4: Perform Inference
* Take the single, real observation data, $y^{obs}_{real}$.
* Feed this observation into the trained network.
* The network's output is the approximate posterior distribution, $p(U, P | y^{obs}_{real})$. From this distribution, you can calculate point estimates (like the mean) and credible intervals to quantify the uncertainty in your estimates for **U** and **P**.