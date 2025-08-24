# Project Proposal: Inferring Stochastic Model Parameters via 2D Convolutional Neural Networks

## 1. Context and Motivation

The paper "Inference and Prediction for Stochastic Models of Biological Populations" details a workflow for inferring parameters (like initial occupancy **U** and movement probability **P**) from a discrete random walk model. The standard method presented involves simplifying the 2D spatial output of the simulation into a 1D vector of column counts ($N_i$)

While this 1D summary statistic is computationally efficient, it discards a significant amount of spatial information. Details about agent clustering, local density variations, and the specific shape of the migrating front are lost. This project proposes leveraging this discarded information using modern machine learning techniques for image processing.

## 2. The Proposed Approach: Direct 2D Inference with ML

The core idea is to treat the output of the discrete simulation not as a collection of column counts, but as a **2D image or density map**. This "image" would be a matrix where the value of each pixel corresponds to the number of agents at that specific lattice site.



By preserving the full 2D structure, we can use powerful machine learning models, specifically **Convolutional Neural Networks (CNNs)**, to perform inference. CNNs are the state-of-the-art for image recognition tasks because they are explicitly designed to learn and identify spatial patterns and features—exactly the type of information that is lost in the 1D summary.

## 3. The Machine Learning Workflow

The project would follow a simulation-based inference workflow:

1.  **Define Priors**: Specify prior distributions for the parameters of interest, `U` and `P`.

2.  **Generate Training Data**: Create a large dataset of parameter-image pairs.
    * For `i` in `1...M` simulations:
        * Sample a parameter set $(U_i, P_i)$ from the priors.
        * Run the full discrete random walk simulation using these parameters.
        * Convert the final 2D positions of all agents into a 2D image matrix, `image_i`.
        * Store the pair: $((U_i, P_i), \text{image}_i)$.

3.  **Train the CNN**:
    * Design a CNN that takes a 2D image as input and outputs an estimate of the posterior distribution for the parameters `(U, P)`.
    * Train this network on the `M` generated pairs. The CNN will learn to associate subtle spatial patterns in the images with the underlying parameters that created them.

4.  **Perform Inference**:
    * Take a single observation image from a "real" experiment or target simulation.
    * Feed this image through the trained CNN to obtain the posterior distribution, $p(U, P | \text{image}_{real})$, which represents the inferred parameter values and their uncertainty.

## 4. Advantages and Challenges

* **Advantages**:
    * **Richer Information**: The model can learn from complex 2D features, potentially leading to more accurate and precise parameter estimates.
    * **Broader Applicability**: This method could be applied to more complex models where a 1D summary is not meaningful (e.g., populations that split or form Turing patterns).

* **Challenges**:
    * **Computational Cost**: Generating and training on thousands of images is far more computationally expensive than using 1D vectors.
    * **Complexity**: Designing and tuning a CNN architecture requires significant expertise.