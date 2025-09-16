"""
CNN utilities for processing 2D spatial data in NPE inference.

This module provides convolutional neural network architectures
specifically designed for processing 2D spatial agent distributions
from random walk simulations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SpatialCNN(nn.Module):
    """
    Convolutional Neural Network for processing 2D spatial agent distributions.

    This network extracts spatial features from 2D grids and outputs
    feature representations suitable for parameter estimation.
    """

    def __init__(self,
                 input_height: int,
                 input_width: int,
                 output_dim: int = 128,
                 dropout: float = 0.1):
        """
        Initialize the Spatial CNN.

        Parameters:
        -----------
        input_height : int
            Height of input 2D grid (Ly)
        input_width : int
            Width of input 2D grid (Lx)
        output_dim : int
            Dimension of output feature vector
        dropout : float
            Dropout probability
        """
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width
        self.output_dim = output_dim

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Calculate flattened size after adaptive pooling
        self.flattened_size = 128 * 4 * 4  # 128 channels, 4x4 spatial

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters:
        -----------
        x : torch.Tensor of shape (batch_size, height, width)
            Input 2D spatial data

        Returns:
        --------
        torch.Tensor of shape (batch_size, output_dim)
            Extracted feature representation
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch_size, 1, height, width)

        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Adaptive pooling to standardize spatial dimensions
        x = self.adaptive_pool(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class SpatialEmbeddingNet(nn.Module):
    """
    Spatial embedding network that can be used as an embedding_net
    in SBI posterior neural networks for 2D data.
    """

    def __init__(self,
                 input_height: int,
                 input_width: int,
                 output_dim: int = 128,
                 dropout: float = 0.1):
        """
        Initialize the spatial embedding network.

        Parameters:
        -----------
        input_height : int
            Height of input 2D grid (Ly)
        input_width : int
            Width of input 2D grid (Lx)
        output_dim : int
            Dimension of output embedding
        dropout : float
            Dropout probability
        """
        super().__init__()

        self.cnn = SpatialCNN(
            input_height=input_height,
            input_width=input_width,
            output_dim=output_dim,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate spatial embeddings.

        Parameters:
        -----------
        x : torch.Tensor
            Input 2D spatial data

        Returns:
        --------
        torch.Tensor
            Spatial embedding
        """
        return self.cnn(x)


def create_spatial_embedding_net(
    input_height: int,
    input_width: int,
    output_dim: int = 128,
    dropout: float = 0.1
) -> SpatialEmbeddingNet:
    """
    Factory function to create a spatial embedding network.

    Parameters:
    -----------
    input_height : int
        Height of input 2D grid (Ly)
    input_width : int
        Width of input 2D grid (Lx)
    output_dim : int
        Dimension of output embedding
    dropout : float
        Dropout probability

    Returns:
    --------
    SpatialEmbeddingNet
        Configured spatial embedding network
    """
    return SpatialEmbeddingNet(
        input_height=input_height,
        input_width=input_width,
        output_dim=output_dim,
        dropout=dropout
    )


def compute_2d_tensor_shape(Lx: int, Ly: int) -> Tuple[int, int]:
    """
    Compute expected tensor shape for 2D spatial data.

    Parameters:
    -----------
    Lx : int
        Width of lattice
    Ly : int
        Height of lattice

    Returns:
    --------
    Tuple[int, int]
        (height, width) for tensor shape (Ly, Lx)
    """
    return (Ly, Lx)


def validate_2d_input(x: torch.Tensor, expected_height: int, expected_width: int) -> None:
    """
    Validate that input tensor has correct 2D spatial dimensions.

    Parameters:
    -----------
    x : torch.Tensor
        Input tensor to validate
    expected_height : int
        Expected height (Ly)
    expected_width : int
        Expected width (Lx)

    Raises:
    -------
    ValueError
        If tensor dimensions don't match expected shape
    """
    if x.dim() not in [2, 3, 4]:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got {x.dim()}D")

    # Get spatial dimensions (last two dimensions)
    actual_height, actual_width = x.shape[-2:]

    if actual_height != expected_height or actual_width != expected_width:
        raise ValueError(
            f"Expected spatial dimensions ({expected_height}, {expected_width}), "
            f"got ({actual_height}, {actual_width})"
        )