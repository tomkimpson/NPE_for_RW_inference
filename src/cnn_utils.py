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


class SpatialPyramidPooling(nn.Module):
    """
    Spatial Pyramid Pooling layer that preserves left-right asymmetry.

    Instead of global average pooling which destroys spatial asymmetry,
    this pools left/right hemispheres separately at multiple scales.

    Output channels per input channel:
    - Level 0: 1x1 global (1 value)
    - Level 1: 1x2 left/right (2 values)
    - Level 2: 2x4 grid (8 values)
    Total: 11 values per input channel
    """

    def __init__(self, levels: list = None):
        """
        Initialize Spatial Pyramid Pooling.

        Parameters:
        -----------
        levels : list of tuples
            Output sizes for each pyramid level, e.g., [(1,1), (1,2), (2,4)]
        """
        super().__init__()
        self.levels = levels or [(1, 1), (1, 2), (2, 4)]
        self.output_size_per_channel = sum(h * w for h, w in self.levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spatial pyramid pooling.

        Parameters:
        -----------
        x : torch.Tensor of shape (batch_size, channels, height, width)

        Returns:
        --------
        torch.Tensor of shape (batch_size, channels * sum(level_sizes))
        """
        batch_size = x.size(0)
        channels = x.size(1)

        outputs = []
        for h, w in self.levels:
            # Adaptive average pool to (h, w)
            pooled = F.adaptive_avg_pool2d(x, (h, w))
            # Flatten spatial dimensions
            pooled = pooled.view(batch_size, channels, h * w)
            outputs.append(pooled)

        # Concatenate all levels: (batch, channels, total_spatial)
        out = torch.cat(outputs, dim=2)
        # Flatten to (batch, channels * total_spatial)
        return out.view(batch_size, -1)


class SpatialCNN(nn.Module):
    """
    Improved Convolutional Neural Network for processing 2D spatial agent distributions.

    This network uses progressive downsampling instead of aggressive adaptive pooling
    to preserve spatial information while extracting meaningful features.

    Key architectural choices to avoid bias:
    - Per-sample normalization can be disabled (normalize_input=False) to preserve
      absolute density information that correlates with proliferation rate P.
    - Density-preserving normalization (use_density_channels=True) applies z-score
      normalization but preserves mean/std as additional input channels.
    - Spatial pyramid pooling (use_spatial_pyramid=True) preserves left-right
      asymmetry that encodes the rho parameter.
    - Auxiliary features (total count, asymmetry, center of mass) explicitly capture
      information that global pooling would otherwise lose.
    """

    def __init__(self,
                 input_height: int,
                 input_width: int,
                 output_dim: int = 256,
                 dropout: float = 0.05,
                 normalize_input: bool = True,
                 use_auxiliary_features: bool = False,
                 use_density_channels: bool = False,
                 use_spatial_pyramid: bool = False):
        """
        Initialize the Spatial CNN.

        Parameters:
        -----------
        input_height : int
            Height of input 2D grid (Ly)
        input_width : int
            Width of input 2D grid (Lx)
        output_dim : int
            Dimension of output feature vector (increased from 128 to 256)
        dropout : float
            Dropout probability (reduced from 0.1 to 0.05)
        normalize_input : bool
            If True, apply per-sample z-score normalization. Set to False to
            preserve absolute density information (important for P inference).
        use_auxiliary_features : bool
            If True, compute and concatenate auxiliary features (total count,
            left-right asymmetry, center of mass) to help with P and rho inference.
        use_density_channels : bool
            If True, use 3-channel input with density-preserving normalization:
            channel 0 = z-score normalized data, channel 1 = mean, channel 2 = log(std).
            This preserves absolute density information correlated with P.
        use_spatial_pyramid : bool
            If True, use spatial pyramid pooling instead of global average pooling.
            This preserves left-right asymmetry that encodes rho.
        """
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width
        self.output_dim = output_dim
        self.normalize_input = normalize_input
        self.use_auxiliary_features = use_auxiliary_features
        self.use_density_channels = use_density_channels
        self.use_spatial_pyramid = use_spatial_pyramid

        # Number of auxiliary features: total_count, asymmetry, x_center_of_mass
        self.n_auxiliary = 3 if use_auxiliary_features else 0

        # Input channels: 3 if using density channels, else 1
        in_channels = 3 if use_density_channels else 1

        # Progressive downsampling convolutional layers with residual connections
        # Stage 1: 50x200 -> 25x100 (stride=2)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1_residual = nn.Conv2d(in_channels, 32, kernel_size=1, stride=2)  # Residual connection

        # Stage 2: 25x100 -> 13x50 (stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2_residual = nn.Conv2d(32, 64, kernel_size=1, stride=2)

        # Stage 3: 13x50 -> 7x25 (stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3_residual = nn.Conv2d(64, 128, kernel_size=1, stride=2)

        # Stage 4: 7x25 -> 4x13 (stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4_residual = nn.Conv2d(128, 256, kernel_size=1, stride=2)

        # Pooling layer: spatial pyramid or global average
        if use_spatial_pyramid:
            # Spatial pyramid pooling preserves left-right asymmetry
            self.pooling = SpatialPyramidPooling(levels=[(1, 1), (1, 2), (2, 4)])
            # 256 channels * 11 spatial values per channel = 2816
            pooled_size = 256 * self.pooling.output_size_per_channel
        else:
            # Global average pooling (original behavior)
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
            pooled_size = 256

        # Calculate flattened size after pooling
        self.flattened_size = pooled_size + self.n_auxiliary

        # Fully connected layers with better capacity
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)

        self.dropout = nn.Dropout(dropout)

        # Initialize auxiliary feature weights to small values to avoid instability
        if use_auxiliary_features:
            with torch.no_grad():
                # Scale down the weights for auxiliary feature inputs (last 3 columns of fc1)
                self.fc1.weight[:, -3:] *= 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the improved network with progressive downsampling.

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

        batch_size = x.size(0)
        height = x.size(2)
        width = x.size(3)

        # Compute auxiliary features BEFORE normalization (to preserve absolute values)
        auxiliary_features = None
        if self.use_auxiliary_features:
            auxiliary_features = self._compute_auxiliary_features(x)

        # Normalize input data
        if self.use_density_channels:
            # Density-preserving normalization: 3-channel input
            # Channel 0: z-score normalized data
            # Channel 1: mean (constant across spatial dims)
            # Channel 2: log(std) (constant across spatial dims)
            x_flat = x.view(batch_size, -1)
            x_mean = x_flat.mean(dim=1, keepdim=True)  # (batch, 1)
            x_std = x_flat.std(dim=1, keepdim=True) + 1e-8  # (batch, 1)

            # Normalized channel
            x_normalized = (x_flat - x_mean) / x_std
            x_normalized = x_normalized.view(batch_size, 1, height, width)

            # Mean channel (broadcast to spatial dims)
            # Scale mean to reasonable range (divide by typical total count)
            mean_scaled = x_mean / (height * width)  # Normalize by grid size
            mean_channel = mean_scaled.view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)

            # Log-std channel (log transform for numerical stability)
            log_std_channel = torch.log(x_std).view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)

            # Stack to 3 channels
            x = torch.cat([x_normalized, mean_channel, log_std_channel], dim=1)

        elif self.normalize_input:
            # Per-sample z-score normalization (original behavior)
            x_flat = x.view(batch_size, -1)
            x_mean = x_flat.mean(dim=1, keepdim=True)
            x_std = x_flat.std(dim=1, keepdim=True) + 1e-8  # Add epsilon to avoid division by zero
            x_flat = (x_flat - x_mean) / x_std
            x = x_flat.view(batch_size, 1, height, width)
        else:
            # Log-transform: compresses range while preserving density information
            # log1p(x) = log(1 + x) is numerically stable and preserves relative densities
            x = torch.log1p(x)

        # Stage 1: Progressive downsampling with residual connections
        residual = self.conv1_residual(x)
        x = F.relu(self.bn1(self.conv1(x)) + residual)

        # Stage 2:
        residual = self.conv2_residual(x)
        x = F.relu(self.bn2(self.conv2(x)) + residual)

        # Stage 3:
        residual = self.conv3_residual(x)
        x = F.relu(self.bn3(self.conv3(x)) + residual)

        # Stage 4:
        residual = self.conv4_residual(x)
        x = F.relu(self.bn4(self.conv4(x)) + residual)

        # Pooling layer (spatial pyramid or global average)
        if self.use_spatial_pyramid:
            # SpatialPyramidPooling returns already flattened output
            x = self.pooling(x)
        else:
            # Global average pooling
            x = self.pooling(x)
            # Flatten for fully connected layers
            x = x.view(x.size(0), -1)

        # Concatenate auxiliary features if enabled
        if self.use_auxiliary_features and auxiliary_features is not None:
            x = torch.cat([x, auxiliary_features], dim=1)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def _compute_auxiliary_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary features that explicitly capture P and rho information.

        NOTE: sbi may standardize the data (z-score) before passing to embedding,
        so we compute features that are robust to both raw and standardized data.

        Features:
        - Total sum: overall intensity (sign indicates above/below mean if standardized)
        - Left-right asymmetry: encodes directional bias rho
        - X center of mass: encodes mean drift direction

        Parameters:
        -----------
        x : torch.Tensor of shape (batch_size, 1, height, width)
            Input 2D spatial data (may be standardized by sbi)

        Returns:
        --------
        torch.Tensor of shape (batch_size, 3)
            Auxiliary features: [total_sum_scaled, asymmetry, x_com_normalized]
        """
        batch_size = x.size(0)
        height = x.size(2)
        width = x.size(3)

        # Squeeze channel dimension for easier computation
        x_squeezed = x.squeeze(1)  # (batch_size, height, width)

        # 1. Total sum (works for both raw and standardized data)
        # For raw data: correlates with P (more agents = higher sum)
        # For standardized data: indicates if this sample has more/less agents than average
        total_sum = x_squeezed.sum(dim=(1, 2))  # (batch_size,)
        # Scale to reasonable range (standardized data has sum ~0, raw data can be large)
        total_sum_scaled = total_sum / (height * width)  # Normalize by grid size

        # 2. Left-right asymmetry (robust to standardization)
        # Use absolute values in denominator to handle negative values
        mid_x = width // 2
        left_sum = x_squeezed[:, :, :mid_x].sum(dim=(1, 2))
        right_sum = x_squeezed[:, :, mid_x:].sum(dim=(1, 2))
        # Asymmetry that works with negative values
        denom = torch.abs(right_sum) + torch.abs(left_sum) + 1e-8
        asymmetry = (right_sum - left_sum) / denom

        # 3. X center of mass (robust version)
        # Weight by values, but normalize properly
        x_coords = torch.arange(width, device=x.device, dtype=x.dtype)
        x_coords = (x_coords - width / 2) / (width / 2)  # Normalize coords to [-1, 1]
        x_coords = x_coords.view(1, 1, width).expand(batch_size, height, width)

        # Use absolute values for weighting to handle standardized data
        weights = torch.abs(x_squeezed) + 1e-8
        weight_sum = weights.sum(dim=(1, 2))
        x_com = (weights * x_coords).sum(dim=(1, 2)) / weight_sum

        # Stack features and clamp to prevent extreme values
        auxiliary = torch.stack([total_sum_scaled, asymmetry, x_com], dim=1)
        auxiliary = torch.clamp(auxiliary, -2.0, 2.0)  # Clamp to reasonable range

        return auxiliary


class OneDimensionalBranch(nn.Module):
    """
    1D branch for processing column sums from 2D spatial data.

    This branch extracts density information by summing rows to get column counts,
    then applies log1p transformation and an MLP. This preserves absolute density
    information that correlates with the proliferation rate P, which is lost when
    the 2D CNN applies z-score normalization.

    The key insight is that 1D NPE (column sums only) is UNBIASED for P inference,
    while the 2D CNN conflates density and spatial pattern information.
    """

    def __init__(self,
                 input_width: int,
                 output_dim: int = 128,
                 hidden_dims: list = None,
                 dropout: float = 0.05):
        """
        Initialize the 1D branch.

        Parameters:
        -----------
        input_width : int
            Width of input 2D grid (Lx), which becomes the input dimension after row sum
        output_dim : int
            Dimension of output feature vector
        hidden_dims : list
            Hidden layer dimensions for MLP (default: [256, 256])
        dropout : float
            Dropout probability
        """
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]

        layers = []
        in_dim = input_width
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_2d: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: sum rows to get column counts, normalize to distribution, then MLP.

        The normalization removes the total agent count (which correlates with U)
        and exposes the spatial distribution shape (which correlates with P via entropy).

        Parameters:
        -----------
        x_2d : torch.Tensor of shape (batch_size, Ly, Lx)
            Input 2D spatial data

        Returns:
        --------
        torch.Tensor of shape (batch_size, output_dim)
            Feature representation capturing distribution shape (correlates with P)
        """
        # Sum rows to get column counts: (batch, Ly, Lx) -> (batch, Lx)
        column_counts = x_2d.sum(dim=1)

        # Normalize to probability distribution (removes U correlation, exposes P via shape)
        total = column_counts.sum(dim=1, keepdim=True) + 1e-8
        column_probs = column_counts / total

        # Log transform for numerical stability (log of probabilities)
        # Add small epsilon to avoid log(0)
        column_features = torch.log(column_probs + 1e-8)

        return self.mlp(column_features)


class DualBranchCNN(nn.Module):
    """
    Dual-branch CNN architecture for unbiased 2D NPE inference.

    This architecture addresses the P-rho correlation bias by separating
    density information (for P) from spatial pattern information (for rho/U):

    - 1D Branch: Sums rows to get column counts, applies log1p + MLP.
                 This captures absolute density information for accurate P inference.
    - 2D Branch: Applies z-score normalization + CNN with spatial pyramid pooling.
                 This captures spatial patterns (left-right asymmetry) for rho/U inference.

    The two branches are concatenated and optionally fused via a linear layer.
    """

    def __init__(self,
                 input_height: int,
                 input_width: int,
                 output_dim: int = 256,
                 branch_1d_dim: int = 128,
                 branch_2d_dim: int = 128,
                 dropout: float = 0.05,
                 use_spatial_pyramid: bool = True):
        """
        Initialize the dual-branch CNN.

        Parameters:
        -----------
        input_height : int
            Height of input 2D grid (Ly)
        input_width : int
            Width of input 2D grid (Lx)
        output_dim : int
            Final output dimension after fusion
        branch_1d_dim : int
            Output dimension of 1D branch
        branch_2d_dim : int
            Output dimension of 2D branch
        dropout : float
            Dropout probability
        use_spatial_pyramid : bool
            If True, use spatial pyramid pooling in 2D branch
        """
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width

        # 1D branch: column sums -> MLP (captures P/density)
        self.branch_1d = OneDimensionalBranch(
            input_width=input_width,
            output_dim=branch_1d_dim,
            dropout=dropout
        )

        # 2D branch: z-score normalized CNN (captures rho/U spatial patterns)
        self.branch_2d = SpatialCNN(
            input_height=input_height,
            input_width=input_width,
            output_dim=branch_2d_dim,
            dropout=dropout,
            normalize_input=True,  # Z-score normalization for pattern extraction
            use_spatial_pyramid=use_spatial_pyramid
        )

        # Fusion layer: combine branch outputs
        combined_dim = branch_1d_dim + branch_2d_dim
        if combined_dim != output_dim:
            self.fusion = nn.Linear(combined_dim, output_dim)
        else:
            self.fusion = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through both branches with fusion.

        Parameters:
        -----------
        x : torch.Tensor
            Input 2D spatial data, shape (batch, Ly, Lx) or (batch, 1, Ly, Lx)

        Returns:
        --------
        torch.Tensor of shape (batch_size, output_dim)
            Fused feature representation
        """
        # Handle 4D input (batch, 1, Ly, Lx) -> (batch, Ly, Lx)
        if x.dim() == 4:
            x = x.squeeze(1)

        # 1D branch: operates on raw data (log1p inside)
        feat_1d = self.branch_1d(x)

        # 2D branch: applies z-score normalization internally
        feat_2d = self.branch_2d(x)

        # Concatenate and fuse
        combined = torch.cat([feat_1d, feat_2d], dim=1)
        return self.fusion(combined)


class DualBranchEmbeddingNet(nn.Module):
    """
    Dual-branch embedding network for use with sbi.

    This wrapper handles the input reshaping from sbi's flattened format
    and passes through to the DualBranchCNN.
    """

    def __init__(self,
                 input_height: int,
                 input_width: int,
                 output_dim: int = 256,
                 branch_1d_dim: int = 128,
                 branch_2d_dim: int = 128,
                 dropout: float = 0.05,
                 use_spatial_pyramid: bool = True):
        """
        Initialize the dual-branch embedding network.

        Parameters:
        -----------
        input_height : int
            Height of input 2D grid (Ly)
        input_width : int
            Width of input 2D grid (Lx)
        output_dim : int
            Final output dimension
        branch_1d_dim : int
            Output dimension of 1D branch
        branch_2d_dim : int
            Output dimension of 2D branch
        dropout : float
            Dropout probability
        use_spatial_pyramid : bool
            If True, use spatial pyramid pooling in 2D branch
        """
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width

        self.dual_cnn = DualBranchCNN(
            input_height=input_height,
            input_width=input_width,
            output_dim=output_dim,
            branch_1d_dim=branch_1d_dim,
            branch_2d_dim=branch_2d_dim,
            dropout=dropout,
            use_spatial_pyramid=use_spatial_pyramid
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with input reshaping for sbi compatibility.

        sbi flattens observations to (batch, Ly*Lx) before passing to
        the embedding net, so we reshape back to (batch, Ly, Lx).

        Parameters:
        -----------
        x : torch.Tensor
            Input data - either (batch, Ly, Lx) or (batch, Ly*Lx) from sbi

        Returns:
        --------
        torch.Tensor
            Dual-branch embedding
        """
        if x.dim() == 2:
            # sbi flattened: (batch, Ly*Lx) -> (batch, Ly, Lx)
            x = x.view(-1, self.input_height, self.input_width)
        return self.dual_cnn(x)


def create_dual_branch_embedding_net(
    input_height: int,
    input_width: int,
    output_dim: int = 256,
    branch_1d_dim: int = 128,
    branch_2d_dim: int = 128,
    dropout: float = 0.05,
    use_spatial_pyramid: bool = True
) -> DualBranchEmbeddingNet:
    """
    Factory function to create a dual-branch embedding network.

    Parameters:
    -----------
    input_height : int
        Height of input 2D grid (Ly)
    input_width : int
        Width of input 2D grid (Lx)
    output_dim : int
        Final output dimension (default: 256)
    branch_1d_dim : int
        Output dimension of 1D branch (default: 128)
    branch_2d_dim : int
        Output dimension of 2D branch (default: 128)
    dropout : float
        Dropout probability (default: 0.05)
    use_spatial_pyramid : bool
        If True, use spatial pyramid pooling in 2D branch (default: True)

    Returns:
    --------
    DualBranchEmbeddingNet
        Configured dual-branch embedding network
    """
    return DualBranchEmbeddingNet(
        input_height=input_height,
        input_width=input_width,
        output_dim=output_dim,
        branch_1d_dim=branch_1d_dim,
        branch_2d_dim=branch_2d_dim,
        dropout=dropout,
        use_spatial_pyramid=use_spatial_pyramid
    )


class SpatialEmbeddingNet(nn.Module):
    """
    Spatial embedding network that can be used as an embedding_net
    in SBI posterior neural networks for 2D data.
    """

    def __init__(self,
                 input_height: int,
                 input_width: int,
                 output_dim: int = 256,
                 dropout: float = 0.05,
                 normalize_input: bool = True,
                 use_auxiliary_features: bool = False,
                 use_density_channels: bool = False,
                 use_spatial_pyramid: bool = False):
        """
        Initialize the spatial embedding network.

        Parameters:
        -----------
        input_height : int
            Height of input 2D grid (Ly)
        input_width : int
            Width of input 2D grid (Lx)
        output_dim : int
            Dimension of output embedding (default: 256)
        dropout : float
            Dropout probability (default: 0.05)
        normalize_input : bool
            If True, apply per-sample z-score normalization. Set to False to
            preserve absolute density information (important for P inference).
        use_auxiliary_features : bool
            If True, compute and concatenate auxiliary features (total count,
            left-right asymmetry, center of mass) to help with P and rho inference.
        use_density_channels : bool
            If True, use 3-channel input with density-preserving normalization:
            channel 0 = z-score normalized data, channel 1 = mean, channel 2 = log(std).
        use_spatial_pyramid : bool
            If True, use spatial pyramid pooling instead of global average pooling
            to preserve left-right asymmetry.
        """
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width

        self.cnn = SpatialCNN(
            input_height=input_height,
            input_width=input_width,
            output_dim=output_dim,
            dropout=dropout,
            normalize_input=normalize_input,
            use_auxiliary_features=use_auxiliary_features,
            use_density_channels=use_density_channels,
            use_spatial_pyramid=use_spatial_pyramid
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate spatial embeddings.

        sbi flattens observations to (batch, Ly*Lx) before passing to
        the embedding net, so we reshape back to (batch, Ly, Lx) if
        the input is 2-D.

        Parameters:
        -----------
        x : torch.Tensor
            Input data — either (batch, Ly, Lx), (batch, 1, Ly, Lx),
            or (batch, Ly*Lx) from sbi's internal flattening.

        Returns:
        --------
        torch.Tensor
            Spatial embedding
        """
        if x.dim() == 2:
            # sbi flattened: (batch, Ly*Lx) -> (batch, Ly, Lx)
            x = x.view(-1, self.input_height, self.input_width)
        return self.cnn(x)


def create_spatial_embedding_net(
    input_height: int,
    input_width: int,
    output_dim: int = 256,
    dropout: float = 0.05,
    normalize_input: bool = True,
    use_auxiliary_features: bool = False,
    use_density_channels: bool = False,
    use_spatial_pyramid: bool = False
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
        Dimension of output embedding (default: 256)
    dropout : float
        Dropout probability (default: 0.05)
    normalize_input : bool
        If True, apply per-sample z-score normalization. Set to False to
        preserve absolute density information (important for P inference).
    use_auxiliary_features : bool
        If True, compute and concatenate auxiliary features (total count,
        left-right asymmetry, center of mass) to help with P and rho inference.
    use_density_channels : bool
        If True, use 3-channel input with density-preserving normalization:
        channel 0 = z-score normalized data, channel 1 = mean, channel 2 = log(std).
        This preserves absolute density information correlated with P.
    use_spatial_pyramid : bool
        If True, use spatial pyramid pooling instead of global average pooling
        to preserve left-right asymmetry that encodes rho.

    Returns:
    --------
    SpatialEmbeddingNet
        Configured spatial embedding network
    """
    return SpatialEmbeddingNet(
        input_height=input_height,
        input_width=input_width,
        output_dim=output_dim,
        dropout=dropout,
        normalize_input=normalize_input,
        use_auxiliary_features=use_auxiliary_features,
        use_density_channels=use_density_channels,
        use_spatial_pyramid=use_spatial_pyramid
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