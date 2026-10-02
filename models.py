"""
models.py - Modular Spatial-Temporal and Baseline Neural Architectures
Features:
  1. Kipf & Welling (2017) Spectral Graph Convolutional Layer
  2. Official TS2Vec (Yue et al., 2022) Bidirectional Dilated ConvBlock
  3. Advanced Interleaved Spatial-Temporal Block with Residual Identity Highways
  4. Complete Fused Backbone & Standalone TS2Vec Backbone
  5. Baseline Recurrent Regressors (LSTM & GRU) with Dropout(0.2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. SPATIAL GCN SUB-LAYER (KIPF & WELLING 2017)
# =========================================================================
class GCNLayer(nn.Module):
    """
    Symmetric Graph Convolutional Layer.
    Applies self-loops, computes degree normalization, and performs spatial message passing.
    """
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, M, T, D_dim = H.shape
        device = H.device

        # Self-Loops
        I_M = torch.eye(M, device=device).unsqueeze(0).expand(B, -1, -1)
        A_tilde = A + I_M

        # Symmetric Degree Normalization (Renormalization Trick)
        degrees = torch.sum(A_tilde, dim=-1)
        deg_inv_sqrt = torch.pow(degrees, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        D_tilde_inv_sqrt = torch.diagonal_scatter(
            torch.zeros(B, M, M, device=device), deg_inv_sqrt, dim1=1, dim2=2
        )

        A_hat = torch.bmm(torch.bmm(D_tilde_inv_sqrt, A_tilde), D_tilde_inv_sqrt)
        H_reshaped = H.permute(0, 1, 2, 3).reshape(B, M, T * D_dim)
        H_space_raw = torch.bmm(A_hat, H_reshaped)
        return H_space_raw.reshape(B, M, T, D_dim)


# =========================================================================
# 2. OFFICIAL TS2VEC TEMPORAL SUB-LAYER (YUE ET AL., 2022)
# =========================================================================
class SamePadConv(nn.Module):
    """
    Exact TS2Vec Symmetric Convolution (from dilated_conv.py).
    Enforces bidirectional contextual padding.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, :-self.remove]
        return out


class TS2VecConvBlock(nn.Module):
    """
    Exact ConvBlock from TS2Vec (from dilated_conv.py):
    GELU -> SamePadConv1 -> GELU -> SamePadConv2 + Residual Projection
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


# =========================================================================
# 3. INTERLEAVED SPATIAL-TEMPORAL BLOCK
# =========================================================================
class AdvancedInterleavedBlock(nn.Module):
    """
    Hierarchical block combining:
    1. GCN spatial smoothing across sectors (Kipf & Welling)
    2. Official TS2Vec Bidirectional Dilated ConvBlock (Yue et al.)
    3. Temporal Instance Normalization
    4. Residual Identity Skip Connection
    """
    def __init__(self, hidden_dim: int, kernel_size: int = 3, dilation: int = 1):
        super(AdvancedInterleavedBlock, self).__init__()
        self.spatial_layer = GCNLayer()
        self.temporal_layer = TS2VecConvBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=dilation)
        self.instance_norm = nn.InstanceNorm1d(hidden_dim, affine=True)

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, M, T, D_dim = H.shape

        # 1. Spatial Cross-Asset GCN Pass
        H_space = self.spatial_layer(H, A)

        # 2. Reshape for Independent Asset Temporal Modeling
        H_temp = H_space.permute(0, 1, 3, 2)
        H_flat = H_temp.reshape(B * M, D_dim, T)

        # 3. Official TS2Vec Temporal Dilated Convolution Pass
        H_time_flat = self.temporal_layer(H_flat)

        # 4. Normalization and Activation
        H_norm_flat = self.instance_norm(H_time_flat)
        H_time = H_norm_flat.reshape(B, M, D_dim, T).permute(0, 1, 3, 2)

        # 5. Global Block Residual Highway
        return F.relu(H_time) + H


# =========================================================================
# 4. COMPLETE BACKBONES
# =========================================================================
class SpatialTemporalBackbone(nn.Module):
    """Stacked hierarchical backbone with exponentially growing dilations."""
    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, num_blocks: int = 2):
        super(SpatialTemporalBackbone, self).__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            AdvancedInterleavedBlock(hidden_dim, kernel_size=3, dilation=2 ** l)
            for l in range(num_blocks)
        ])

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        H = self.projection(X)
        for block in self.blocks:
            H = block(H, A)
        return H


class StandaloneTS2VecBackbone(nn.Module):
    """
    Standalone TS2Vec Architecture (Yue et al., 2022).
    Pure temporal multi-scale convolutional model WITHOUT graph convolutions.
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, num_blocks: int = 2):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                TS2VecConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2 ** l),
                nn.InstanceNorm1d(hidden_dim, affine=True)
            ])
            for l in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.projection(x)  # [B, T, D]
        for conv_layer, norm_layer in self.blocks:
            h_trans = h.transpose(1, 2)  # [B, D, T]
            h_conv = conv_layer(h_trans)
            h_norm = norm_layer(h_conv)
            h = F.relu(h_norm.transpose(1, 2)) + h  # Residual identity
        return h


# =========================================================================
# 5. BASELINE RECURRENT MODELS
# =========================================================================
class LSTMRegression(nn.Module):
    """Kasui Wei (2025) Section 2.2.2: 1-Layer LSTM with Dropout(0.2)"""
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])
        return self.fc(out)


class GRURegression(nn.Module):
    """Kasui Wei (2025) Section 2.2.2: 1-Layer GRU with Dropout(0.2)"""
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        out = self.dropout(h_n[-1])
        return self.fc(out)