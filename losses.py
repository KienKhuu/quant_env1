"""
losses.py - Tri-Channel Multi-Task Contrastive Learning Objectives
Implements:
  1. Temporal Contrastive Loss (TS2Vec Contextual Invariance)
  2. Instance Contrastive Loss (TS2Vec Global Scale Invariance)
  3. Symmetrical Spatial Contrastive Loss (Dolphin et al. Cross-Asset Co-Movement)
"""

import torch
import torch.nn as nn

def cosine_sim(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Computes cosine similarity across the latent embedding dimension."""
    u_norm = u / (u.norm(dim=-1, keepdim=True) + 1e-8)
    v_norm = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.matmul(u_norm, v_norm.transpose(-1, -2))

def compute_temporal_loss(Z_A: torch.Tensor, Z_B: torch.Tensor, temp: float = 0.5) -> torch.Tensor:
    """
    Temporal Contrastive Loss Channel (Enforces Contextual Invariance).
    Aligns identical timestamps across parallel augmented views.
    """
    B, M, T_crop, D = Z_A.shape
    z_a = Z_A.reshape(B * M, T_crop, D)
    z_b = Z_B.reshape(B * M, T_crop, D)
    sim = cosine_sim(z_a, z_b) / temp
    targets = torch.arange(T_crop, device=Z_A.device).unsqueeze(0).expand(B * M, -1)
    return nn.CrossEntropyLoss()(sim.reshape(-1, T_crop), targets.reshape(-1))

def compute_instance_loss(Z_A: torch.Tensor, Z_B: torch.Tensor, temp: float = 0.5) -> torch.Tensor:
    """
    Instance Contrastive Loss Channel (Enforces Scale Invariance).
    Aligns global macro environment representations via global max pooling over time.
    """
    B, M, T_crop, D = Z_A.shape
    z_a = torch.max(Z_A, dim=2)[0].reshape(B * M, D)
    z_b = torch.max(Z_B, dim=2)[0].reshape(B * M, D)
    sim = torch.matmul(z_a, z_b.T) / temp
    targets = torch.arange(B * M, device=Z_A.device)
    return nn.CrossEntropyLoss()(sim, targets)

def compute_spatial_loss(Z_A: torch.Tensor, Z_B: torch.Tensor, A_true: torch.Tensor, temp: float = 0.5) -> torch.Tensor:
    """
    Symmetrical Spatial Loss Channel (Enforces Sector Co-Movement & Systemic Risk).
    Pulls connected sector peers together and pushes independent assets apart.
    """
    B, M, T_crop, D = Z_A.shape
    device = Z_A.device
    z_a = torch.mean(Z_A, dim=2)
    z_b = torch.mean(Z_B, dim=2)

    z_a_norm = z_a / (z_a.norm(dim=-1, keepdim=True) + 1e-8)
    z_b_norm = z_b / (z_b.norm(dim=-1, keepdim=True) + 1e-8)

    # Pass 1: View A -> View B
    sim_ab = torch.bmm(z_a_norm, z_b_norm.transpose(1, 2)) / temp
    exp_sim_ab = torch.exp(sim_ab)
    pos_sum_ab = torch.sum(A_true * exp_sim_ab, dim=-1)
    total_sum_ab = torch.sum(exp_sim_ab, dim=-1)
    loss_ab = -torch.log((pos_sum_ab + 1e-8) / (total_sum_ab + 1e-8))

    # Pass 2: View B -> View A (Symmetrical Match)
    sim_ba = torch.bmm(z_b_norm, z_a_norm.transpose(1, 2)) / temp
    exp_sim_ba = torch.exp(sim_ba)
    pos_sum_ba = torch.sum(A_true * exp_sim_ba, dim=-1)
    total_sum_ba = torch.sum(exp_sim_ba, dim=-1)
    loss_ba = -torch.log((pos_sum_ba + 1e-8) / (total_sum_ba + 1e-8))

    return 0.5 * (torch.mean(loss_ab) + torch.mean(loss_ba))


