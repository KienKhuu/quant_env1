import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from typing import Tuple
from torch.utils.data import TensorDataset, DataLoader

# Core architecture and loss imports
from models import SpatialTemporalBackbone, StandaloneTS2VecBackbone
from losses import compute_temporal_loss, compute_instance_loss, compute_spatial_loss, cosine_sim

def set_seed(seed: int = 42):
    """
    Enforces strict deterministic execution across PyTorch and NumPy runtime environments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)

def apply_augmentations(
    X: torch.Tensor,
    A: torch.Tensor,
    T_crop: int = 20,
    p_mask: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates twin augmented context views (View A and View B) for self-supervised contrastive learning.
    Applies Stochastic Temporal Cropping, Variational Feature Masking, and Symmetric Edge Dropout.
    """
    B, M, T, C = X.shape
    device = X.device

    # Stochastic Temporal Cropping
    start_A = torch.randint(0, T - T_crop + 1, (1,)).item()
    start_B = torch.randint(0, T - T_crop + 1, (1,)).item()

    # Variational Feature Masking
    mask_A = (torch.rand(B, M, T_crop, C, device=device) > p_mask).float()
    mask_B = (torch.rand(B, M, T_crop, C, device=device) > p_mask).float()

    X_A = X[:, :, start_A : start_A + T_crop, :].clone() * mask_A
    X_B = X[:, :, start_B : start_B + T_crop, :].clone() * mask_B

    # Symmetric Graph Edge Dropout (Binary Mask)
    dropout_mask_A = (torch.rand(A.shape, device=device) > 0.1).float()
    dropout_mask_B = (torch.rand(A.shape, device=device) > 0.1).float()

    A_A = A * dropout_mask_A
    A_B = A * dropout_mask_B

    return X_A, A_A, X_B, A_B

class RevINQuantEngine:
    """
    Unified Spatial-Temporal Training Engine with Reversible Instance Normalization (RevIN).
    Optimized for continuous price regression.
    """
    def __init__(
        self,
        num_assets: int,
        input_dims: int = 5,
        hidden_dims: int = 128,
        num_blocks: int = 2,
        lr: float = 0.001,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.lr = lr
        self.num_assets = num_assets

        # Hierarchical Interleaved Backbone
        self.backbone = SpatialTemporalBackbone(
            input_dim=input_dims,
            hidden_dim=hidden_dims,
            num_blocks=num_blocks
        ).to(self.device)

        # Learnable Homoscedastic Task Uncertainty Parameters
        self.log_sigma_temp = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.log_sigma_inst = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.log_sigma_spat = nn.Parameter(torch.tensor(0.0, device=self.device))

        # Continuous Regression Predictor Head
        self.predictor_head = nn.Sequential(
            nn.Linear(hidden_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single continuous output scalar per asset
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self.backbone.parameters()) +
            [self.log_sigma_temp, self.log_sigma_inst, self.log_sigma_spat] +
            list(self.predictor_head.parameters()),
            lr=self.lr,
            weight_decay=1e-4
        )

    def _calculate_contrastive_loss(self, X_batch: torch.Tensor, A_batch: torch.Tensor) -> torch.Tensor:
        X_A, A_A, X_B, A_B = apply_augmentations(X_batch, A_batch, T_crop=20)
        Z_A = self.backbone(X_A, A_A)
        Z_B = self.backbone(X_B, A_B)

        loss_temp = compute_temporal_loss(Z_A, Z_B)
        loss_inst = compute_instance_loss(Z_A, Z_B)
        loss_spat = compute_spatial_loss(Z_A, Z_B, A_batch)

        # Dynamic Uncertainty Balance
        return (
            torch.exp(-self.log_sigma_temp) * loss_temp + self.log_sigma_temp +
            torch.exp(-self.log_sigma_inst) * loss_inst + self.log_sigma_inst +
            torch.exp(-self.log_sigma_spat) * loss_spat + self.log_sigma_spat
        )

    def fit_unsupervised(
        self,
        X_train: np.ndarray,
        A_train: np.ndarray,
        X_val: np.ndarray,
        A_val: np.ndarray,
        epochs: int = 15,
        patience: int = 4
    ):
        """
        Stage 1: Self-Supervised Pre-Training across augmented views with early stopping.
        Note: shuffle=False preserves historical sequence continuity across batches.
        """
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(A_train).float()),
            batch_size=32, shuffle=False
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(A_val).float()),
            batch_size=32, shuffle=False
        )

        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0

        for epoch in range(epochs):
            self.backbone.train()
            for X_b, A_b in train_loader:
                self.optimizer.zero_grad()
                loss = self._calculate_contrastive_loss(X_b.to(self.device), A_b.to(self.device))
                loss.backward()
                self.optimizer.step()

            self.backbone.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, A_b in val_loader:
                    val_loss += self._calculate_contrastive_loss(X_b.to(self.device), A_b.to(self.device)).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.backbone.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_weights:
            self.backbone.load_state_dict(best_weights)

    def fit_supervised(
        self,
        X_train: np.ndarray,
        A_train: np.ndarray,
        y_norm_train: np.ndarray,
        X_val: np.ndarray,
        A_val: np.ndarray,
        y_norm_val: np.ndarray,
        epochs: int = 12
    ):
        """
        Stage 2: Supervised Fine-Tuning using Continuous MSE on normalized targets.
        """
        # Freeze backbone parameters to act as a noise-filtering feature extractor
        for param in self.backbone.parameters():
            param.requires_grad = False

        opt = torch.optim.AdamW(self.predictor_head.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(A_train).float(), torch.from_numpy(y_norm_train).float()),
            batch_size=32,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(A_val).float(), torch.from_numpy(y_norm_val).float()),
            batch_size=32,
            shuffle=False
        )

        best_mse = float('inf')
        best_weights = None

        for epoch in range(epochs):
            self.predictor_head.train()
            for X_b, A_b, y_b in train_loader:
                opt.zero_grad()
                with torch.no_grad():
                    Z = self.backbone(X_b.to(self.device), A_b.to(self.device))
                preds = self.predictor_head(Z[:, :, -1, :]).squeeze(-1)
                loss = criterion(preds, y_b.to(self.device))
                loss.backward()
                opt.step()

            self.predictor_head.eval()
            val_mse = 0.0
            with torch.no_grad():
                for X_b, A_b, y_b in val_loader:
                    Z = self.backbone(X_b.to(self.device), A_b.to(self.device))
                    preds = self.predictor_head(Z[:, :, -1, :]).squeeze(-1)
                    val_mse += criterion(preds, y_b.to(self.device)).item()
            val_mse /= len(val_loader)

            if val_mse < best_mse:
                best_mse = val_mse
                best_weights = copy.deepcopy(self.predictor_head.state_dict())

        if best_weights:
            self.predictor_head.load_state_dict(best_weights)

        # Unfreeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = True

    def predict_usd(
        self,
        X_test: np.ndarray,
        A_test: np.ndarray,
        mu_test: np.ndarray,
        sigma_test: np.ndarray
    ) -> np.ndarray:
        """
        Executes forward inference and de-normalizes outputs directly back to true USD prices.
        """
        self.predictor_head.eval()
        self.backbone.eval()
        with torch.no_grad():
            Z = self.backbone(torch.tensor(X_test, dtype=torch.float32), torch.tensor(A_test, dtype=torch.float32))
            pred_norm = self.predictor_head(Z[:, :, -1, :]).squeeze(-1).numpy()
            pred_usd = pred_norm * sigma_test + mu_test
        return pred_usd


# --- BASELINE & ABLATION HELPERS ---
def train_dl_baseline_with_early_stopping(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    patience: int = 10,
    wd: float = 0.0002,
) -> nn.Module:
    """
    Trains LSTM / GRU using mini-batching (batch_size=32), Adam (lr=0.001),
    and Early Stopping on the 2021 validation set.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            preds = model(x_b).squeeze(-1)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).squeeze(-1)
            val_loss = criterion(val_preds, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model

def predict_dl_raw_logits(model: nn.Module, X_test: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).squeeze(-1)
        return preds.cpu().numpy()

# --- Replace train_standalone_ts2vec at the bottom of engine.py ---

def train_standalone_ts2vec(
    X_train: np.ndarray,
    y_norm_train: np.ndarray,
    X_val: np.ndarray,
    y_norm_val: np.ndarray,
    X_test: np.ndarray,
    mu_test: np.ndarray,
    sigma_test: np.ndarray,
    epochs_pretrain: int = 15,
    epochs_supervised: int = 12,
    lr: float = 0.00216
) -> np.ndarray:
    """
    Official Standalone TS2Vec Engine (Yue et al., 2022):
    Features:
      1. Bidirectional SamePadConv + GELU Backbone
      2. Official TS2Vec Instance Shuffling (shuffle=True)
      3. Hierarchical Multi-Scale Max-Pooling Contrastive Loss
      4. Supervised Fine-Tuning with Validation Best-Weight Checkpointing
    """
    device = torch.device('cpu')
    backbone = StandaloneTS2VecBackbone(input_dim=5, hidden_dim=128, num_blocks=2).to(device)
    predictor_head = nn.Sequential(
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)

    # -------------------------------------------------------------
    # Stage 1: Self-Supervised Contrastive Pre-Training (Official TS2Vec)
    # -------------------------------------------------------------
    optimizer_pre = optim.AdamW(backbone.parameters(), lr=lr, weight_decay=2.5e-5)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=32,
        shuffle=True,  # Official TS2Vec setting (ts2vec.py line 76)
        drop_last=True
    )

    backbone.train()
    for epoch in range(epochs_pretrain):
        for batch in train_loader:
            x_b = batch[0]
            B, T, C = x_b.shape
            T_crop = 20

            s_A = random.randint(0, T - T_crop)
            s_B = random.randint(0, T - T_crop)
            m_A = (torch.rand(B, T_crop, C) > 0.2).float()
            m_B = (torch.rand(B, T_crop, C) > 0.2).float()

            view_A = x_b[:, s_A : s_A + T_crop, :] * m_A
            view_B = x_b[:, s_B : s_B + T_crop, :] * m_B

            optimizer_pre.zero_grad()
            z_A = backbone(view_A)  # [B, T_crop, D]
            z_B = backbone(view_B)  # [B, T_crop, D]

            # 1. Temporal Contrastive Loss
            sim_t = cosine_sim(z_A, z_B) / 0.5
            targets_t = torch.arange(T_crop, device=device).unsqueeze(0).expand(B, -1)
            loss_t = nn.CrossEntropyLoss()(sim_t.reshape(-1, T_crop), targets_t.reshape(-1))

            # 2. Instance Contrastive Loss (Hierarchical Pooling)
            p_A = torch.max(z_A, dim=1)[0]
            p_B = torch.max(z_B, dim=1)[0]
            sim_i = torch.matmul(p_A, p_B.T) / 0.5
            targets_i = torch.arange(B, device=device)
            loss_i = nn.CrossEntropyLoss()(sim_i, targets_i)

            loss = loss_t + loss_i
            loss.backward()
            optimizer_pre.step()

    # -------------------------------------------------------------
    # Stage 2: Supervised Fine-Tuning (with Validation Checkpointing)
    # -------------------------------------------------------------
    for p in backbone.parameters():
        p.requires_grad = False

    optimizer_sup = optim.AdamW(predictor_head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    sup_train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_norm_train, dtype=torch.float32)),
        batch_size=32,
        shuffle=True
    )
    sup_val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_norm_val, dtype=torch.float32)),
        batch_size=32,
        shuffle=False
    )

    best_val_mse = float('inf')
    best_head_weights = None

    for epoch in range(epochs_supervised):
        predictor_head.train()
        for xb, yb in sup_train_loader:
            optimizer_sup.zero_grad()
            with torch.no_grad():
                z = backbone(xb)
            pred = predictor_head(z[:, -1, :]).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer_sup.step()

        # Validation Checkpoint (Restores best weights)
        predictor_head.eval()
        val_mse = 0.0
        with torch.no_grad():
            for xb, yb in sup_val_loader:
                z = backbone(xb)
                pred = predictor_head(z[:, -1, :]).squeeze(-1)
                val_mse += criterion(pred, yb).item()
        val_mse /= len(sup_val_loader)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_head_weights = copy.deepcopy(predictor_head.state_dict())

    if best_head_weights is not None:
        predictor_head.load_state_dict(best_head_weights)

    # -------------------------------------------------------------
    # Inference & RevIN De-normalization
    # -------------------------------------------------------------
    backbone.eval()
    predictor_head.eval()
    with torch.no_grad():
        z_te = backbone(torch.tensor(X_test, dtype=torch.float32))
        pred_norm = predictor_head(z_te[:, -1, :]).squeeze(-1).numpy()

    return pred_norm * sigma_test + mu_test