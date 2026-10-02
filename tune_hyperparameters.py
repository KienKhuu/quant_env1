"""
tune_hyperparameters.py - Clean, Shape-Safe Bayesian Optimization Pipeline
Uses 2D [B_val, M] arrays with [:, [spy_idx]] indexing to eliminate broadcasting errors.
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import json
import warnings
import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn

from dataset import build_or_load_dataset, process_features, generate_revin_binders_and_prices, DEFAULT_TICKERS
from models import LSTMRegression, GRURegression, StandaloneTS2VecBackbone
from engine import RevINQuantEngine, predict_dl_raw_logits, set_seed
from backtester import (
    run_kasui_wei_rule_strategy,
    compute_rolling_garch_volatility,
    compute_wei_usd_metrics
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
set_seed(42)

# =========================================================================
# 1. CLEAN 2D FITNESS EVALUATOR (FROM ORIGINAL CODE)
# =========================================================================
def evaluate_fitness(preds_usd: np.ndarray, y_val_usd: np.ndarray, p_curr_val: np.ndarray,
                     actual_val_returns: np.ndarray, rsi_val: np.ndarray, garch_vol_val: np.ndarray,
                     spy_idx: int) -> float:
    # 1. Forecasting Accuracy
    _, _, val_rmse, _ = compute_wei_usd_metrics(y_val_usd[:, spy_idx], preds_usd[:, spy_idx])

    # 2. Trading Performance (Causal 2D column extraction [:, [spy_idx]])
    pred_ret = (preds_usd[:, [spy_idx]] - p_curr_val[:, [spy_idx]]) / p_curr_val[:, [spy_idx]]
    ann_ret_val, _, sharpe_val, _, n_trades_val, _ = run_kasui_wei_rule_strategy(
        pred_ret, actual_val_returns[:, [spy_idx]], rsi_val[:, [spy_idx]], garch_vol_val[:, [spy_idx]],
        entry_threshold=0.005, take_profit_mult=2.0, stop_loss_mult=1.5, max_hold_days=5, fee=0.001
    )

    activity_bonus = min(n_trades_val / 20.0, 1.0)
    fitness = sharpe_val + activity_bonus - (val_rmse / 5.0)
    return fitness


def run_all_models_tuning():
    print("\n" + "=" * 105)
    print(" UNIVERSAL BAYESIAN HYPERPARAMETER OPTIMIZATION (ALL MODELS)")
    print("=" * 105 + "\n")

    master_df = build_or_load_dataset(tickers=DEFAULT_TICKERS, start_date="2010-01-01", end_date="2023-12-31")
    feature_tensor, aligned_dates, prices_close_df = process_features(master_df, DEFAULT_TICKERS, k=20)
    spy_idx = DEFAULT_TICKERS.index('SPY')
    T = 30
    M = len(DEFAULT_TICKERS)

    # 1. Base 5D Binders
    X_all_base, _, y_norm_all, y_usd_all, mu_all, sigma_all = generate_revin_binders_and_prices(
        prices_close_df, feature_tensor, T=T, corr_threshold=0.40
    )
    decision_dates = aligned_dates[T - 1: T - 1 + len(X_all_base)]

    train_mask = (decision_dates >= "2010-01-01") & (decision_dates <= "2020-12-31")
    val_mask   = (decision_dates >= "2021-01-01") & (decision_dates <= "2021-12-31")

    X_tr_b, y_tr_norm = X_all_base[train_mask], y_norm_all[train_mask]
    val_idx = np.where(val_mask)[0]
    X_v_b, y_v_norm, y_v_usd = X_all_base[val_idx], y_norm_all[val_idx], y_usd_all[val_idx]
    mu_v, sigma_v = mu_all[val_idx], sigma_all[val_idx]

    B_tr, _, T_len, C = X_tr_b.shape
    B_val = X_v_b.shape[0]

    X_tr_dl = X_tr_b.reshape(B_tr * M, T_len, C)
    y_tr_dl = y_tr_norm.reshape(B_tr * M)
    X_v_dl  = X_v_b.reshape(B_val * M, T_len, C)

    # All validation matrices are 2D: [B_val, M]
    p_curr_val = prices_close_df.iloc[val_idx + T - 1].to_numpy()
    f0_log_returns = feature_tensor[:, :, 0]
    actual_val_returns = f0_log_returns[val_idx + T, :]
    f3_rsi = feature_tensor[:, :, 3]
    rsi_val = f3_rsi[T - 1: T - 1 + len(X_all_base), :][val_idx]

    R_train_val = f0_log_returns[(aligned_dates >= "2010-01-01") & (aligned_dates <= "2020-12-31"), :]
    garch_vol_val, _, _ = compute_rolling_garch_volatility(R_train_val, actual_val_returns)

    # -------------------------------------------------------------------------
    # A. TUNE LSTM
    # -------------------------------------------------------------------------
    print("[1/4] Tuning Sequential LSTM on 2021 Validation Set...")
    def lstm_objective(trial):
        h_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        set_seed(42)
        model = LSTMRegression(input_dim=C, hidden_dim=h_dim, dropout=0.2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_tr_dl, dtype=torch.float32)
        y_t = torch.tensor(y_tr_dl, dtype=torch.float32)
        model.train()
        for _ in range(15):
            optimizer.zero_grad()
            preds = model(X_t).squeeze(-1)
            loss = criterion(preds, y_t)
            loss.backward()
            optimizer.step()

        preds_usd = predict_dl_raw_logits(model, X_v_dl).reshape(B_val, M) * sigma_v + mu_v
        return evaluate_fitness(preds_usd, y_v_usd, p_curr_val, actual_val_returns, rsi_val, garch_vol_val, spy_idx)

    study_lstm = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_lstm.optimize(lstm_objective, n_trials=8, show_progress_bar=False)

    # -------------------------------------------------------------------------
    # B. TUNE GRU
    # -------------------------------------------------------------------------
    print("[2/4] Tuning Sequential GRU on 2021 Validation Set...")
    def gru_objective(trial):
        h_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        set_seed(42)
        model = GRURegression(input_dim=C, hidden_dim=h_dim, dropout=0.2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_tr_dl, dtype=torch.float32)
        y_t = torch.tensor(y_tr_dl, dtype=torch.float32)
        model.train()
        for _ in range(15):
            optimizer.zero_grad()
            preds = model(X_t).squeeze(-1)
            loss = criterion(preds, y_t)
            loss.backward()
            optimizer.step()

        preds_usd = predict_dl_raw_logits(model, X_v_dl).reshape(B_val, M) * sigma_v + mu_v
        return evaluate_fitness(preds_usd, y_v_usd, p_curr_val, actual_val_returns, rsi_val, garch_vol_val, spy_idx)

    study_gru = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_gru.optimize(gru_objective, n_trials=8, show_progress_bar=False)

    # -------------------------------------------------------------------------
    # C. TUNE STANDALONE TS2VEC (NO GRAPH)
    # -------------------------------------------------------------------------
    print("[3/4] Tuning Standalone TS2Vec (Ablation) on 2021 Validation Set...")
    def ts2vec_objective(trial):
        h_dim = trial.suggest_categorical("hidden_dim", [64, 128])
        n_blocks = trial.suggest_int("num_blocks", 1, 2)
        lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True)

        set_seed(42)
        backbone = StandaloneTS2VecBackbone(input_dim=C, hidden_dim=h_dim, num_blocks=n_blocks)
        head = nn.Sequential(nn.Linear(h_dim, 32), nn.ReLU(), nn.Linear(32, 1))

        opt_sup = torch.optim.AdamW(list(backbone.parameters()) + list(head.parameters()), lr=lr, weight_decay=wd)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_tr_dl, dtype=torch.float32)
        y_t = torch.tensor(y_tr_dl, dtype=torch.float32)
        backbone.train(); head.train()
        for _ in range(12):
            opt_sup.zero_grad()
            z = backbone(X_t)
            preds = head(z[:, -1, :]).squeeze(-1)
            loss = criterion(preds, y_t)
            loss.backward()
            opt_sup.step()

        backbone.eval(); head.eval()
        with torch.no_grad():
            z_v = backbone(torch.tensor(X_v_dl, dtype=torch.float32))
            preds_norm = head(z_v[:, -1, :]).squeeze(-1).numpy()
        preds_usd = preds_norm.reshape(B_val, M) * sigma_v + mu_v
        return evaluate_fitness(preds_usd, y_v_usd, p_curr_val, actual_val_returns, rsi_val, garch_vol_val, spy_idx)

    study_ts2vec = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_ts2vec.optimize(ts2vec_objective, n_trials=6, show_progress_bar=False)

    # -------------------------------------------------------------------------
    # D. TUNE FUSED MODEL (OURS) - STRICTLY THRESHOLD 0.40
    # -------------------------------------------------------------------------
    print("[4/4] Tuning Fused Spatial-Temporal Backbone on 2021 Validation Set...")
    def fused_objective(trial):
        h_dim = trial.suggest_categorical("hidden_dim", [64, 128])
        n_blocks = trial.suggest_int("num_blocks", 1, 2)
        lr = trial.suggest_float("lr", 1e-3, 4e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True)

        X_all_f, A_all_f, y_norm_all_f, y_usd_all_f, mu_all_f, sigma_all_f = generate_revin_binders_and_prices(
            prices_close_df, feature_tensor, T=T, corr_threshold=0.40
        )
        X_tr_f, y_tr_f, A_tr_f = X_all_f[train_mask], y_norm_all_f[train_mask], A_all_f[train_mask]
        X_v_f, y_v_f, A_v_f    = X_all_f[val_idx], y_norm_all_f[val_idx], A_all_f[val_idx]
        y_v_usd_f, mu_v_f, sigma_v_f = y_usd_all_f[val_idx], mu_all_f[val_idx], sigma_all_f[val_idx]

        set_seed(42)
        engine = RevINQuantEngine(num_assets=M, input_dims=C, hidden_dims=h_dim, num_blocks=n_blocks, lr=lr, device='cpu')
        for pg in engine.optimizer.param_groups: pg['weight_decay'] = wd

        engine.fit_unsupervised(X_tr_f, A_tr_f, X_v_f, A_v_f, epochs=8, patience=3)
        engine.fit_supervised(X_tr_f, A_tr_f, y_tr_f, X_v_f, A_v_f, y_v_f, epochs=10)

        preds_usd = engine.predict_usd(X_v_f, A_v_f, mu_v_f, sigma_v_f)
        return evaluate_fitness(preds_usd, y_v_usd_f, p_curr_val, actual_val_returns, rsi_val, garch_vol_val, spy_idx)

    study_fused = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_fused.optimize(fused_objective, n_trials=6, show_progress_bar=False)

    # -------------------------------------------------------------------------
    # E. SAVE PARAMETERS (PRESERVING CORR_THRESHOLD = 0.40)
    # -------------------------------------------------------------------------
    fused_best = study_fused.best_params
    fused_best["corr_threshold"] = 0.40

    all_best_params = {
        "LSTM": study_lstm.best_params,
        "GRU": study_gru.best_params,
        "TS2Vec": study_ts2vec.best_params,
        "FusedModel": fused_best
    }

    json_path = "best_hyperparameters.json"
    with open(json_path, "w") as f:
        json.dump(all_best_params, f, indent=4)

    print("\n" + "=" * 85)
    print(f" [✓] OPTUNA TUNING COMPLETED (SAVED TO '{json_path}')")
    print("=" * 85)
    print(json.dumps(all_best_params, indent=4))
    print("=" * 85 + "\n")

if __name__ == "__main__":
    run_all_models_tuning()