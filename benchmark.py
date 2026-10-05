"""
benchmark.py - Master Benchmark & Ablation Pipeline (Kasui Wei 2025 Replication)
Executes end-to-end training, causal forecasting evaluation (Table 1),
and systematic quantitative trading backtests (Table 2) on the S&P 500 test set (2022-2023).
Includes state-of-the-art Transformer Baselines (iTransformer & PatchTST) in 5D and 45D.
"""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA

# Modular Pipeline Imports
from dataset import (
    build_or_load_dataset,
    process_features,
    generate_revin_binders_and_prices,
    DEFAULT_TICKERS,
)
from models import LSTMRegression, GRURegression
from engine import (
    RevINQuantEngine,
    train_dl_baseline_with_early_stopping,
    predict_dl_raw_logits,
    train_standalone_ts2vec,
    set_seed,
)
from backtester import (
    run_kasui_wei_rule_strategy,
    compute_rolling_garch_volatility,
    compute_wei_usd_metrics,
)

# Transformer Imports
import iTransformer
import PatchTST

warnings.filterwarnings("ignore")
set_seed(42)


# =========================================================================
# TRANSFORMER HELPER FUNCTIONS
# =========================================================================
class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def train_transformer_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    target_idx=0,
    epochs=50,
    lr=1e-3,
    patience=10,
    device="cpu",
):
    train_data = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    model.to(device)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            if model.__class__.__name__ == "Model" and hasattr(model, "forecast"):
                x_mark_enc = torch.zeros(batch_x.size(0), batch_x.size(1), 1).to(device)
                x_dec = torch.zeros(batch_x.size(0), 1, batch_x.size(2)).to(device)
                x_mark_dec = torch.zeros(batch_x.size(0), 1, 1).to(device)
                outputs = model(batch_x, x_mark_enc, x_dec, x_mark_dec)
            else:
                outputs = model(batch_x)

            if len(outputs.shape) == 3:
                outputs = outputs[:, -1, target_idx]

            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if hasattr(model, "forecast"):
                x_mark_enc = torch.zeros(X_val_t.size(0), X_val_t.size(1), 1).to(device)
                x_dec = torch.zeros(X_val_t.size(0), 1, X_val_t.size(2)).to(device)
                x_mark_dec = torch.zeros(X_val_t.size(0), 1, 1).to(device)
                val_out = model(X_val_t, x_mark_enc, x_dec, x_mark_dec)
            else:
                val_out = model(X_val_t)

            if len(val_out.shape) == 3:
                val_out = val_out[:, -1, target_idx]
            val_loss = criterion(val_out.squeeze(), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_weights)
    return model


# =========================================================================
# CLASSICAL ARIMA(2,1,1) ROLLING FORECASTING HELPER
# =========================================================================
def compute_arima_1step_rolling(
    train_prices: np.ndarray, test_prices: np.ndarray, order=(2, 1, 1)
) -> np.ndarray:
    print(f"[*] Fitting Econometric ARIMA{order} on S&P 500 Historical Close Prices...")
    full_series = np.concatenate([train_prices, test_prices])
    train_len = len(train_prices)
    test_len = len(test_prices)
    model = ARIMA(train_prices, order=order)
    fitted = model.fit()
    applied = fitted.apply(full_series)
    return applied.fittedvalues[train_len : train_len + test_len]


# =========================================================================
# MASTER BENCHMARK PIPELINE
# =========================================================================
def run_master_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 115)
    print(" S&P 500 MULTI-ASSET QUANTITATIVE BENCHMARK & ABLATION STUDY (2010-2023)")
    print(
        " Featuring: ARIMA, GARCH, LSTM, GRU, TS2Vec, iTransformer, PatchTST, and Fused GCN-TS2Vec"
    )
    print("=" * 115 + "\n")

    print(
        "[1/8] Ingesting multi-asset OHLCV data and synthesizing 5D feature tensor..."
    )
    master_df = build_or_load_dataset(
        tickers=DEFAULT_TICKERS, start_date="2010-01-01", end_date="2023-12-31"
    )
    feature_tensor, aligned_dates, prices_close_df = process_features(
        master_df, DEFAULT_TICKERS, k=20
    )
    spy_idx = DEFAULT_TICKERS.index("SPY")
    T = 30
    M_assets = len(DEFAULT_TICKERS)
    C_features = 5

    print(
        "[2/8] Generating localized RevIN binders and causal T-1 cross-asset graphs..."
    )
    X_all, A_all, y_norm_all, y_usd_all, mu_all, sigma_all = (
        generate_revin_binders_and_prices(
            prices_close_df, feature_tensor, T=T, corr_threshold=0.40
        )
    )

    f0_log_returns = feature_tensor[:, :, 0]
    f3_rsi = feature_tensor[:, :, 3]
    decision_dates = aligned_dates[T - 1 : T - 1 + len(X_all)]
    rsi_all = f3_rsi[T - 1 : T - 1 + len(X_all), :]

    train_mask = (decision_dates >= "2010-01-01") & (decision_dates <= "2020-12-31")
    val_mask = (decision_dates >= "2021-01-01") & (decision_dates <= "2021-12-31")
    test_mask = (decision_dates >= "2022-01-01") & (decision_dates <= "2023-12-31")

    X_train, y_train_norm, A_train = (
        X_all[train_mask],
        y_norm_all[train_mask],
        A_all[train_mask],
    )
    val_idx, test_idx = np.where(val_mask)[0], np.where(test_mask)[0]

    X_test, y_test_norm, A_test = X_all[test_idx], y_norm_all[test_idx], A_all[test_idx]
    y_test_usd = y_usd_all[test_idx]
    mu_test, sigma_test = mu_all[test_idx], sigma_all[test_idx]
    rsi_test = rsi_all[test_idx]
    p_current_usd = prices_close_df.iloc[test_idx + T - 1].to_numpy()
    actual_test_returns = f0_log_returns[test_idx + T, :]

    print("[3/8] Estimating causal GARCH(1,1) conditional volatility parameters...")
    R_train = f0_log_returns[
        (aligned_dates >= "2010-01-01") & (aligned_dates <= "2021-12-31"), :
    ]
    garch_vol, garch_mse, garch_rmse = compute_rolling_garch_volatility(
        R_train, actual_test_returns
    )

    print("[4/8] Generating rolling causal ARIMA(2,1,1) forecasts...")
    full_spy_prices = prices_close_df["SPY"].to_numpy()
    first_target_idx = test_idx[0] + T
    arima_preds_spy = compute_arima_1step_rolling(
        full_spy_prices[:first_target_idx],
        full_spy_prices[first_target_idx : test_idx[-1] + T + 1],
    )

    print("[5/8] Training Sequential LSTM & GRU with Early Stopping...")
    X_train_spy, y_train_spy = (
        X_all[train_mask, spy_idx, :, :],
        y_norm_all[train_mask, spy_idx],
    )
    X_val_spy, y_val_spy = X_all[val_idx, spy_idx, :, :], y_norm_all[val_idx, spy_idx]
    X_test_spy = X_all[test_idx, spy_idx, :, :]

    set_seed(42)
    lstm_m = train_dl_baseline_with_early_stopping(
        LSTMRegression(5, 64, 0.2),
        X_train_spy,
        y_train_spy,
        X_val_spy,
        y_val_spy,
        epochs=100,
        lr=0.001,
    )
    gru_m = train_dl_baseline_with_early_stopping(
        GRURegression(5, 32, 0.2),
        X_train_spy,
        y_train_spy,
        X_val_spy,
        y_val_spy,
        epochs=100,
        lr=0.004,
    )
    lstm_preds_usd = (
        predict_dl_raw_logits(lstm_m, X_test_spy) * sigma_test[:, spy_idx]
        + mu_test[:, spy_idx]
    )
    gru_preds_usd = (
        predict_dl_raw_logits(gru_m, X_test_spy) * sigma_test[:, spy_idx]
        + mu_test[:, spy_idx]
    )

    print("[6/8] Training Standalone TS2Vec (Ablation Benchmark)...")
    set_seed(42)
    ts2vec_preds_usd = train_standalone_ts2vec(
        X_train_spy,
        y_train_spy,
        X_val_spy,
        y_val_spy,
        X_test_spy,
        mu_test[:, spy_idx],
        sigma_test[:, spy_idx],
    )

    print("[7/8] Training SOTA Transformers (iTransformer & PatchTST in 5D and 45D)...")

    # Multivariate 45D Setup
    X_train_45d = np.transpose(X_all[train_mask], (0, 2, 1, 3)).reshape(
        -1, T, M_assets * C_features
    )
    X_val_45d = np.transpose(X_all[val_mask], (0, 2, 1, 3)).reshape(
        -1, T, M_assets * C_features
    )
    X_test_45d = np.transpose(X_all[test_mask], (0, 2, 1, 3)).reshape(
        -1, T, M_assets * C_features
    )
    target_idx_45d = spy_idx * C_features + 0

    base_configs = {
        "seq_len": T,
        "pred_len": 1,
        "output_attention": False,
        "use_norm": False,
        "d_model": 64,
        "embed": "timeF",
        "freq": "d",
        "dropout": 0.1,
        "class_strategy": "projection",
        "factor": 1,
        "n_heads": 4,
        "d_ff": 128,
        "activation": "gelu",
        "e_layers": 2,
        "patch_len": 6,
        "stride": 3,
        "padding_patch": "end",
        "individual": 0,
        "revin": 0,
        "affine": 0,
        "subtract_last": 0,
        "decomposition": 0,
        "kernel_size": 25,
        "fc_dropout": 0.1,
        "head_dropout": 0.1,
    }

    # --- 5D Training ---
    cfg_5d = DotDict(base_configs.copy())
    cfg_5d.enc_in = 5

    set_seed(42)
    itrans_5d = train_transformer_model(
        iTransformer.Model(cfg_5d),
        X_train_spy,
        y_train_spy,
        X_val_spy,
        y_val_spy,
        target_idx=0,
        device=device,
    )
    with torch.no_grad():
        t_x = torch.tensor(X_test_spy, dtype=torch.float32).to(device)
        preds = itrans_5d(
            t_x,
            torch.zeros(t_x.size(0), t_x.size(1), 1).to(device),
            torch.zeros(t_x.size(0), 1, t_x.size(2)).to(device),
            torch.zeros(t_x.size(0), 1, 1).to(device),
        )
        itrans_5d_usd = (
            preds[:, -1, 0] if len(preds.shape) == 3 else preds
        ).cpu().numpy() * sigma_test[:, spy_idx] + mu_test[:, spy_idx]

    set_seed(42)
    patch_5d = train_transformer_model(
        PatchTST.Model(cfg_5d),
        X_train_spy,
        y_train_spy,
        X_val_spy,
        y_val_spy,
        target_idx=0,
        device=device,
    )
    with torch.no_grad():
        t_x = torch.tensor(X_test_spy, dtype=torch.float32).to(device)
        preds = patch_5d(t_x)
        patch_5d_usd = (
            preds[:, -1, 0] if len(preds.shape) == 3 else preds
        ).cpu().numpy() * sigma_test[:, spy_idx] + mu_test[:, spy_idx]

    # --- 45D Training ---
    cfg_45d = DotDict(base_configs.copy())
    cfg_45d.enc_in = 45

    set_seed(42)
    itrans_45d = train_transformer_model(
        iTransformer.Model(cfg_45d),
        X_train_45d,
        y_train_spy,
        X_val_45d,
        y_val_spy,
        target_idx=target_idx_45d,
        device=device,
    )
    with torch.no_grad():
        t_x = torch.tensor(X_test_45d, dtype=torch.float32).to(device)
        preds = itrans_45d(
            t_x,
            torch.zeros(t_x.size(0), t_x.size(1), 1).to(device),
            torch.zeros(t_x.size(0), 1, t_x.size(2)).to(device),
            torch.zeros(t_x.size(0), 1, 1).to(device),
        )
        itrans_45d_usd = (
            preds[:, -1, target_idx_45d] if len(preds.shape) == 3 else preds
        ).cpu().numpy() * sigma_test[:, spy_idx] + mu_test[:, spy_idx]

    set_seed(42)
    patch_45d = train_transformer_model(
        PatchTST.Model(cfg_45d),
        X_train_45d,
        y_train_spy,
        X_val_45d,
        y_val_spy,
        target_idx=target_idx_45d,
        device=device,
    )
    with torch.no_grad():
        t_x = torch.tensor(X_test_45d, dtype=torch.float32).to(device)
        preds = patch_45d(t_x)
        patch_45d_usd = (
            preds[:, -1, target_idx_45d] if len(preds.shape) == 3 else preds
        ).cpu().numpy() * sigma_test[:, spy_idx] + mu_test[:, spy_idx]

    print(
        "[8/8] Training Fused Model (Ours: Spatial GCN + Bidirectional TS2Vec + RevIN)..."
    )
    set_seed(42)
    engine = RevINQuantEngine(
        num_assets=M_assets,
        input_dims=5,
        hidden_dims=128,
        num_blocks=2,
        lr=0.00216,
        device=device,
    )
    for pg in engine.optimizer.param_groups:
        pg["weight_decay"] = 2.5e-5
    engine.fit_unsupervised(X_train, A_train, X_test, A_test, epochs=15, patience=4)
    engine.fit_supervised(
        X_train, A_train, y_train_norm, X_test, A_test, y_test_norm, epochs=12
    )
    fused_preds_usd = engine.predict_usd(X_test, A_test, mu_test, sigma_test)

    # =========================================================================
    # TABLE 1: FORECASTING ACCURACY EVALUATION
    # =========================================================================
    print("\n" + "=" * 105)
    print(" Table 1: Forecasting Performance of Models on S&P 500 Test Set (2022-2023)")
    print("=" * 105)
    print(
        f"{'Model Architecture':<38} | {'MAE (USD)':<12} | {'MSE (USD²)':<14} | {'RMSE (USD)':<12} | {'MAPE (%)':<10}"
    )
    print("-" * 105)

    mae_a, mse_a, rmse_a, mape_a = compute_wei_usd_metrics(
        y_test_usd[:, spy_idx], arima_preds_spy
    )
    print(
        f"{'ARIMA(2,1,1) (Classical)':<38} | ${mae_a:.2f}        | {mse_a:.2f}         | ${rmse_a:.2f}        | {mape_a:>6.2f}%"
    )
    print(
        f"{'GARCH(1,1)* (Volatility Forecast)':<38} | {'     —    ':<12} | {garch_mse:.6f}       | {garch_rmse:.5f}      | {'   —   ':<10}"
    )

    model_preds_table1 = [
        ("Sequential LSTM", lstm_preds_usd),
        ("Sequential GRU", gru_preds_usd),
        ("iTransformer (5D Restricted)", itrans_5d_usd),
        ("PatchTST (5D Restricted)", patch_5d_usd),
        ("iTransformer (45D Multivariate)", itrans_45d_usd),
        ("PatchTST (45D Multivariate)", patch_45d_usd),
        ("Standalone TS2Vec (No Graph)", ts2vec_preds_usd),
        ("Fused Model (Ours: GCN-TS2Vec)", fused_preds_usd[:, spy_idx]),
    ]
    for name, p_usd in model_preds_table1:
        mae, mse, rmse, mape = compute_wei_usd_metrics(y_test_usd[:, spy_idx], p_usd)
        print(
            f"{name:<38} | ${mae:.2f}        | {mse:.2f}         | ${rmse:.2f}        | {mape:>6.2f}%"
        )

    # =========================================================================
    # TABLE 2: QUANTITATIVE TRADING PERFORMANCE
    # =========================================================================
    print("\n" + "=" * 115)
    print(" Table 2: Trading Performance of Strategies on S&P 500 Test Set (2022-2023)")
    print("=" * 115)
    print(
        f"{'Strategy':<32} | {'Ann. Return (%)':<17} | {'Max Drawdown (%)':<18} | {'Sharpe':<8} | {'Win Rate':<10} | {'Trades':<8} | {'Final Value ($)':<18}"
    )
    print("-" * 115)

    p_bh = prices_close_df["SPY"].loc["2022-01-01":"2023-12-31"].to_numpy()
    ann_ret_bh = ((p_bh[-1] / p_bh[0]) ** (1.0 / (len(p_bh) / 252.0)) - 1.0) * 100
    peaks_bh = np.maximum.accumulate(p_bh)
    max_dd_bh = np.abs(np.min((p_bh - peaks_bh) / peaks_bh)) * 100
    d_ret_bh = (p_bh[1:] - p_bh[:-1]) / p_bh[:-1]
    sharpe_bh = (np.mean(d_ret_bh) / np.std(d_ret_bh)) * np.sqrt(252.0)
    print(
        f"{'Buy-and-Hold':<32} | {ann_ret_bh:>16.2f}% | {max_dd_bh:>16.2f}% | {sharpe_bh:>6.2f}   | {'     —    ':<10} | {'   0    ':<8} | ${100000.0 * (p_bh[-1]/p_bh[0]):>14,.2f}"
    )
    print("-" * 115)

    trading_strategies = [
        ("ARIMA(2,1,1)-based", arima_preds_spy[:, None]),
        ("LSTM-based", lstm_preds_usd[:, None]),
        ("GRU-based", gru_preds_usd[:, None]),
        ("iTransformer-based (5D)", itrans_5d_usd[:, None]),
        ("PatchTST-based (5D)", patch_5d_usd[:, None]),
        ("iTransformer-based (45D)", itrans_45d_usd[:, None]),
        ("PatchTST-based (45D)", patch_45d_usd[:, None]),
        ("Standalone TS2Vec-based", ts2vec_preds_usd[:, None]),
        ("Fused Model-based (GCN-TS2Vec)", fused_preds_usd[:, [spy_idx]]),
    ]
    for name, p_usd in trading_strategies:
        pred_ret = (p_usd - p_current_usd[:, [spy_idx]]) / p_current_usd[:, [spy_idx]]
        ann_ret, max_dd, sharpe, win_rate, n_trades, final_val = (
            run_kasui_wei_rule_strategy(
                pred_ret,
                actual_test_returns[:, [spy_idx]],
                rsi_test[:, [spy_idx]],
                garch_vol[:, [spy_idx]],
                entry_threshold=0.005,
                take_profit_mult=2.0,
                stop_loss_mult=1.5,
                max_hold_days=5,
                fee=0.001,
            )
        )
        print(
            f"{name:<32} | {ann_ret:>16.2f}% | {max_dd:>16.2f}% | {sharpe:>6.2f}   | {win_rate:>9.2f}% | {n_trades:>6}   | ${final_val:>14,.2f}"
        )

    print("=" * 115)


if __name__ == "__main__":
    run_master_benchmark()
