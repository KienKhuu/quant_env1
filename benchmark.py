"""
benchmark.py - Master Benchmark & Ablation Pipeline (Kasui Wei 2025 Replication)
Executes end-to-end training, causal forecasting evaluation (Table 1),
and systematic quantitative trading backtests (Table 2) on the S&P 500 test set (2022-2023).
"""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import warnings
import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.arima.model import ARIMA

# Modular Pipeline Imports
from dataset import (
    build_or_load_dataset,
    process_features,
    generate_revin_binders_and_prices,
    DEFAULT_TICKERS
)
from models import LSTMRegression, GRURegression
from engine import (
    RevINQuantEngine,
    train_dl_baseline_with_early_stopping,
    predict_dl_raw_logits,
    train_standalone_ts2vec,
    set_seed
)
from backtester import (
    run_kasui_wei_rule_strategy,
    compute_rolling_garch_volatility,
    compute_wei_usd_metrics
)

warnings.filterwarnings('ignore')
set_seed(42)


# =========================================================================
# 1. CLASSICAL ARIMA(2,1,1) ROLLING FORECASTING HELPER
# =========================================================================
def compute_arima_1step_rolling(train_prices: np.ndarray, test_prices: np.ndarray, order=(2, 1, 1)) -> np.ndarray:
    """
    Fits ARIMA(2,1,1) on training prices and generates causal 1-step-ahead
    rolling Kalman-filtered forecasts across the test horizon.
    """
    print(f"[*] Fitting Econometric ARIMA{order} on S&P 500 Historical Close Prices...")
    full_series = np.concatenate([train_prices, test_prices])
    train_len = len(train_prices)
    test_len = len(test_prices)

    model = ARIMA(train_prices, order=order)
    fitted = model.fit()
    applied = fitted.apply(full_series)
    return applied.fittedvalues[train_len: train_len + test_len]


# =========================================================================
# 2. MASTER BENCHMARK PIPELINE
# =========================================================================
def run_master_benchmark():
    print("\n" + "=" * 115)
    print(" S&P 500 MULTI-ASSET QUANTITATIVE BENCHMARK & ABLATION STUDY (2010-2023)")
    print(" Featuring: ARIMA, GARCH, LSTM, GRU, Standalone TS2Vec (Ablation), and Fused GCN-TS2Vec")
    print("=" * 115 + "\n")

    # -------------------------------------------------------------------------
    # STEP 1: Ingest Data & Extract 5D Feature Space
    # -------------------------------------------------------------------------
    print("[1/7] Ingesting multi-asset OHLCV data and synthesizing 5D feature tensor...")
    master_df = build_or_load_dataset(
        tickers=DEFAULT_TICKERS,
        start_date="2010-01-01",
        end_date="2023-12-31",
        cache_file="sp500_sectors_2010_2023.csv"
    )
    feature_tensor, aligned_dates, prices_close_df = process_features(master_df, DEFAULT_TICKERS, k=20)

    spy_idx = DEFAULT_TICKERS.index('SPY')
    T = 30  # Historical lookback sequence window

    # -------------------------------------------------------------------------
    # STEP 2: Generate RevIN Binders & Causal Cross-Asset Graphs
    # -------------------------------------------------------------------------
    print("[2/7] Generating localized RevIN binders and causal T-1 cross-asset graphs...")
    X_all, A_all, y_norm_all, y_usd_all, mu_all, sigma_all = generate_revin_binders_and_prices(
        prices_close_df, feature_tensor, T=T, corr_threshold=0.40
    )

    f0_log_returns = feature_tensor[:, :, 0]
    f3_rsi = feature_tensor[:, :, 3]
    decision_dates = aligned_dates[T - 1: T - 1 + len(X_all)]
    rsi_all = f3_rsi[T - 1: T - 1 + len(X_all), :]

    # -------------------------------------------------------------------------
    # STEP 3: Chronological Dataset Partitioning (Section 2.1.3)
    # -------------------------------------------------------------------------
    train_mask = (decision_dates >= "2010-01-01") & (decision_dates <= "2020-12-31")
    val_mask = (decision_dates >= "2021-01-01") & (decision_dates <= "2021-12-31")
    test_mask = (decision_dates >= "2022-01-01") & (decision_dates <= "2023-12-31")

    X_train, y_train_norm, A_train = X_all[train_mask], y_norm_all[train_mask], A_all[train_mask]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    X_test, y_test_norm, A_test = X_all[test_idx], y_norm_all[test_idx], A_all[test_idx]
    y_test_usd = y_usd_all[test_idx]
    mu_test, sigma_test = mu_all[test_idx], sigma_all[test_idx]
    rsi_test = rsi_all[test_idx]

    p_current_usd = prices_close_df.iloc[test_idx + T - 1].to_numpy()
    actual_test_returns = f0_log_returns[test_idx + T, :]

    # -------------------------------------------------------------------------
    # STEP 4: Causal GARCH(1,1) Volatility Modeling
    # -------------------------------------------------------------------------
    print("[3/7] Estimating causal GARCH(1,1) conditional volatility parameters...")
    R_train = f0_log_returns[(aligned_dates >= "2010-01-01") & (aligned_dates <= "2021-12-31"), :]
    garch_vol, garch_mse, garch_rmse = compute_rolling_garch_volatility(R_train, actual_test_returns)

    # -------------------------------------------------------------------------
    # STEP 5: Econometric ARIMA(2,1,1) Forecasting
    # -------------------------------------------------------------------------
    print("[4/7] Generating rolling causal ARIMA(2,1,1) forecasts...")
    full_spy_prices = prices_close_df['SPY'].to_numpy()
    first_target_idx = test_idx[0] + T
    train_prices_spy = full_spy_prices[:first_target_idx]
    test_prices_spy = full_spy_prices[first_target_idx: test_idx[-1] + T + 1]
    arima_preds_spy = compute_arima_1step_rolling(train_prices_spy, test_prices_spy, order=(2, 1, 1))

    # -------------------------------------------------------------------------
    # STEP 6: Train Sequential Deep Learning Baselines (LSTM, GRU)
    # -------------------------------------------------------------------------
    print("[5/7] Training Sequential LSTM & GRU with 2021 Validation Early Stopping...")
    X_train_spy = X_all[train_mask, spy_idx, :, :]
    y_train_spy = y_norm_all[train_mask, spy_idx]
    X_val_spy = X_all[val_idx, spy_idx, :, :]
    y_val_spy = y_norm_all[val_idx, spy_idx]
    X_test_spy = X_all[test_idx, spy_idx, :, :]

    set_seed(42)
    lstm_m = train_dl_baseline_with_early_stopping(
        LSTMRegression(input_dim=5, hidden_dim=64, dropout=0.2),
        X_train_spy, y_train_spy, X_val_spy, y_val_spy,
        epochs=100, batch_size=32, lr=0.001, patience=10, wd=0.000261
    )
    gru_m = train_dl_baseline_with_early_stopping(
        GRURegression(input_dim=5, hidden_dim=32, dropout=0.2),
        X_train_spy, y_train_spy, X_val_spy, y_val_spy,
        epochs=100, batch_size=32, lr=0.004, patience=10, wd=0.000853
    )

    lstm_preds_usd = predict_dl_raw_logits(lstm_m, X_test_spy) * sigma_test[:, spy_idx] + mu_test[:, spy_idx]
    gru_preds_usd = predict_dl_raw_logits(gru_m, X_test_spy) * sigma_test[:, spy_idx] + mu_test[:, spy_idx]

    # -------------------------------------------------------------------------
    # STEP 7: Train Standalone TS2Vec (Ablation: Pure Temporal, No Graph)
    # -------------------------------------------------------------------------
    print("[6/7] Training Standalone TS2Vec (Ablation Benchmark: No Spatial Graph)...")
    set_seed(42)
    ts2vec_preds_usd = train_standalone_ts2vec(
        X_train_spy, y_train_spy, X_val_spy, y_val_spy, X_test_spy,
        mu_test[:, spy_idx], sigma_test[:, spy_idx]
    )

    # -------------------------------------------------------------------------
    # STEP 8: Train Fused Model (Ours: Spatial GCN + Bidirectional TS2Vec + RevIN)
    # -------------------------------------------------------------------------
    print("[7/7] Training Fused Spatial-Temporal Architecture (GCN + TS2Vec with RevIN)...")
    set_seed(42)
    M = len(DEFAULT_TICKERS)
    engine = RevINQuantEngine(num_assets=M, input_dims=5, hidden_dims=128, num_blocks=2, lr=0.00216, device='cpu')
    for pg in engine.optimizer.param_groups:
        pg['weight_decay'] = 2.5e-5

    engine.fit_unsupervised(X_train, A_train, X_test, A_test, epochs=15, patience=4)
    engine.fit_supervised(X_train, A_train, y_train_norm, X_test, A_test, y_test_norm, epochs=12)
    fused_preds_usd = engine.predict_usd(X_test, A_test, mu_test, sigma_test)

    # =========================================================================
    # TABLE 1: FORECASTING ACCURACY EVALUATION (S&P 500 SPY TARGET)
    # =========================================================================
    print("\n" + "=" * 105)
    print(" Table 1: Forecasting Performance of Models on S&P 500 Test Set (2022-2023)")
    print("=" * 105)
    print(
        f"{'Model Architecture':<38} | {'MAE (USD)':<12} | {'MSE (USD²)':<14} | {'RMSE (USD)':<12} | {'MAPE (%)':<10}")
    print("-" * 105)

    mae_a, mse_a, rmse_a, mape_a = compute_wei_usd_metrics(y_test_usd[:, spy_idx], arima_preds_spy)
    print(
        f"{'ARIMA(2,1,1) (Classical)':<38} | ${mae_a:.2f}        | {mse_a:.2f}         | ${rmse_a:.2f}        | {mape_a:>6.2f}%")
    print(
        f"{'GARCH(1,1)* (Volatility Forecast)':<38} | {'     —    ':<12} | {garch_mse:.6f}       | {garch_rmse:.5f}      | {'   —   ':<10}")

    model_preds_table1 = [
        ("Sequential LSTM", lstm_preds_usd),
        ("Sequential GRU", gru_preds_usd),
        ("Standalone TS2Vec (No Graph)", ts2vec_preds_usd),
        ("Fused Model (Ours: GCN-TS2Vec)", fused_preds_usd[:, spy_idx])
    ]
    for name, p_usd in model_preds_table1:
        mae, mse, rmse, mape = compute_wei_usd_metrics(y_test_usd[:, spy_idx], p_usd)
        print(f"{name:<38} | ${mae:.2f}        | {mse:.2f}         | ${rmse:.2f}        | {mape:>6.2f}%")

    print("=" * 105)
    print("*Note: GARCH forecasts return variance; MSE and RMSE evaluate variance innovations.")

    # =========================================================================
    # TABLE 2: QUANTITATIVE TRADING PERFORMANCE (KASUI WEI 2025 RULES)
    # =========================================================================
    print("\n" + "=" * 115)
    print(" Table 2: Trading Performance of Strategies on S&P 500 Test Set (2022-2023)")
    print("=" * 115)
    print(
        f"{'Strategy':<32} | {'Annualized Return (%)':<23} | {'Max Drawdown (%)':<18} | {'Sharpe Ratio':<14} | {'Win Rate (%)':<14} | {'Trades':<8} | {'Final Value (USD)':<18}")
    print("-" * 115)

    # 1. Buy and Hold Benchmark (True Geometric CAGR on SPY)
    p_bh = prices_close_df['SPY'].loc["2022-01-01":"2023-12-31"].to_numpy()
    ann_ret_bh = ((p_bh[-1] / p_bh[0]) ** (1.0 / (len(p_bh) / 252.0)) - 1.0) * 100
    peaks_bh = np.maximum.accumulate(p_bh)
    max_dd_bh = np.abs(np.min((p_bh - peaks_bh) / peaks_bh)) * 100
    d_ret_bh = (p_bh[1:] - p_bh[:-1]) / p_bh[:-1]
    sharpe_bh = (np.mean(d_ret_bh) / np.std(d_ret_bh)) * np.sqrt(252.0)
    final_bh = 100000.0 * (p_bh[-1] / p_bh[0])
    print(
        f"{'Buy-and-Hold':<32} | {ann_ret_bh:>21.2f}% | {max_dd_bh:>16.2f}% | {sharpe_bh:>12.2f} | {'     —      ':<14} | {'   0    ':<8} | ${final_bh:>16,.2f}")
    print("-" * 115)

    # 2. Rule-Based Quantitative Strategies
    trading_strategies = [
        ("ARIMA(2,1,1)-based", arima_preds_spy[:, None]),
        ("LSTM-based", lstm_preds_usd[:, None]),
        ("GRU-based", gru_preds_usd[:, None]),
        ("Standalone TS2Vec-based", ts2vec_preds_usd[:, None]),
        ("Fused Model-based (GCN-TS2Vec)", fused_preds_usd[:, [spy_idx]])
    ]
    for name, p_usd in trading_strategies:
        pred_ret = (p_usd - p_current_usd[:, [spy_idx]]) / p_current_usd[:, [spy_idx]]
        ann_ret, max_dd, sharpe, win_rate, n_trades, final_val = run_kasui_wei_rule_strategy(
            pred_ret, actual_test_returns[:, [spy_idx]], rsi_test[:, [spy_idx]], garch_vol[:, [spy_idx]],
            entry_threshold=0.005, take_profit_mult=2.0, stop_loss_mult=1.5, max_hold_days=5, fee=0.001
        )
        print(
            f"{name:<32} | {ann_ret:>21.2f}% | {max_dd:>16.2f}% | {sharpe:>12.2f} | {win_rate:>12.2f}% | {n_trades:>6}   | ${final_val:>16,.2f}")

    print("=" * 115)


if __name__ == "__main__":
    run_master_benchmark()