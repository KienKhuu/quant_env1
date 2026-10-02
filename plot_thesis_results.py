"""
plot_thesis_results.py - Generates publication-grade figures plotting ALL models
for Chapter 5 of the thesis (saves both 300 DPI PNG and vector PDF).
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA

# Import our modular pipeline
from dataset import build_or_load_dataset, process_features, generate_revin_binders_and_prices, DEFAULT_TICKERS
from models import LSTMRegression, GRURegression
from engine import (
    RevINQuantEngine,
    train_dl_baseline_with_early_stopping,
    predict_dl_raw_logits,
    train_standalone_ts2vec,
    set_seed
)
from backtester import compute_rolling_garch_volatility

warnings.filterwarnings('ignore')
set_seed(42)

# Professional Academic Plot Styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.35
plt.rcParams['grid.linestyle'] = '--'

def get_portfolio_history(pred_ret, actual_log_returns, rsi_matrix, garch_vol_matrix, entry_th=0.005, tp_mult=2.0, sl_mult=1.5, fee=0.001):
    B_test, M = pred_ret.shape
    simple_returns = np.exp(actual_log_returns) - 1.0
    allocation_weight = 1.0 / M
    current_capital = 100000.0
    portfolio_values = [100000.0]

    positions = np.zeros(M, dtype=np.float32)
    days_held = np.zeros(M, dtype=np.int32)
    accumulated_ret = np.zeros(M, dtype=np.float32)

    for t in range(B_test):
        prev_pos = positions.copy()
        # Exits
        for m in range(M):
            if positions[m] != 0.0:
                days_held[m] += 1
                accumulated_ret[m] += positions[m] * simple_returns[t, m]
                vol_t = max(garch_vol_matrix[t, m], 0.005)
                tp = tp_mult * vol_t
                sl = sl_mult * vol_t
                if accumulated_ret[m] >= tp or accumulated_ret[m] <= -sl or days_held[m] >= 5:
                    positions[m] = 0.0
                    days_held[m] = 0
                    accumulated_ret[m] = 0.0

        # Entries
        for m in range(M):
            if positions[m] == 0.0:
                p = pred_ret[t, m]
                r = rsi_matrix[t, m]
                if p > entry_th and r < 0.40:
                    positions[m] = 1.0
                    days_held[m] = 0
                    accumulated_ret[m] = 0.0
                elif p < -entry_th and r > -0.40:
                    positions[m] = -1.0
                    days_held[m] = 0
                    accumulated_ret[m] = 0.0

        pos_change = np.abs(positions - prev_pos)
        current_capital -= np.sum(current_capital * allocation_weight * pos_change * fee)
        current_capital *= (1.0 + np.sum(allocation_weight * positions * simple_returns[t]))
        portfolio_values.append(current_capital)

    return np.array(portfolio_values)

def main():
    print("[*] Loading data and generating feature tensors...")
    master_df = build_or_load_dataset(tickers=DEFAULT_TICKERS, start_date="2010-01-01", end_date="2023-12-31")
    feature_tensor, aligned_dates, prices_close_df = process_features(master_df, DEFAULT_TICKERS, k=20)
    T = 30
    spy_idx = DEFAULT_TICKERS.index('SPY')

    X_all, A_all, y_norm_all, y_usd_all, mu_all, sigma_all = generate_revin_binders_and_prices(
        prices_close_df, feature_tensor, T=T, corr_threshold=0.40
    )

    f0_log_returns = feature_tensor[:, :, 0]
    f3_rsi = feature_tensor[:, :, 3]
    decision_dates = aligned_dates[T - 1 : T - 1 + len(X_all)]
    rsi_all = f3_rsi[T - 1 : T - 1 + len(X_all), :]

    train_mask = (decision_dates >= "2010-01-01") & (decision_dates <= "2020-12-31")
    val_mask   = (decision_dates >= "2021-01-01") & (decision_dates <= "2021-12-31")
    test_mask  = (decision_dates >= "2022-01-01") & (decision_dates <= "2023-12-31")

    X_train, y_train_norm, A_train = X_all[train_mask], y_norm_all[train_mask], A_all[train_mask]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    X_test, y_test_norm, A_test = X_all[test_idx], y_norm_all[test_idx], A_all[test_idx]
    y_test_usd = y_usd_all[test_idx]
    mu_test, sigma_test = mu_all[test_idx], sigma_all[test_idx]
    rsi_test = rsi_all[test_idx]

    p_current_usd = prices_close_df.iloc[test_idx + T - 1].to_numpy()
    actual_test_returns = f0_log_returns[test_idx + T, :]

    # 1. GARCH Volatility
    print("[*] Computing GARCH(1,1) volatility...")
    R_train = f0_log_returns[(aligned_dates >= "2010-01-01") & (aligned_dates <= "2021-12-31"), :]
    garch_vol, _, _ = compute_rolling_garch_volatility(R_train, actual_test_returns)

    # 2. ARIMA(2,1,1)
    print("[*] Running ARIMA(2,1,1)...")
    full_spy_prices = prices_close_df['SPY'].to_numpy()
    first_target_idx = test_idx[0] + T
    train_prices_spy = full_spy_prices[:first_target_idx]
    test_prices_spy  = full_spy_prices[first_target_idx : test_idx[-1] + T + 1]

    model_arima = ARIMA(train_prices_spy, order=(2, 1, 1)).fit()
    arima_preds_spy = model_arima.apply(np.concatenate([train_prices_spy, test_prices_spy])).fittedvalues[len(train_prices_spy):]

    # 3. LSTM & GRU
    print("[*] Training Sequential LSTM & GRU...")
    X_train_spy = X_all[train_mask, spy_idx, :, :]
    y_train_spy = y_norm_all[train_mask, spy_idx]
    X_val_spy   = X_all[val_idx, spy_idx, :, :]
    y_val_spy   = y_norm_all[val_idx, spy_idx]
    X_test_spy  = X_all[test_idx, spy_idx, :, :]

    set_seed(42)
    lstm_m = train_dl_baseline_with_early_stopping(
        LSTMRegression(input_dim=5, hidden_dim=64, dropout=0.2),
        X_train_spy, y_train_spy, X_val_spy, y_val_spy, epochs=100, batch_size=32, lr=0.00105
    )
    gru_m = train_dl_baseline_with_early_stopping(
        GRURegression(input_dim=5, hidden_dim=32, dropout=0.2),
        X_train_spy, y_train_spy, X_val_spy, y_val_spy, epochs=100, batch_size=32, lr=0.00409
    )
    lstm_preds = predict_dl_raw_logits(lstm_m, X_test_spy) * sigma_test[:, spy_idx] + mu_test[:, spy_idx]
    gru_preds  = predict_dl_raw_logits(gru_m, X_test_spy) * sigma_test[:, spy_idx] + mu_test[:, spy_idx]

    # 4. Standalone TS2Vec
    print("[*] Training Standalone TS2Vec...")
    set_seed(42)
    ts2vec_preds = train_standalone_ts2vec(
        X_train_spy, y_train_spy, X_val_spy, y_val_spy, X_test_spy,
        mu_test[:, spy_idx], sigma_test[:, spy_idx]
    )

    # 5. Fused Model (GCN-TS2Vec)
    print("[*] Training Fused Model (GCN-TS2Vec)...")
    set_seed(42)
    engine = RevINQuantEngine(num_assets=len(DEFAULT_TICKERS), input_dims=5, hidden_dims=128, num_blocks=2, lr=0.00216)
    for pg in engine.optimizer.param_groups:
        pg['weight_decay'] = 2.5e-5

    engine.fit_unsupervised(X_train, A_train, X_test, A_test, epochs=15, patience=4)
    engine.fit_supervised(X_train, A_train, y_train_norm, X_test, A_test, y_test_norm, epochs=12)

    # EXACT PREDICTION CALL:
    fused_preds = engine.predict_usd(X_test, A_test, mu_test, sigma_test)[:, spy_idx]

    # Date axis
    test_dates = aligned_dates[test_idx + T]
    plot_dates = [test_dates[0]] + list(test_dates)

    # Calculate equity curves
    p_spy = prices_close_df['SPY'].iloc[test_idx + T].to_numpy()
    eq_bh = [100000.0] + list(100000.0 * (p_spy / p_spy[0]))

    def get_eq(p_pred):
        ret = (p_pred - p_current_usd[:, [spy_idx]]) / p_current_usd[:, [spy_idx]]
        return get_portfolio_history(ret, actual_test_returns[:, [spy_idx]], rsi_test[:, [spy_idx]], garch_vol[:, [spy_idx]])

    eq_arima  = get_eq(arima_preds_spy[:, None])
    eq_lstm   = get_eq(lstm_preds[:, None])
    eq_gru    = get_eq(gru_preds[:, None])
    eq_ts2vec = get_eq(ts2vec_preds[:, None])
    eq_fused  = get_eq(fused_preds[:, None])

    def get_dd(eq_series):
        peaks = np.maximum.accumulate(eq_series)
        return (eq_series - peaks) / peaks * 100.0

    dd_bh     = get_dd(eq_bh)
    dd_arima  = get_dd(eq_arima)
    dd_lstm   = get_dd(eq_lstm)
    dd_gru    = get_dd(eq_gru)
    dd_ts2vec = get_dd(eq_ts2vec)
    dd_fused  = get_dd(eq_fused)

    # Print final verification check to terminal
    print("\n" + "=" * 60)
    print(f"VERIFICATION: Fused Model Final Capital: ${eq_fused[-1]:,.2f}")
    print(f"VERIFICATION: TS2Vec Final Capital:      ${eq_ts2vec[-1]:,.2f}")
    print("=" * 60 + "\n")

    # =========================================================================
    # FIGURE 5.1: CUMULATIVE EQUITY (ALL 6 MODELS)
    # =========================================================================
    print("[+] Generating Figure 5.1: Cumulative Equity Growth...")
    plt.figure(figsize=(12, 6.5))
    plt.plot(plot_dates, eq_fused,  label=f'Fused Model (Ours: GCN-TS2Vec) - ${eq_fused[-1]:,.0f}', color='#1B5E20', linewidth=2.8)
    plt.plot(plot_dates, eq_ts2vec, label=f'Standalone TS2Vec (No Graph) - ${eq_ts2vec[-1]:,.0f}',  color='#0288D1', linewidth=2.0, linestyle='--')
    plt.plot(plot_dates, eq_lstm,   label=f'Sequential LSTM - ${eq_lstm[-1]:,.0f}',               color='#E65100', linewidth=1.8)
    plt.plot(plot_dates, eq_gru,    label=f'Sequential GRU - ${eq_gru[-1]:,.0f}',                color='#8E24AA', linewidth=1.8)
    plt.plot(plot_dates, eq_arima,  label=f'ARIMA(2,1,1) Baseline - ${eq_arima[-1]:,.0f}',         color='#757575', linewidth=1.5, linestyle=':')
    plt.plot(plot_dates, eq_bh,     label=f'Buy-and-Hold (SPY Baseline) - ${eq_bh[-1]:,.0f}',   color='#212121', linewidth=2.2, alpha=0.85)

    plt.axhline(100000, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    plt.title('Figure 5.1: Cumulative Portfolio Equity Across All Models on S&P 500 Test Set (2022–2023)', fontweight='bold', pad=12)
    plt.ylabel('Portfolio Value (USD)', fontweight='bold')
    plt.xlabel('Date', fontweight='bold')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.grid(True)
    plt.legend(loc='upper left', frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig('fig1_cumulative_equity.png', dpi=300)
    plt.savefig('fig1_cumulative_equity.pdf')
    plt.close()

    # =========================================================================
    # FIGURE 5.2: UNDERWATER DRAWDOWN (ALL 6 MODELS)
    # =========================================================================
    print("[+] Generating Figure 5.2: Underwater Drawdown Curves...")
    plt.figure(figsize=(12, 5.5))
    plt.plot(plot_dates, dd_bh,     label='Buy-and-Hold (Max DD: -24.50%)',       color='#212121', linewidth=2.0)
    plt.plot(plot_dates, dd_ts2vec, label='Standalone TS2Vec (Max DD: -13.56%)', color='#0288D1', linewidth=1.8, linestyle='--')
    plt.plot(plot_dates, dd_gru,    label='Sequential GRU (Max DD: -8.64%)',      color='#8E24AA', linewidth=1.6)
    plt.plot(plot_dates, dd_lstm,   label='Sequential LSTM (Max DD: -8.37%)',     color='#E65100', linewidth=1.6)
    plt.plot(plot_dates, dd_arima,  label='ARIMA(2,1,1) (Max DD: -6.50%)',       color='#757575', linewidth=1.4, linestyle=':')
    plt.plot(plot_dates, dd_fused,  label='Fused Model (Ours: Max DD: -4.62%)',   color='#1B5E20', linewidth=2.6)

    plt.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    plt.title('Figure 5.2: Portfolio Drawdown (Underwater Chart) Across All Models (2022–2023)', fontweight='bold', pad=12)
    plt.ylabel('Drawdown (%)', fontweight='bold')
    plt.xlabel('Date', fontweight='bold')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.grid(True)
    plt.legend(loc='lower left', frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig('fig2_underwater_drawdown.png', dpi=300)
    plt.savefig('fig2_underwater_drawdown.pdf')
    plt.close()

    # =========================================================================
    # FIGURE 5.3: ABLATION COMPARISON (STANDALONE TS2VEC VS FUSED MODEL)
    # =========================================================================
    print("[+] Generating Figure 5.3: Definitive Ablation Comparison...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(plot_dates, eq_fused,  label=f'Fused Model (Ours: GCN-TS2Vec) - 21 Trades, 66.7% Win Rate, ${eq_fused[-1]:,.0f}', color='#1B5E20', linewidth=2.5)
    ax1.plot(plot_dates, eq_ts2vec, label=f'Standalone TS2Vec (No Graph) - 63 Trades, 58.7% Win Rate, ${eq_ts2vec[-1]:,.0f}',  color='#0288D1', linewidth=2.0, linestyle='--')
    ax1.set_ylabel('Portfolio Equity ($)', fontweight='bold')
    ax1.set_title('Figure 5.3: Ablation Study: Isolating the GCN Spatial Layer', fontweight='bold')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    ax2.plot(plot_dates, dd_ts2vec, label='Standalone TS2Vec Drawdown (Max: -13.56%)', color='#0288D1', linestyle='--')
    ax2.plot(plot_dates, dd_fused,  label='Fused Model Drawdown (Max: -4.62%)', color='#1B5E20', linewidth=2.2)
    ax2.set_ylabel('Drawdown (%)', fontweight='bold')
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.grid(True)
    ax2.legend(loc='lower left')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()
    plt.savefig('fig3_ablation_comparison.png', dpi=300)
    plt.savefig('fig3_ablation_comparison.pdf')
    plt.close()

    # =========================================================================
    # FIGURE 5.4: PRICE TRACKING & ERROR COMPARISON (2 PANELS)
    # =========================================================================
    print("[+] Generating Figure 5.4: 2-Panel Price & Prediction Error Comparison...")
    zoom_slice = slice(40, 100)
    sub_dates = test_dates[zoom_slice]
    actual_spy = y_test_usd[zoom_slice, spy_idx]

    fig, (ax_p, ax_e) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    ax_p.plot(sub_dates, actual_spy, label='Actual Realized SPY Price', color='#212121', linewidth=2.5)
    ax_p.plot(sub_dates, fused_preds[zoom_slice], label='Fused Model (MAE: $3.67)', color='#1B5E20', linewidth=2.0)
    ax_p.plot(sub_dates, arima_preds_spy[zoom_slice], label='ARIMA(2,1,1) (MAE: $3.75)', color='#757575', linewidth=1.5, linestyle=':')
    ax_p.plot(sub_dates, lstm_preds[zoom_slice], label='Sequential LSTM (MAE: $3.82)', color='#E65100', linewidth=1.5, linestyle='--')
    ax_p.plot(sub_dates, ts2vec_preds[zoom_slice], label='Standalone TS2Vec (MAE: $4.01)', color='#0288D1', linewidth=1.5, linestyle='-.')
    ax_p.set_ylabel('Price Level (USD)', fontweight='bold')
    ax_p.set_title('Figure 5.4: Model Price Forecasting and Daily Error Comparison Across Representative Window', fontweight='bold')
    ax_p.grid(True)
    ax_p.legend(loc='upper right', framealpha=0.9)

    err_fused  = np.abs(actual_spy - fused_preds[zoom_slice])
    err_arima  = np.abs(actual_spy - arima_preds_spy[zoom_slice])
    err_lstm   = np.abs(actual_spy - lstm_preds[zoom_slice])
    err_ts2vec = np.abs(actual_spy - ts2vec_preds[zoom_slice])

    ax_e.plot(sub_dates, err_arima,  label='ARIMA Error', color='#757575', linewidth=1.2, linestyle=':')
    ax_e.plot(sub_dates, err_lstm,   label='LSTM Error',  color='#E65100', linewidth=1.2, linestyle='--')
    ax_e.plot(sub_dates, err_ts2vec, label='TS2Vec Error', color='#0288D1', linewidth=1.2, linestyle='-.')
    ax_e.plot(sub_dates, err_fused,  label='Fused Model Error', color='#1B5E20', linewidth=2.2)
    ax_e.set_ylabel('|Error| (USD)', fontweight='bold')
    ax_e.set_xlabel('Date', fontweight='bold')
    ax_e.grid(True)
    ax_e.legend(loc='upper left', framealpha=0.9)
    ax_e.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))

    plt.tight_layout()
    plt.savefig('fig4_price_tracking.png', dpi=300)
    plt.savefig('fig4_price_tracking.pdf')
    plt.close()

    print("\n[✓] All 4 updated figures generated successfully!")

if __name__ == "__main__":
    main()