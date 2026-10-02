"""
backtester.py - Institutional Quantitative Backtesting & Econometric Evaluation
Features:
  1. Kasui Wei (2025) Section 2.3 Rule-Based Quantitative Strategy Execution
  2. Causal GARCH(1,1) Conditional Volatility Estimator & Recursive Filter
  3. Institutional Compounded Buy-and-Hold Benchmark
  4. Physical Dollar Forecasting Error Metrics (MAE, MSE, RMSE, MAPE)
"""

import numpy as np
import pandas as pd
from typing import Tuple
from arch import arch_model


# =========================================================================
# 1. KASUI WEI (2025) RULE-BASED QUANTITATIVE STRATEGY EXECUTION
# =========================================================================
def run_kasui_wei_rule_strategy(
    predicted_return: np.ndarray,
    actual_log_returns: np.ndarray,
    rsi_matrix: np.ndarray,
    garch_vol_matrix: np.ndarray,
    entry_threshold: float = 0.005,  # 0.5% return deadband (Section 2.3.1)
    take_profit_mult: float = 2.0,   # 2.0x GARCH volatility take-profit (Section 2.3.2)
    stop_loss_mult: float = 1.5,     # 1.5x GARCH volatility stop-loss (Section 2.3.2)
    max_hold_days: int = 5,          # 5-day time exit (Section 2.3.2)
    initial_capital: float = 100000.0,
    fee: float = 0.001               # 10 bps institutional fee
) -> Tuple[float, float, float, float, int, float]:
    """
    Exact replication of Kasui Wei (2025) Section 2.3 Quantitative Strategy.
    Enforces clean causal alignment: signals evaluated at Day t close trade Day t return.

    Returns:
      (Annualized Return %, Max Drawdown %, Sharpe Ratio, Win Rate %, Trade Count, Final Capital $)
    """
    B_test, M = predicted_return.shape
    simple_returns = np.exp(actual_log_returns) - 1.0
    allocation_weight = 1.0 / M
    current_capital = initial_capital
    portfolio_values = [initial_capital]

    positions = np.zeros(M, dtype=np.float32)       # -1.0 (Short), 0.0 (Flat), +1.0 (Long)
    days_held = np.zeros(M, dtype=np.int32)
    accumulated_ret = np.zeros(M, dtype=np.float32)

    total_closed_trades = 0
    winning_trades = 0

    for t in range(B_test):
        prev_positions = positions.copy()

        # Step 1: Check Exits for Open Positions (Section 2.3.2)
        for m in range(M):
            if positions[m] != 0.0:
                days_held[m] += 1
                accumulated_ret[m] += positions[m] * simple_returns[t, m]

                vol_t = max(garch_vol_matrix[t, m], 0.005)
                tp_target = take_profit_mult * vol_t
                sl_target = stop_loss_mult * vol_t

                # Take-Profit, Stop-Loss, or 5-Day Maximum Time Exit
                if (accumulated_ret[m] >= tp_target) or (accumulated_ret[m] <= -sl_target) or (days_held[m] >= max_hold_days):
                    total_closed_trades += 1
                    if accumulated_ret[m] > 0:
                        winning_trades += 1
                    positions[m] = 0.0
                    days_held[m] = 0
                    accumulated_ret[m] = 0.0

        # Step 2: Check Entry Signals for Flat Assets (Section 2.3.1)
        for m in range(M):
            if positions[m] == 0.0:
                pred = predicted_return[t, m]
                rsi = rsi_matrix[t, m]

                # Long: Predicted Return > +0.5% and RSI < 70 (Normalized RSI < 0.40)
                if pred > entry_threshold and rsi < 0.40:
                    positions[m] = 1.0
                    days_held[m] = 0
                    accumulated_ret[m] = 0.0

                # Short: Predicted Return < -0.5% and RSI > 30 (Normalized RSI > -0.40)
                elif pred < -entry_threshold and rsi > -0.40:
                    positions[m] = -1.0
                    days_held[m] = 0
                    accumulated_ret[m] = 0.0

        # Step 3: Transaction Costs and Geometric Compounding
        position_change = np.abs(positions - prev_positions)
        trade_costs = np.sum(current_capital * allocation_weight * position_change * fee)
        current_capital -= trade_costs
        current_capital *= (1.0 + np.sum(allocation_weight * positions * simple_returns[t]))
        portfolio_values.append(current_capital)

    portfolio_values = np.array(portfolio_values)
    final_value = portfolio_values[-1]
    daily_returns = (portfolio_values[1:] - portfolio_values[:-1]) / (portfolio_values[:-1] + 1e-8)

    # Performance Metrics
    years = B_test / 252.0
    ann_ret = (final_value / initial_capital) ** (1.0 / years) - 1.0
    peaks = np.maximum.accumulate(portfolio_values)
    max_dd = np.abs(np.min((portfolio_values - peaks) / (peaks + 1e-8)))
    std_daily = np.std(daily_returns)
    sharpe = (np.mean(daily_returns) / std_daily) * np.sqrt(252.0) if std_daily > 1e-8 else 0.0

    trade_win_rate = (winning_trades / total_closed_trades * 100.0) if total_closed_trades > 0 else 0.0
    return ann_ret * 100, max_dd * 100, sharpe, trade_win_rate, total_closed_trades, final_value


# =========================================================================
# 2. CAUSAL GARCH(1,1) CONDITIONAL VOLATILITY ESTIMATOR
# =========================================================================
def compute_rolling_garch_volatility(
    R_train: np.ndarray,
    actual_test_returns: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Fits GARCH(1,1) via QMLE on historical percentage training returns.
    Executes causal 1-step-ahead forward variance recursion across test returns.

    Returns:
      (garch_vol_matrix, garch_mse, garch_rmse)
    """
    B_val, M = actual_test_returns.shape
    garch_vol_matrix = np.zeros((B_val, M), dtype=np.float32)
    actual_test_variance = []
    predicted_garch_variance = []

    for m in range(M):
        asset_train_ret = R_train[:, m] * 100.0  # Percentage scale for optimizer convergence
        asset_test_ret  = actual_test_returns[:, m] * 100.0

        am = arch_model(asset_train_ret, vol='Garch', p=1, q=1, dist='Normal', rescale=False)
        res = am.fit(disp='off', show_warning=False)

        omega, alpha, beta = res.params['omega'], res.params['alpha[1]'], res.params['beta[1]']
        last_var = res.conditional_volatility[-1] ** 2
        last_ret = asset_train_ret[-1]

        for t in range(B_val):
            # Forward recursive conditional variance
            forecast_var = omega + alpha * (last_ret ** 2) + beta * last_var
            forecast_vol = np.sqrt(max(forecast_var, 1e-6)) / 100.0  # Decimal volatility
            garch_vol_matrix[t, m] = forecast_vol

            realized_var = (actual_test_returns[t, m]) ** 2
            actual_test_variance.append(realized_var)
            predicted_garch_variance.append(forecast_vol ** 2)

            # Advance state using realized return
            last_var = forecast_var
            last_ret = asset_test_ret[t]

    act_v = np.array(actual_test_variance)
    pred_v = np.array(predicted_garch_variance)
    garch_mse = np.mean((act_v - pred_v) ** 2)
    garch_rmse = np.sqrt(garch_mse)

    return garch_vol_matrix, garch_mse, garch_rmse


# =========================================================================
# 3. INSTITUTIONAL BUY-AND-HOLD BENCHMARK
# =========================================================================
def run_true_buy_and_hold_benchmark(
    prices_df: pd.DataFrame,
    start_pos: int,
    B_val: int,
    initial_capital: float = 100000.0,
    fee: float = 0.001
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Computes an institutional Buy-and-Hold benchmark tracking fixed share weights over time.
    """
    window_prices = prices_df.iloc[start_pos : start_pos + B_val].to_numpy()
    B_test, M = window_prices.shape

    capital_per_asset = (initial_capital * (1.0 - fee)) / M
    shares_held = capital_per_asset / window_prices[0, :]

    portfolio_values = []
    for t in range(B_test):
        daily_value = np.sum(shares_held * window_prices[t, :])
        portfolio_values.append(daily_value)

    portfolio_values = np.array(portfolio_values)
    final_value = portfolio_values[-1]
    daily_returns = (portfolio_values[1:] - portfolio_values[:-1]) / (portfolio_values[:-1] + 1e-8)

    years = B_test / 252.0
    ann_ret = (final_value / initial_capital) ** (1.0 / years) - 1.0
    peaks = np.maximum.accumulate(portfolio_values)
    max_dd = np.abs(np.min((portfolio_values - peaks) / (peaks + 1e-8)))

    std_daily = np.std(daily_returns)
    sharpe = (np.mean(daily_returns) / std_daily) * np.sqrt(252.0) if std_daily > 1e-8 else 0.0
    win_rate = np.mean(daily_returns > 0.0)

    downside = daily_returns[daily_returns < 0.0]
    gross_gains = np.sum(daily_returns[daily_returns > 0.0])
    sortino = (np.mean(daily_returns) / np.std(downside)) * np.sqrt(252.0) if len(downside) > 1 and np.std(downside) > 1e-8 else 0.0
    pf = gross_gains / (np.abs(np.sum(downside)) + 1e-8)

    return ann_ret * 100, max_dd * 100, sharpe, sortino, pf, win_rate * 100, final_value


# =========================================================================
# 4. KASUI WEI FORECASTING ACCURACY METRICS (IN USD & %)
# =========================================================================
def compute_wei_usd_metrics(y_true_usd: np.ndarray, y_pred_usd: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Computes exact Table 1 forecasting accuracy metrics in physical dollar units.
    """
    flat_true = y_true_usd.reshape(-1)
    flat_pred = y_pred_usd.reshape(-1)

    mae_usd = np.mean(np.abs(flat_true - flat_pred))
    mse_usd = np.mean((flat_true - flat_pred) ** 2)
    rmse_usd = np.sqrt(mse_usd)
    mape_pct = np.mean(np.abs((flat_true - flat_pred) / flat_true)) * 100.0

    return mae_usd, mse_usd, rmse_usd, mape_pct