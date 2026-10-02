"""
dataset.py - High-Performance Quantitative Data Pipeline
Synthesizes the 5-Dimensional Feature Space, executes Reversible Instance Normalization (RevIN),
and constructs causal empirical cross-asset graphs over historical sliding lookback binders.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, List

DEFAULT_TICKERS = sorted(['SPY', 'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU'])
MASTER_CACHE = "sp500_sectors_2010_2023.csv"


def build_or_load_dataset(
    tickers: List[str] = None,
    start_date: str = "2010-01-01",
    end_date: str = "2023-12-31",
    cache_file: str = MASTER_CACHE
) -> pd.DataFrame:
    """
    Downloads and caches multi-asset OHLCV data locally.
    Guarantees aligned date indices and eliminates live API drift.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS
    else:
        tickers = sorted(list(set(tickers)))

    if not os.path.exists(cache_file):
        print(f"[*] Downloading clean OHLCV data for {tickers} from {start_date} to {end_date}...")
        raw_data = yf.download(tickers, start=start_date, end=end_date, group_by='column', threads=False)

        prices_open = raw_data['Open'][tickers].ffill().bfill()
        prices_high = raw_data['High'][tickers].ffill().bfill()
        prices_low = raw_data['Low'][tickers].ffill().bfill()
        prices_close = raw_data['Close'][tickers].ffill().bfill()
        volumes = raw_data['Volume'][tickers].replace(0, 1e-4).ffill().bfill()

        common_index = prices_close.index
        for df in [prices_open, prices_high, prices_low, volumes]:
            common_index = common_index.intersection(df.index)

        combined_df = pd.DataFrame(index=common_index)
        for t in tickers:
            combined_df[f"{t}_Open"] = prices_open.loc[common_index, t]
            combined_df[f"{t}_High"] = prices_high.loc[common_index, t]
            combined_df[f"{t}_Low"] = prices_low.loc[common_index, t]
            combined_df[f"{t}_Close"] = prices_close.loc[common_index, t]
            combined_df[f"{t}_Volume"] = volumes.loc[common_index, t]

        combined_df.to_csv(cache_file)
        print(f"[+] Static database written to: {cache_file}")

    return pd.read_csv(cache_file, index_col=0, parse_dates=True)


def process_features(
    master_df: pd.DataFrame,
    tickers: List[str] = None,
    k: int = 20
) -> Tuple[np.ndarray, pd.DatetimeIndex, pd.DataFrame]:
    """
    Constructs the 5-Dimensional Stationary Feature Space (C = 5):
      - Channel 0: First-Order Logarithmic Returns (basis for RevIN Price)
      - Channel 1: Strictly Backward-Looking Rolling Volume Z-Score (k = 20)
      - Channel 2: Normalized Garman-Klass Extreme-Value Intraday Volatility
      - Channel 3: Normalized Relative Strength Index (RSI-14 scaled to [-1, 1])
      - Channel 4: Normalized Moving Average Convergence Divergence (MACD Histogram)

    Returns:
      - feature_tensor_3d: [Days, Assets, 5]
      - valid_idx: DatetimeIndex after indicator warmup truncation (46 days)
      - prices_close_df: Aligned Close Prices [Days, Assets]
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    prices_open  = pd.DataFrame({t: master_df[f"{t}_Open"] for t in tickers}, index=master_df.index).ffill().bfill()
    prices_high  = pd.DataFrame({t: master_df[f"{t}_High"] for t in tickers}, index=master_df.index).ffill().bfill()
    prices_low   = pd.DataFrame({t: master_df[f"{t}_Low"] for t in tickers}, index=master_df.index).ffill().bfill()
    prices_close = pd.DataFrame({t: master_df[f"{t}_Close"] for t in tickers}, index=master_df.index).ffill().bfill()
    volumes      = pd.DataFrame({t: master_df[f"{t}_Volume"] for t in tickers}, index=master_df.index).replace(0, 1e-4).ffill().bfill()

    # 1. Log Returns
    log_returns = np.log(prices_close / prices_close.shift(1))

    # 2. Strictly Backward-Looking Volume Z-Score (Shifted to eliminate lookahead bias)
    shifted_v = volumes.shift(1)
    v_mean = shifted_v.rolling(window=k, min_periods=k).mean()
    v_std = shifted_v.rolling(window=k, min_periods=k).std()
    volume_zscore = (volumes - v_mean) / (v_std + 1e-8)

    # 3. Garman-Klass Extreme-Value Volatility
    ln_hl = np.log(prices_high / prices_low)
    ln_co = np.log(prices_close / prices_open)
    gk_vol = 0.5 * (ln_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (ln_co ** 2)
    gk_vol_norm = (gk_vol - gk_vol.rolling(k).mean()) / (gk_vol.rolling(k).std() + 1e-8)

    # 4. Normalized Relative Strength Index (RSI-14 centered on [-1, +1])
    delta = prices_close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi_norm = (rsi - 50.0) / 50.0

    # 5. Normalized MACD Histogram
    ema_fast = prices_close.ewm(span=12, adjust=False).mean()
    ema_slow = prices_close.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    macd_hist_norm = (macd_hist - macd_hist.rolling(k).mean()) / (macd_hist.rolling(k).std() + 1e-8)

    # Truncate warmup period (26 + k = 46 days)
    warmup_cutoff = 26 + k
    valid_idx = log_returns.index[warmup_cutoff:]

    f0 = log_returns.loc[valid_idx].to_numpy()
    f1 = volume_zscore.loc[valid_idx].to_numpy()
    f2 = gk_vol_norm.loc[valid_idx].to_numpy()
    f3 = rsi_norm.loc[valid_idx].to_numpy()
    f4 = macd_hist_norm.loc[valid_idx].to_numpy()

    feature_tensor_3d = np.stack([f0, f1, f2, f3, f4], axis=-1)
    return feature_tensor_3d, valid_idx, prices_close.loc[valid_idx]


def generate_revin_binders_and_prices(
    prices_df: pd.DataFrame,
    feature_tensor_3d: np.ndarray,
    T: int = 30,
    corr_threshold: float = 0.40
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs historical lookback binders [B, M, T, C] where Channel 0 is normalized
    via local Reversible Instance Normalization (RevIN).

    Enforces strict causal spatial graph construction over T-1 days (zero lookahead leakage).

    Returns:
      - X_revin: [Binders, Assets, T, 5]
      - A_batch: [Binders, Assets, Assets] (Symmetric dynamic graphs)
      - y_norm_target: Normalized future price targets [Binders, Assets]
      - y_usd_target: True realized future USD closing prices [Binders, Assets]
      - window_means: Localized window means mu_b [Binders, Assets]
      - window_stds: Localized window standard deviations sigma_b [Binders, Assets]
    """
    raw_prices = prices_df.to_numpy()
    D_total, M, C = feature_tensor_3d.shape
    B = D_total - T

    X_revin, A_batch = [], []
    y_norm_target, y_usd_target = [], []
    window_means, window_stds = [], []

    for b in range(B):
        p_slice = raw_prices[b : b + T, :]       # Days t=1 to t=T
        p_tomorrow = raw_prices[b + T, :]       # Day t=T+1

        # 1. Local RevIN Parameters per 30-day window
        mu = np.mean(p_slice, axis=0)
        sigma = np.std(p_slice, axis=0) + 1e-8

        # 2. RevIN Normalization
        p_norm = (p_slice - mu) / sigma
        p_norm_tomorrow = (p_tomorrow - mu) / sigma

        # 3. Channel 0 Overwrite with Normalized Price
        feat_block = feature_tensor_3d[b : b + T, :, :].copy()
        feat_block[:, :, 0] = p_norm
        binder = np.transpose(feat_block, (1, 0, 2))  # [M, T, 5]

        # 4. Strict Causal Spatial Correlation Graph across T-1 days (Eliminates Day T leakage)
        causal_returns = feature_tensor_3d[b : b + T - 1, :, 0]
        corr_matrix = np.corrcoef(causal_returns.T)
        np.fill_diagonal(corr_matrix, 0.0)
        adj_matrix = (corr_matrix > corr_threshold).astype(np.float32)

        X_revin.append(binder)
        A_batch.append(adj_matrix)
        y_norm_target.append(p_norm_tomorrow)
        y_usd_target.append(p_tomorrow)
        window_means.append(mu)
        window_stds.append(sigma)

    return (
        np.stack(X_revin, axis=0),
        np.stack(A_batch, axis=0),
        np.stack(y_norm_target, axis=0),
        np.stack(y_usd_target, axis=0),
        np.stack(window_means, axis=0),
        np.stack(window_stds, axis=0)
    )