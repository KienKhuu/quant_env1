
# Learning Contrastive Representations for Stock Price Prediction
### A Spatial-Temporal Framework with Graph Convolutions and Reversible Normalization
> **Capstone Project / Graduation Thesis (Đồ án tốt nghiệp / Luận văn tốt nghiệp)**  
> **Topic Code:** `HK253-DATN-031`  
> **Department:** Foundations of Computer Science, Faculty of Computer Science and Engineering  
> **Institution:** Ho Chi Minh City University of Technology (HCMUT – VNU-HCM)  
> **Academic Year:** 2025 – 2026

---

## 👥 Authors & Academic Supervision

* **Supervisor:** **TS. Trương Vĩnh Lân** (Faculty of Computer Science and Engineering)
* **Student Research Team:**
  1. **Khưu Vĩnh Kiên** — Student ID: `2211720`
  2. **Hoàng Minh Quân** — Student ID: `2212787`
  3. **Đinh Gia Kiệt** — Student ID: `2252399`

---

## 📌 Executive Summary

Predicting financial asset prices with machine learning is notoriously challenging due to **extreme market noise (low SNR $< 5\%$)**, **decadal price drift (unit-root non-stationarity)**, and the **MSE Paradox** (where models minimize quadratic error by simply predicting no change, failing in live trading).

This repository contains the official implementation of **GCN-TS2Vec with Reversible Instance Normalization (RevIN)**. Our framework solves these core problems by combining:
1. **Local Scale Invariance (RevIN):** Normalizes each 30-day window independently, eliminating 14 years of secular price drift while preserving candlestick shapes and enabling exact dollar price reconstruction.
2. **Horizontal Spatial Communication (GCN):** Employs spectral Graph Convolutional Networks (Kipf & Welling, 2017) across the S&P 500 and 8 major sector ETFs, acting as an automated **false-breakout filter**.
3. **Multi-Scale Temporal Convolution (TS2Vec):** Leverages bidirectional dilated convolutions with GELU activations to scan multi-day trends without recurrent sequence lag.
4. **Self-Supervised Decoupled Pre-Training:** Pre-trains the neural backbone on masked context views using a tri-channel InfoNCE loss with learnable homoscedastic uncertainty, insulating the model from daily white-noise memorization.

---

## 🏆 Key Empirical Benchmark Results (S&P 500: 2022–2023)

Evaluated across the strictly out-of-sample 2022–2023 test set ($N = 502$ trading days) covering the 2022 bear market crash ($-24.5\%$ drawdown) and the 2023 bull recovery:

### Table 1: Statistical Point-Forecasting Accuracy
| Model Architecture | MAE (USD) ↓ | MSE (USD²) ↓ | RMSE (USD) ↓ | MAPE (%) ↓ |
| :--- | :---: | :---: | :---: | :---: |
| **ARIMA(2,1,1)** *(Linear Persistence Baseline)* | \$3.75 | 23.63 | \$4.86 | 0.96\% |
| **Sequential LSTM** *(Recurrent Baseline)* | \$3.82 | 24.09 | \$4.91 | 0.98\% |
| **Sequential GRU** *(Gated Recurrent Baseline)* | \$3.83 | 24.60 | \$4.96 | 0.98\% |
| **Standalone TS2Vec** *(Pure Temporal - No Graph)* | \$4.01 | 27.08 | \$5.20 | 1.03\% |
| **Fused Model (Ours: GCN-TS2Vec)** | **\$3.67** | **23.05** | **\$4.80** | **0.94\%** |

*Our Fused Model is the only architecture to break below the analytical random-walk noise floor ($\$3.75$).*

### Table 2: Quantitative Trading Performance (Kasui Wei Rules)
| Strategy | Annualized Return ↑ | Max Drawdown ↓ | Sharpe Ratio ↑ | Win Rate ↑ | Trades | Final Capital ($100k Init) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Buy-and-Hold (SPY Baseline)** | 1.32\% | 24.50\% | 0.17 | — | 0 | \$102,648.54 |
| **ARIMA(2,1,1)-based** | 1.48\% | 6.50\% | 0.28 | 62.50\% | 8 | \$102,949.17 |
| **LSTM-based** | 10.51\% | 8.37\% | 0.91 | 53.66\% | 41 | \$121,924.51 |
| **GRU-based** | 8.56\% | 8.64\% | 0.70 | 46.51\% | 43 | \$117,690.89 |
| **Standalone TS2Vec-based** | 13.14\% | 13.56\% | 0.92 | 58.73\% | 63 | \$127,748.83 |
| **Fused Model-based (GCN-TS2Vec)** | **13.81\%** | **4.62\%** | **1.55** | **66.67\%** | **21** | **\$129,268.69** |

* **The GCN Ablation Proof:** On an identical setup, adding the GCN cut drawdown by **$66\%$** (from $13.56\% \to 4.62\%$) and boosted strategy win rate to **$66.67\%$** (2 out of 3 trades winning).

---

## 📁 Repository Structure

```text
├── dataset.py                # 5D feature pipeline, RevIN binders, and causal T-1 graph synthesis
├── losses.py                 # Tri-channel InfoNCE losses (Temporal, Instance, Spatial Co-Movement)
├── models.py                 # Kipf GCN, TS2Vec SamePadConv blocks, and Interleaved Backbone
├── engine.py                 # RevINQuantEngine, Standalone TS2Vec trainer, and deterministic seeds
├── backtester.py             # Kasui Wei trading rules, causal GARCH(1,1), and USD metrics
├── benchmark.py              # Master script executing Tables 1 & 2 with full ablation
├── plot_thesis_results.py    # Generates publication-grade Figures 5.1 - 5.4 in PDF/PNG
├── plot_neural_architecture.py # Generates Figure 3.2 (Detailed Neural Architecture Blueprint)
├── sp500_sectors_2010_2023.csv # Static cached multi-asset database (2010 - 2023)
├── best_hyperparameters.json # Serialized optimal Optuna hyperparameters
├── requirements.txt          # Minimal Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/HK253-DATN-031-Stock-Prediction.git
cd HK253-DATN-031-Stock-Prediction
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running Experiments & Reproducing Results

All experiments are engineered for **deterministic reproducibility** on commodity hardware (evaluated on an 11th Gen Intel Core i5 laptop with 8GB RAM in pure CPU execution mode).

### Run the Master Benchmark (Tables 1 & 2):
```bash
python benchmark.py
```
*This executes end-to-end training and out-of-sample testing for all 6 models, outputting Tables 1 and 2 directly to your console.*

### Generate All Thesis Plots (Figures 5.1 – 5.4):
```bash
python plot_thesis_results.py
```
*Outputs:*
* `fig1_cumulative_equity.pdf` (Cumulative Portfolio Equity Curves)
* `fig2_underwater_drawdown.pdf` (Underwater Drawdown Curves)
* `fig3_ablation_comparison.pdf` (Ablation: Standalone TS2Vec vs. Fused Model)
* `fig4_price_tracking.pdf` (Price Level and Daily Error Tracking)

### Generate the Neural Architecture Diagram (Figure 3.2):
```bash
python plot_neural_architecture.py
```
*Outputs `fig_model_architecture.pdf` and `fig_model_architecture.png` at 300 DPI.*

---

## 🔬 Methodological Highlights

1. **Strict $T-1$ Causal Graph Isolation:**  
   The dynamic correlation graph for window $b$ is computed strictly over the first $T-1 = 29$ days of the lookback window, completely excluding Day $T$ (today) to guarantee **zero contemporaneous lookahead bias**.
2. **Resolution of the MSE Paradox:**  
   We demonstrate that minimizing price-level MSE forces models like ARIMA to output $\hat{P}_{t+1} \approx P_t$, which achieves low error on paper but yields zero return ($0.0\%$) and zero trades in execution.
3. **CPU Accessibility:**  
   By avoiding the quadratic complexity $\mathcal{O}(T^2)$ of self-attention transformers in favor of linear dilated convolutions $\mathcal{O}(T)$, the complete 14-year pipeline trains in under 4 minutes on a consumer laptop CPU.

---

## 📜 References & Acknowledgments
* **TS2Vec:** Zhihan Yue et al. (AAAI 2022), *TS2Vec: Towards Universal Representation of Time Series*.
* **GCN:** Thomas N. Kipf & Max Welling (ICLR 2017), *Semi-Supervised Classification with Graph Convolutional Networks*.
* **RevIN:** Taesung Kim et al. (ICLR 2022), *Reversible Instance Normalization for Accurate Time-Series Forecasting*.
* **Asset Embeddings:** Rian Dolphin et al. (2024), *Contrastive Learning of Asset Embeddings from Financial Time Series*.
* **Trading Protocol:** Kasui Wei (DECS 2025), *Deep Learning-Based Financial Time Series Forecasting and Quantitative Trading Strategy Optimization*.

---