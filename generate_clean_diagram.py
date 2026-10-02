"""
generate_clean_diagram.py - Generates a perfectly proportioned vertical diagram
specifically designed for A4 thesis pages (saves 'system_workflow.png' at 300 DPI).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_vertical_workflow():
    # Standard portrait proportions for A4 page width
    fig, ax = plt.subplots(figsize=(10, 13), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette
    c_blue   = '#E3F2FD'  # Data & Features
    c_purple = '#EDE7F6'  # Binders & Graphs
    c_teal   = '#E0F2F1'  # Stage 1 Pre-Training
    c_orange = '#FFF3E0'  # Stage 2 Fine-Tuning
    c_green  = '#E8F5E9'  # Execution Engine
    c_border = '#37474F'  # Dark Slate Border
    c_arrow  = '#263238'  # Arrow Color

    def draw_box(x, y, w, h, title, lines=[], bg='#FFFFFF'):
        # Rounded Card Box
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.8,rounding_size=1.2",
            facecolor=bg, edgecolor=c_border, linewidth=1.5
        )
        ax.add_patch(rect)
        # Header Title
        ax.text(x + w/2, y + h - 1.8, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#0D47A1')
        # Subtext lines
        y_text = y + h - 3.8
        for line in lines:
            ax.text(x + w/2, y_text, line, ha='center', va='center', fontsize=8.8, color='#263238')
            y_text -= 1.7

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor=c_arrow, edgecolor=c_arrow, width=1.5, headwidth=6, headlength=6, shrink=0.08)
        )

    # -------------------------------------------------------------
    # 1. DATA INGESTION
    # -------------------------------------------------------------
    draw_box(15, 90.5, 70, 7.5, "1. Multi-Asset Market Data Ingestion (2010 - 2023)", [
        "S&P 500 Index ETF (SPY) + 8 Core Sector ETFs (XLK, XLF, XLE, XLV, etc.)",
        "Raw Panels: Open, High, Low, Close, Volume [Total Days x 9 Assets]"
    ], c_blue)
    draw_arrow(50, 90.5, 50, 85.5)

    # -------------------------------------------------------------
    # 2. FEATURE EXTRACTION
    # -------------------------------------------------------------
    draw_box(10, 77.5, 80, 8.0, "2. 5-Dimensional Feature Engineering Space (C = 5)", [
        "Channel 0: Log Returns (RevIN Price)   |   Channel 1: Rolling Volume Z-Score (k=20)",
        "Channel 2: Garman-Klass Volatility    |   Channel 3: Normalized RSI-14 ([-1, +1])",
        "Channel 4: Normalized MACD Histogram   |   Warmup Truncation: 46 Days"
    ], c_blue)
    draw_arrow(50, 77.5, 50, 72.5)

    # -------------------------------------------------------------
    # 3. BINDERS & CAUSAL GRAPHS (SPLIT ROW)
    # -------------------------------------------------------------
    draw_box(6, 59.5, 42, 13.0, "3A. RevIN Sliding Binders", [
        "Lookback Window: T = 30 Days (Stride s = 1)",
        "Tensor Footprint: X in R^{B x M x T x C}",
        "Local Parameters: mu_b, sigma_b per window",
        "Normalizes prices to N(0, 1) container",
        "Eliminates 14-year decadal price drift"
    ], c_purple)

    draw_box(52, 59.5, 42, 13.0, "3B. Causal Empirical Graphs", [
        "Time-Locked Adjacency: A in {0, 1}^{B x M x M}",
        "Strict T-1 Historical Lookback (29 Days)",
        "Zero Lookahead Bias into Day T (Today)",
        "Pearson Correlation Threshold = 0.40",
        "Dynamic, regime-adaptive sector links"
    ], c_purple)

    # Arrows from both boxes merging into Backbone
    draw_arrow(27, 59.5, 45, 53.0)
    draw_arrow(73, 59.5, 55, 53.0)

    # -------------------------------------------------------------
    # 4. STAGE 1: SELF-SUPERVISED CONTRASTIVE PRE-TRAINING
    # -------------------------------------------------------------
    draw_box(6, 36.5, 88, 16.5, "4. Stage 1: Self-Supervised Tri-Channel Contrastive Pre-Training", [
        "Dual-View Stochastic Augmentations: Temporal Crop (T_crop=20) + Bernoulli Mask (p=0.2) + Edge Dropout (p=0.1)",
        "Interleaved Backbone: Spectral GCN (Kipf & Welling) + Bidirectional TS2Vec ConvBlocks (SamePad + GELU)",
        "Temporal Contrastive Loss (L_Temp): Enforces timestamp-level Contextual Invariance across masked views",
        "Instance Contrastive Loss (L_Inst): Enforces global Scale Invariance across market eras via max-pooling",
        "Spatial Contrastive Loss (L_Spat): Enforces Cross-Asset Sector Co-Movement using empirical graph A",
        "Dynamic Uncertainty Balancing: L_Total = exp(-s1)L_Temp + s1 + exp(-s2)L_Inst + s2 + exp(-s3)L_Spat + s3"
    ], c_teal)
    draw_arrow(50, 36.5, 50, 31.0)

    # -------------------------------------------------------------
    # 5. STAGE 2: SUPERVISED FINE-TUNING & INVERSION
    # -------------------------------------------------------------
    draw_box(6, 18.5, 88, 12.5, "5. Stage 2: Supervised Fine-Tuning & Invertible Inference", [
        "Backbone Parameters FROZEN (Locked Feature Extractor) -> Eliminates Representation Collapse",
        "Coordinate Slicing: Extract terminal Day T latent representation vector: Z_target in R^{B x M x 128}",
        "Linear Predictor Head (MLP): 128 -> 32 -> 1, trained with MSE on RevIN Normalized Targets (y_norm)",
        "RevIN Exact Inversion: P_hat_{t+1} = y_hat_norm * sigma_b + mu_b (Exact USD Price Level Restored)",
        "Expected Return Synthesis: R_hat_{t+1} = (P_hat_{t+1} - P_t) / P_t"
    ], c_orange)
    draw_arrow(50, 18.5, 50, 13.0)

    # -------------------------------------------------------------
    # 6. TRADING EXECUTION ENGINE
    # -------------------------------------------------------------
    draw_box(10, 1.5, 80, 11.5, "6. Causal Quantitative Execution Engine (Out-of-Sample 2022 - 2023)", [
        "Directional Entry Threshold (+/- 0.5% return deadband) + RSI Momentum Filtering (30 < RSI < 70)",
        "Causal GARCH(1,1) Volatility Limits: Dynamic Take-Profit (2.0x sigma) & Stop-Loss (1.5x sigma)",
        "Time-Based De-Risking: 5-Day Maximum Holding Period Exit  |  Realistic 10 bps Transaction Frictions",
        "Empirical Results: 13.81% Annualized Return | 4.62% Max Drawdown | 1.55 Sharpe | 66.67% Win Rate"
    ], c_green)

    plt.tight_layout()
    plt.savefig("system_workflow.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Successfully generated: 'system_workflow.png' (Perfect Vertical Layout, 300 DPI)")

if __name__ == "__main__":
    generate_vertical_workflow()