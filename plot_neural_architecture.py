"""
plot_neural_architecture.py - Generates an in-depth, publication-quality
neural network architectural diagram for Chapter 3 and thesis defense slides.
Saves both 'fig_model_architecture.pdf' and 'fig_model_architecture.png' at 300 DPI.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_neural_architecture():
    fig, ax = plt.subplots(figsize=(15, 12), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette (Modern Academic)
    c_input    = '#E3F2FD'  # Soft Blue
    c_proj     = '#E8EAF6'  # Indigo Tint
    c_gcn      = '#EDE7F6'  # Purple (Spatial GCN)
    c_ts2vec   = '#E0F2F1'  # Teal (Temporal TS2Vec)
    c_norm     = '#FFF8E1'  # Amber (Norm & Activation)
    c_head     = '#FFF3E0'  # Orange (Predictor Head)
    c_out      = '#E8F5E9'  # Green (Dollar Output)
    c_border   = '#37474F'  # Slate Gray
    c_residual = '#D32F2F'  # Crimson Red (Residual Highway)

    def draw_card(x, y, w, h, title, subtext=[], bg='#FFFFFF', edge=c_border, lw=1.5):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=bg, edgecolor=edge, linewidth=lw,
            zorder=2
        )
        ax.add_patch(rect)
        if title:
            ax.text(x + w/2, y + h - 1.8, title, ha='center', va='center', fontsize=10, fontweight='bold', color='#1A237E', zorder=3)
        y_pos = y + h - 3.8
        for line in subtext:
            ax.text(x + w/2, y_pos, line, ha='center', va='center', fontsize=8.2, color='#263238', zorder=3)
            y_pos -= 1.6

    def draw_arrow(x1, y1, x2, y2, color=c_border, lw=1.8, scale=18):
        # zorder=20 forces the arrow and its head to render ON TOP of everything
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='-|>',
            mutation_scale=scale,
            color=color,
            linewidth=lw,
            shrinkA=0, shrinkB=0,
            zorder=20
        )
        ax.add_patch(arrow)

    # =========================================================================
    # LEFT COLUMN: MACRO ARCHITECTURE PIPELINE (Width: 24, from x=3 to x=27)
    # =========================================================================
    draw_card(3, 86, 24, 11, "Input Tensors", [
        "Feature Tensor: X in R^{B x M x T x 5}",
        "Dynamic Graph: A in {0, 1}^{B x M x M}",
        "B: Batch  |  M: 9 Assets  |  T: 30 Days"
    ], c_input)
    draw_arrow(15, 86, 15, 80)

    draw_card(3, 71, 24, 9, "Feature Projection Gateway", [
        "Dense Linear Layer: 5 -> 128",
        "Elevated Space: H^(0) in R^{B x M x T x 128}",
        "Cross-Channel Homogeneity"
    ], c_proj)
    draw_arrow(15, 71, 15, 65)

    draw_card(3, 49, 24, 16, "Stacked Interleaved Backbone", [
        "[ Block l = 0 ]  (Dilation d = 1)",
        "  - Spatial GCN Message Passing",
        "  - Temporal TS2Vec Dilated Conv",
        "  - InstanceNorm1d + ReLU + Residual",
        "               ▼",
        "[ Block l = 1 ]  (Dilation d = 2)",
        "  - Expanded Receptive Field",
        "Output: Z_global in R^{B x M x T x 128}"
    ], c_gcn)
    draw_arrow(15, 49, 15, 43)

    draw_card(3, 34, 24, 9, "Terminal Coordinate Slicing", [
        "Slice at Day t = T (Today's Close)",
        "Z_target = Z_global[:, :, T, :]",
        "Target Embedding: R^{B x M x 128}"
    ], c_head)
    draw_arrow(15, 34, 15, 28)

    draw_card(3, 19, 24, 9, "Linear Predictor Head", [
        "Shallow MLP: 128 -> 32 -> 1",
        "Trained via MSE on RevIN Target",
        "Output: y_hat_norm in R^{B x M}"
    ], c_head)
    draw_arrow(15, 19, 15, 13)

    draw_card(3, 4, 24, 9, "RevIN Exact Inversion", [
        "P_hat_{t+1} = y_hat_norm * sigma_b + mu_b",
        "Predicted Return: (P_hat - P_t) / P_t",
        "Exact Real-World USD Price Restored"
    ], c_out)

    # =========================================================================
    # ZOOM DETAIL CONNECTOR (CLEAR ARROWHEAD & POSITIONED TEXT)
    # =========================================================================
    # Wide corridor from x=27.5 to x=36.5
    draw_arrow(27.5, 57.0, 37.0, 57.0, color='#1A237E', lw=2.2, scale=20)
    # Text centered cleanly ABOVE the arrow with zero collision
    ax.text(32.2, 59.2, "Zoom Detail", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1A237E', zorder=25)

    # =========================================================================
    # RIGHT CONTAINER: DETAILED ZOOM-IN OF ADVANCED INTERLEAVED BLOCK
    # =========================================================================
    # Big Outline Container Box (Starts at x=37.5, ends at x=98.5)
    big_box = patches.FancyBboxPatch(
        (37.5, 3), 60.5, 94,
        boxstyle="round,pad=1.0,rounding_size=1.5",
        facecolor='#FAFAFA', edgecolor='#1A237E', linewidth=2.0, linestyle='--',
        zorder=1
    )
    ax.add_patch(big_box)
    ax.text(67.5, 94.5, "Detailed Architecture: AdvancedInterleavedBlock [Block l]", ha='center', va='center', fontsize=12, fontweight='bold', color='#1A237E', zorder=3)

    # Sub-block 1: Input to Block (Width: 46, from x=41 to x=87)
    draw_card(41, 83.5, 46, 7.5, "Input Hidden Tensor H^(l) in R^{B x M x T x 128}", [
        "Incoming spatial-temporal representation from previous block (or projection)"
    ], '#FFFFFF')
    draw_arrow(64, 83.5, 64, 78)

    # Sub-block 2: Spatial Sub-layer (GCN)
    draw_card(41, 66, 46, 12, "1. Spatial Graph Convolutional Sub-Layer (Kipf & Welling GCN)", [
        "Self-Loop Addition: A~ = A + I_M  (Preserves asset candlestick identity)",
        "Symmetric Degree Normalization: A_hat = D~^{-1/2} A~ D~^{-1/2}",
        "Batch Matrix Multiplication (BMM across batch B):",
        "H_space = BMM(A_hat, H_reshaped)  --> Reshape to [B x M x T x 128]",
        "Captures sector peer co-movement & cross-asset confirmation"
    ], c_gcn)
    draw_arrow(64, 66, 64, 60.5)

    # Sub-block 3: Temporal Reshaping
    draw_card(43, 55, 42, 5.5, "Tensor Permute & Flattening: [B x M x T x 128] --> [(B*M) x 128 x T]", [
        "Isolates each asset timeline so temporal convolutions do not bleed across stocks"
    ], c_proj)
    draw_arrow(64, 55, 64, 49.5)

    # Sub-block 4: Temporal Sub-layer (TS2Vec ConvBlock)
    draw_card(41, 31.5, 46, 18, "2. Temporal TS2Vec Dilated ConvBlock (Yue et al., 2022)", [
        "Dual-Convolution Residual Structure (Dilation d = 2^l):",
        "  - SamePadConv_1 (Symmetric Bidirectional Padding, K=3)",
        "  - GELU Non-Linear Activation Valve",
        "  - SamePadConv_2 (Symmetric Bidirectional Padding, K=3)",
        "  - GELU Non-Linear Activation Valve",
        "  - Internal 1x1 Residual Projection Shortcut: x_conv + x_residual",
        "Maintains exact sequence length invariance: T_out == T == 30"
    ], c_ts2vec)
    draw_arrow(64, 31.5, 64, 26)

    # Sub-block 5: Normalization & Non-Linearity
    draw_card(41, 16.5, 46, 9.5, "3. Temporal InstanceNorm1d & ReLU Activation", [
        "InstanceNorm1d(affine=True): Normalizes each sequence over time independently",
        "Reshape back to canonical 4D layout: [B x M x T x 128]",
        "ReLU Directional Valve: Passes positive momentum, filters noise"
    ], c_norm)
    draw_arrow(64, 16.5, 64, 11)

    # Sub-block 6: Global Block Residual Highway
    draw_card(41, 5, 46, 6, "4. Global Block Residual Highway: H^(l+1) = ReLU(H_norm) + H^(l)", [
        "Gradient Highway: Passes error gradients backward unimpeded (Derivative + I)"
    ], '#FFFFFF')

    # =========================================================================
    # CLEAN, CONTINUOUS RESIDUAL SKIP CONNECTION (CRIMSON RED)
    # =========================================================================
    # Step A: Top exit from Input Box (x=87.5, y=87.2) horizontally out to x=92.0
    ax.plot([87.5, 92.0], [87.2, 87.2], color=c_residual, lw=2.0, zorder=18)

    # Step B: Clean vertical drop down the corridor from y=87.2 to y=8.0
    ax.plot([92.0, 92.0], [87.2, 8.0], color=c_residual, lw=2.0, zorder=18)

    # Step C: Sharp vector arrow turning LEFT directly into Box 4 (+ H^(l))
    draw_arrow(92.0, 8.0, 87.5, 8.0, color=c_residual, lw=2.0, scale=18)

    # Rotated label centered in the right corridor (with zero line overlap)
    ax.text(94.5, 48.0, "Residual Skip Connection  (+ H^(l))", color=c_residual, fontsize=9.5, fontweight='bold', va='center', rotation=-90, zorder=22)

    plt.tight_layout()
    plt.savefig('fig_model_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig_model_architecture.pdf', bbox_inches='tight')
    plt.close()
    print("[+] Successfully regenerated 'fig_model_architecture.png' and 'fig_model_architecture.pdf' (300 DPI)")

if __name__ == "__main__":
    draw_neural_architecture()