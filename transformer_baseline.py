import os
import torch
import torch.nn as nn
import numpy as np

# Import your existing pipeline modules
from dataset import build_or_load_dataset, process_features, generate_revin_binders_and_prices, DEFAULT_TICKERS
from backtester import compute_wei_usd_metrics
from engine import set_seed
import iTransformer
import PatchTST

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def train_transformer_model(model, X_train, y_train, X_val, y_val, epochs=50, lr=1e-3, patience=10, device='cpu'):
    train_data = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    model.to(device)
    
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            # iTransformer requires 4 inputs, PatchTST requires 1
            if model.__class__.__name__ == 'Model' and hasattr(model, 'forecast'):
                x_mark_enc = torch.zeros(batch_x.size(0), batch_x.size(1), 1).to(device)
                x_dec = torch.zeros(batch_x.size(0), 1, batch_x.size(2)).to(device)
                x_mark_dec = torch.zeros(batch_x.size(0), 1, 1).to(device)
                outputs = model(batch_x, x_mark_enc, x_dec, x_mark_dec)
            else:
                outputs = model(batch_x)
                
            if len(outputs.shape) == 3:
                outputs = outputs[:, -1, 0]
            
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'forecast'):
                x_mark_enc = torch.zeros(X_val_t.size(0), X_val_t.size(1), 1).to(device)
                x_dec = torch.zeros(X_val_t.size(0), 1, X_val_t.size(2)).to(device)
                x_mark_dec = torch.zeros(X_val_t.size(0), 1, 1).to(device)
                val_out = model(X_val_t, x_mark_enc, x_dec, x_mark_dec)
            else:
                val_out = model(X_val_t)
            
            if len(val_out.shape) == 3:
                val_out = val_out[:, -1, 0]
                
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

def main():
    print("="*80)
    print(" TRANSFORMER BASELINE (iTransformer & PatchTST)")
    print("="*80)
    set_seed(42)
    
    # 1. Load Data
    print("[1/3] Loading Data...")
    master_df = build_or_load_dataset(tickers=DEFAULT_TICKERS, start_date="2010-01-01", end_date="2023-12-31", cache_file="sp500_sectors_2010_2023.csv")
    feature_tensor, aligned_dates, prices_close_df = process_features(master_df, DEFAULT_TICKERS, k=20)
    spy_idx = DEFAULT_TICKERS.index('SPY')
    T = 30
    
    X_all, A_all, y_norm_all, y_usd_all, mu_all, sigma_all = generate_revin_binders_and_prices(
        prices_close_df, feature_tensor, T=T, corr_threshold=0.40
    )
    
    decision_dates = aligned_dates[T - 1: T - 1 + len(X_all)]
    train_mask = (decision_dates >= "2010-01-01") & (decision_dates <= "2020-12-31")
    val_mask = (decision_dates >= "2021-01-01") & (decision_dates <= "2021-12-31")
    test_mask = (decision_dates >= "2022-01-01") & (decision_dates <= "2023-12-31")
    
    test_idx = np.where(test_mask)[0]
    y_test_usd = y_usd_all[test_idx]
    mu_test, sigma_test = mu_all[test_idx], sigma_all[test_idx]
    
    # The 'Fairness' Constraint: Muting spatial features for single-asset SPY input
    X_train_spy = X_all[train_mask, spy_idx, :, :]
    y_train_spy = y_norm_all[train_mask, spy_idx]
    X_val_spy = X_all[val_mask, spy_idx, :, :]
    y_val_spy = y_norm_all[val_mask, spy_idx]
    X_test_spy = X_all[test_mask, spy_idx, :, :]
    
    configs = DotDict({
        'seq_len': T, 'pred_len': 1, 'output_attention': False, 'use_norm': True,
        'd_model': 64, 'embed': 'timeF', 'freq': 'd', 'dropout': 0.1, 
        'class_strategy': 'projection', 'factor': 1, 'n_heads': 4, 'd_ff': 128, 
        'activation': 'gelu', 'e_layers': 2, 'enc_in': 5, 'patch_len': 15, 
        'stride': 8, 'padding_patch': 'end', 'individual': 0, 'revin': 1, 
        'affine': 0, 'subtract_last': 0, 'decomposition': 0, 'kernel_size': 25, 
        'fc_dropout': 0.1, 'head_dropout': 0.1
    })
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("[2/3] Training iTransformer (Restricted 5D SPY Input)...")
    set_seed(42)
    itransformer = iTransformer.Model(configs)
    itransformer = train_transformer_model(itransformer, X_train_spy, y_train_spy, X_val_spy, y_val_spy, device=device)
    
    itransformer.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_spy, dtype=torch.float32).to(device)
        x_mark_enc = torch.zeros(X_test_t.size(0), X_test_t.size(1), 1).to(device)
        x_dec = torch.zeros(X_test_t.size(0), 1, X_test_t.size(2)).to(device)
        x_mark_dec = torch.zeros(X_test_t.size(0), 1, 1).to(device)
        preds_i = itransformer(X_test_t, x_mark_enc, x_dec, x_mark_dec)
        if len(preds_i.shape) == 3: preds_i = preds_i[:, -1, 0]
        preds_i = preds_i.cpu().numpy()
        
    itrans_usd = preds_i * sigma_test[:, spy_idx] + mu_test[:, spy_idx]
    
    print("[3/3] Training PatchTST (Restricted 5D SPY Input)...")
    set_seed(42)
    patchtst = PatchTST.Model(configs)
    patchtst = train_transformer_model(patchtst, X_train_spy, y_train_spy, X_val_spy, y_val_spy, device=device)
    
    patchtst.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_spy, dtype=torch.float32).to(device)
        preds_p = patchtst(X_test_t)
        if len(preds_p.shape) == 3: preds_p = preds_p[:, -1, 0]
        preds_p = preds_p.cpu().numpy()
        
    patch_usd = preds_p * sigma_test[:, spy_idx] + mu_test[:, spy_idx]
    
    print("\n" + "="*80)
    print(" Transformer Baseline Evaluation on S&P 500 (SPY)")
    print("="*80)
    print(f"{'Model Architecture':<30} | {'MAE (USD)':<10} | {'MSE (USD²)':<12} | {'RMSE (USD)':<10} | {'MAPE (%)':<8}")
    print("-" * 80)
    
    mae, mse, rmse, mape = compute_wei_usd_metrics(y_test_usd[:, spy_idx], itrans_usd)
    print(f"{'iTransformer (Restricted)':<30} | ${mae:.2f}      | {mse:.2f}       | ${rmse:.2f}      | {mape:>5.2f}%")
    
    mae, mse, rmse, mape = compute_wei_usd_metrics(y_test_usd[:, spy_idx], patch_usd)
    print(f"{'PatchTST (Restricted)':<30} | ${mae:.2f}      | {mse:.2f}       | ${rmse:.2f}      | {mape:>5.2f}%")

if __name__ == '__main__':
    main()
