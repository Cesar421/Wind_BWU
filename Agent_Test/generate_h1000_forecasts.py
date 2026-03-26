"""
Generate h=1000 forecasts and visualizations for all models
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import load_single_building
from train_utils import DEVICE

# Import all models
from models.cnn_lstm import CNNLSTM
sys.path.insert(0, str(ROOT / "lstm"))
from models.lstm import PureLSTM
sys.path.insert(0, str(ROOT / "tcn"))
from models.tcn import TCN
sys.path.insert(0, str(ROOT / "transformer"))
from models.transformer import TransformerForecaster
sys.path.insert(0, str(ROOT / "ann"))
from models.ann import FeedforwardANN

# Config
SEQ_LENGTH = 100
STEP = 10
HORIZON = 1
BATCH_SIZE = 256

def load_model_checkpoint(model, checkpoint_path):
    """Load trained model from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model

def autoregressive_forecast(model, seed_seq, horizon, mu, sigma):
    """
    Generate autoregressive forecast for h steps.
    
    Args:
        model: trained PyTorch model
        seed_seq: initial sequence (seq_length, n_features) normalized
        horizon: number of steps to forecast
        mu, sigma: denormalization parameters
    
    Returns:
        forecast: (horizon,) array of denormalized predictions
    """
    model.eval()
    seq = torch.tensor(seed_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, seq_len, 4)
    forecast = []
    
    with torch.no_grad():
        for _ in range(horizon):
            pred_norm = model(seq).cpu().numpy()[0, 0]  # scalar
            pred = pred_norm * sigma + mu  # denormalize
            forecast.append(pred)
            
            # Update sequence: shift left and append prediction
            # We need to append [pred_norm, 0, 0, 0] as new timestep
            # (only windward Cp is predicted, rest are zeros)
            new_step = torch.zeros((1, 1, 4), dtype=torch.float32, device=seq.device)
            new_step[0, 0, 0] = torch.tensor(pred_norm, dtype=torch.float32)
            seq = torch.cat([seq[:, 1:, :], new_step], dim=1)
    
    return np.array(forecast)

def main():
    print("="*70)
    print("Generating h=1000 forecasts for all models (Alpha1_4/2_1_3)")
    print("="*70)
    
    # Load data
    data = load_single_building(alpha='Alpha1_4', ratio='2_1_3',
                                seq_length=SEQ_LENGTH, step=STEP, horizon=HORIZON)
    mu_w = float(data['mu'][0])
    sigma_w = float(data['sigma'][0])
    seed_seq = data['test_seed']  # (seq_length, 4)
    y_future = data['y_future']   # normalized ground truth
    gt = y_future * sigma_w + mu_w  # denormalized
    
    print(f"Ground truth length: {len(gt)}")
    print(f"Mu: {mu_w:.4f}, Sigma: {sigma_w:.4f}")
    
    # Define models and checkpoints
    models_config = {
        'cnn_lstm': {
            'model': CNNLSTM(
                n_features=4, seq_length=SEQ_LENGTH, horizon=HORIZON,
                cnn_channels=(64, 128, 256), lstm_hidden=256,
                lstm_layers=2, lstm_dropout=0.2,
                fc_hidden=128, dropout=0.3
            ),
            'checkpoint': ROOT / 'cnn_lstm' / 'checkpoints' / 'cnn_lstm_best.pt',
            'color': '#e74c3c'
        },
        'lstm': {
            'model': PureLSTM(
                n_features=4, hidden_size=256, num_layers=2, 
                dropout=0.2, fc_hidden=128, fc_dropout=0.3, horizon=HORIZON
            ),
            'checkpoint': ROOT / 'lstm' / 'checkpoints' / 'lstm_best.pt',
            'color': '#3498db'
        },
        'tcn': {
            'model': TCN(
                n_features=4, num_channels=(64, 128, 256), 
                kernel_size=3, dropout=0.2, fc_hidden=128, horizon=HORIZON
            ),
            'checkpoint': ROOT / 'tcn' / 'checkpoints' / 'tcn_best.pt',
            'color': '#2ecc71'
        },
        'transformer': {
            'model': TransformerForecaster(
                n_features=4, d_model=128, nhead=8, num_layers=3, 
                dim_feedforward=256, dropout=0.1, horizon=HORIZON, seq_length=SEQ_LENGTH
            ),
            'checkpoint': ROOT / 'transformer' / 'checkpoints' / 'transformer_best.pt',
            'color': '#9b59b6'
        },
        'ann': {
            'model': FeedforwardANN(
                n_features=4, seq_length=SEQ_LENGTH, 
                hidden_layers=(100, 500, 500, 100), dropout=0.3, horizon=HORIZON
            ),
            'checkpoint': ROOT / 'ann' / 'checkpoints' / 'ann_best.pt',
            'color': '#f39c12'
        }
    }
    
    # Generate forecasts
    forecasts = {}
    print("\nGenerating forecasts...")
    for name, config in models_config.items():
        print(f"  {name.upper()}...", end=" ", flush=True)
        try:
            model = load_model_checkpoint(config['model'], config['checkpoint'])
            model = model.to(DEVICE)
            forecast = autoregressive_forecast(model, seed_seq, 1000, mu_w, sigma_w)
            forecasts[name] = forecast
            
            # Save forecast
            forecast_dir = ROOT / name / 'results' / 'forecasts'
            forecast_dir.mkdir(parents=True, exist_ok=True)
            np.save(forecast_dir / f'{name}_h1000.npy', forecast)
            print(f"✓ (saved to {name}/results/forecasts/)")
        except Exception as e:
            print(f"✗ Error: {e}")
            forecasts[name] = None
    
    # Create visualizations
    print("\nGenerating visualizations...")
    results_dir = ROOT / 'results' / 'plots'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out failed models
    valid_models = {k: v for k, v in forecasts.items() if v is not None}
    
    if len(valid_models) == 0:
        print("No valid forecasts generated. Exiting.")
        return
    
    # === PLOT 1: All models overlaid (h=1000) ===
    fig, ax = plt.subplots(figsize=(20, 8))
    n = min(1000, len(gt))
    ax.plot(range(n), gt[:n], 'k-', lw=2.5, label='Ground Truth', zorder=10, alpha=0.8)
    
    for name, forecast in valid_models.items():
        color = models_config[name]['color']
        ax.plot(range(n), forecast[:n], lw=1.5, color=color, 
                alpha=0.85, label=name.upper())
    
    ax.set_title('1000-Step Autoregressive Forecasts - Alpha1_4/2_1_3 (All Models)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Forecast Step', fontsize=13)
    ax.set_ylabel('Cp (windward)', fontsize=13)
    ax.legend(fontsize=11, loc='best', ncol=3)
    ax.grid(alpha=0.3, linestyle='--')
    fig.tight_layout()
    out1 = results_dir / 'all_models_h1000_overlay.png'
    fig.savefig(str(out1), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out1}")
    
    # === PLOT 2: Individual model panels (h=1000) ===
    n_models = len(valid_models)
    fig, axes = plt.subplots(n_models, 1, figsize=(20, 4*n_models), sharex=True)
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, forecast) in zip(axes, valid_models.items()):
        color = models_config[name]['color']
        ax.plot(range(n), gt[:n], 'k-', lw=1.8, label='Ground Truth', alpha=0.7)
        ax.plot(range(n), forecast[:n], lw=1.5, color=color, 
                alpha=0.9, label=f'{name.upper()} forecast')
        rmse = float(np.sqrt(np.mean((gt[:n] - forecast[:n])**2)))
        mae = float(np.mean(np.abs(gt[:n] - forecast[:n])))
        ax.set_title(f'{name.upper()} - h=1000  (RMSE={rmse:.4f}, MAE={mae:.4f})', 
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Cp', fontsize=11)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(alpha=0.3)
    
    axes[-1].set_xlabel('Forecast Step', fontsize=13)
    fig.suptitle('1000-Step Autoregressive Forecasts by Model', 
                 fontsize=16, fontweight='bold', y=1.002)
    fig.tight_layout()
    out2 = results_dir / 'all_models_h1000_individual.png'
    fig.savefig(str(out2), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out2}")
    
    # === PLOT 3: Error accumulation over 1000 steps ===
    fig, ax = plt.subplots(figsize=(16, 7))
    for name, forecast in valid_models.items():
        color = models_config[name]['color']
        cum_err = np.cumsum(np.abs(gt[:n] - forecast[:n]))
        ax.plot(range(n), cum_err, lw=2.5, color=color, label=name.upper())
    
    ax.set_xlabel('Forecast Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Absolute Error', fontsize=13, fontweight='bold')
    ax.set_title('Error Accumulation Over 1000-Step Horizon', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    fig.tight_layout()
    out3 = results_dir / 'forecast_error_accumulation_h1000.png'
    fig.savefig(str(out3), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out3}")
    
    # === PLOT 4: Multi-horizon comparison (500 vs 1000) ===
    # Try to load h=500 forecasts
    forecasts_500 = {}
    for name in valid_models.keys():
        f500_path = ROOT / name / 'results' / 'forecasts' / f'{name}_h500.npy'
        if f500_path.exists():
            forecasts_500[name] = np.load(str(f500_path))
    
    if len(forecasts_500) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(20, 12), sharex=False)
        
        # h=500 panel
        n500 = min(500, len(gt))
        axes[0].plot(range(n500), gt[:n500], 'k-', lw=2.5, 
                     label='Ground Truth', zorder=10, alpha=0.8)
        for name, forecast in forecasts_500.items():
            color = models_config[name]['color']
            axes[0].plot(range(n500), forecast[:n500], lw=1.5, 
                        color=color, alpha=0.85, label=name.upper())
        axes[0].set_title('Horizon = 500 steps', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Cp (windward)', fontsize=12)
        axes[0].legend(fontsize=10, ncol=3, loc='best')
        axes[0].grid(alpha=0.3)
        
        # h=1000 panel
        axes[1].plot(range(n), gt[:n], 'k-', lw=2.5, 
                     label='Ground Truth', zorder=10, alpha=0.8)
        for name, forecast in valid_models.items():
            color = models_config[name]['color']
            axes[1].plot(range(n), forecast[:n], lw=1.5, 
                        color=color, alpha=0.85, label=name.upper())
        axes[1].set_title('Horizon = 1000 steps', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Forecast Step', fontsize=12)
        axes[1].set_ylabel('Cp (windward)', fontsize=12)
        axes[1].legend(fontsize=10, ncol=3, loc='best')
        axes[1].grid(alpha=0.3)
        
        fig.suptitle('Multi-Horizon Comparison - Alpha1_4/2_1_3', 
                     fontsize=16, fontweight='bold')
        fig.tight_layout()
        out4 = results_dir / 'comparison_h500_vs_h1000.png'
        fig.savefig(str(out4), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out4}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - h=1000 Forecast Metrics")
    print("="*70)
    print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'Final Error':<15}")
    print("-"*70)
    for name, forecast in valid_models.items():
        rmse = float(np.sqrt(np.mean((gt[:n] - forecast[:n])**2)))
        mae = float(np.mean(np.abs(gt[:n] - forecast[:n])))
        final_err = float(abs(gt[n-1] - forecast[n-1]))
        print(f"{name.upper():<15} {rmse:<10.4f} {mae:<10.4f} {final_err:<15.4f}")
    
    print("\nDone! All h=1000 forecasts and plots saved.")

if __name__ == "__main__":
    main()
