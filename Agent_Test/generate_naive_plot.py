"""Generate naive persistence forecast plot for h=500 to match other model plots."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from data_loader import load_single_building

OUT_DIR = HERE / "naive" / "results" / "forecasts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRES_FIG_DIR = Path(
    r"c:\Users\verwalter\Desktop\Wind_ML_TimeSeries\Thesis\Latex_Document\presentation\figures"
)

# Same defaults used by every other model in train_utils.py
data = load_single_building(
    alpha="Alpha1_4", ratio="2_1_3", seq_length=100, step=10, horizon=1
)
mu_w, sigma_w = float(data["mu"][0]), float(data["sigma"][0])
test_seed = data["test_seed"]           # (100, 4), normalised
y_future = data["y_future"]             # (500,), normalised windward

# Naive persistence: forecast = last observed windward value, repeated
last_val_norm = float(test_seed[-1, 0])
preds_norm = np.full(500, last_val_norm, dtype=np.float32)

# Denormalise
gt = y_future * sigma_w + mu_w
pred = preds_norm * sigma_w + mu_w

# Save npy for completeness
np.save(OUT_DIR / "naive_h500.npy", pred)

# Plot — identical style to save_forecasts() in train_utils.py
n = min(len(gt), len(pred))
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(gt[:n], label="Ground Truth", color="royalblue", lw=1.2)
ax.plot(pred[:n], label="Forecast (h=500)", color="tomato", lw=1.2, ls="--")
ax.set(xlabel="Step", ylabel="Cp (windward)",
       title="naive — Multi-step Forecast (h=500)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()

local_path = OUT_DIR / "naive_forecast_h500.png"
fig.savefig(local_path, dpi=150)
fig.savefig(PRES_FIG_DIR / "naive_forecast_h500.png", dpi=150)
plt.close(fig)

# Quick metric so user knows what they're showing
rmse = float(np.sqrt(np.mean((gt[:n] - pred[:n]) ** 2)))
print(f"Naive RMSE @ h=500: {rmse:.4f}")
print(f"Saved: {local_path}")
print(f"Saved: {PRES_FIG_DIR / 'naive_forecast_h500.png'}")
