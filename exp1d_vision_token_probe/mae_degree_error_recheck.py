"""
MAE / Degree-Error Recheck: Is Orientation R2 Actually a Problem?
=====================================================================
Check 2 of the sanity check found roll/pitch/yaw have ~5-10x LOWER
variance than position/gripper in this dataset. R2 is variance-
normalized, so a small absolute error can look catastrophic in R2
terms even if it's practically fine for control.

This script retrains the SAME small MLP probe architecture used in
Experiment 1b (best layer) and reports:

  1. R2 (for continuity with prior experiments)
  2. MAE in raw action units
  3. MAE converted to DEGREES (assuming actions are in radians, the
     LIBERO/pi0 convention -- consistent with roll/pitch ranges of
     ~0.3-0.5, not the 5-30 range you'd expect if units were degrees)
  4. MAE as a PERCENTAGE of each dimension's own range, for a fair
     cross-dimension comparison that isn't distorted by unit choice
  5. A comparison against a trivial "always predict the mean" baseline
     -- if the MLP's MAE is only marginally better than always
     predicting the mean, R2 being low is TELLING THE TRUTH. If MLP's
     MAE is dramatically better than the mean-baseline's MAE, but R2
     still looks bad, that confirms R2 was the misleading part.

No GPU required for interpretation; retraining the small MLP is fast
even on CPU for this dataset size.

Requires: features.npz from Experiment 1a (or 1b) with the same layout.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

FEATURES_PATH = "/kaggle/input/datasets/tahak9/boyfla/features.npz"
SAVE_DIR = Path("/kaggle/working/probe_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

ACTION_DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
ORIENTATION_DIMS = ["roll", "pitch", "yaw"]
PROBE_LAYERS = [-1, -2, -3, -4]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MLP_HIDDEN, MLP_DROPOUT, MLP_WD, MLP_LR = 256, 0.2, 1e-4, 1e-3
MLP_EPOCHS, MLP_PATIENCE, MLP_BATCH = 200, 20, 64


class SmallMLPProbe(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, input_dim, output_dim):
    model = SmallMLPProbe(input_dim, output_dim, MLP_HIDDEN, MLP_DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    loss_fn = nn.MSELoss()
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)

    n = len(X_train_t)
    best_val_loss, best_state, patience_counter = float("inf"), None, 0
    for epoch in range(MLP_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, MLP_BATCH):
            idx = perm[i:i + MLP_BATCH]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train_t[idx]), y_train_t[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        if val_loss < best_val_loss:
            best_val_loss, best_state, patience_counter = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_counter += 1
        if patience_counter >= MLP_PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model


def load_features(features_path=FEATURES_PATH):
    data = np.load(features_path)
    actions_array = data["actions"]
    layer_arrays = {}
    for key in data.files:
        if key.startswith("layer_"):
            suffix = key[len("layer_"):]
            try:
                layer_arrays[int(suffix)] = data[key]
            except ValueError:
                layer_arrays[suffix] = data[key]
    return layer_arrays, actions_array


def main():
    print("=" * 70)
    print("MAE / DEGREE-ERROR RECHECK")
    print("=" * 70)

    layer_arrays, actions_array = load_features()
    n = len(actions_array)
    act_dim = actions_array.shape[1]
    dim_names = ACTION_DIM_NAMES[:act_dim]

    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=SEED)
    subtrain_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=SEED)

    actions_train = actions_array[train_idx]
    actions_test  = actions_array[test_idx]
    y_subtrain    = actions_array[subtrain_idx]
    y_val         = actions_array[val_idx]

    best_layer_key = -2 if -2 in layer_arrays else list(layer_arrays.keys())[0]
    print(f"\nUsing layer {best_layer_key} (best performer in prior experiments)\n")

    X = layer_arrays[best_layer_key]
    X_subtrain, X_val, X_test = X[subtrain_idx], X[val_idx], X[test_idx]

    scaler = StandardScaler()
    X_subtrain_sc = scaler.fit_transform(X_subtrain)
    X_val_sc      = scaler.transform(X_val)
    X_test_sc     = scaler.transform(X_test)

    print("Training MLP probe...")
    model = train_mlp(X_subtrain_sc, y_subtrain, X_val_sc, y_val, X.shape[1], act_dim)

    with torch.no_grad():
        X_test_t = torch.tensor(X_test_sc, dtype=torch.float32, device=DEVICE)
        preds = model(X_test_t).cpu().numpy()

    mean_baseline_pred = np.tile(actions_train.mean(axis=0), (len(actions_test), 1))

    print("\n" + "=" * 70)
    print(f"{'Dim':<10} {'R2':>8} {'MLP MAE':>10} {'Mean MAE':>10} {'MAE % range':>12} {'Deg (if rad)':>13}")
    print("=" * 70)

    results = []
    for i, name in enumerate(dim_names):
        y_true = actions_test[:, i]
        y_pred = preds[:, i]
        y_mean_pred = mean_baseline_pred[:, i]

        r2 = r2_score(y_true, y_pred)
        mae_mlp = mean_absolute_error(y_true, y_pred)
        mae_mean = mean_absolute_error(y_true, y_mean_pred)

        full_range = actions_array[:, i].max() - actions_array[:, i].min()
        mae_pct_range = (mae_mlp / full_range) * 100 if full_range > 0 else float("nan")

        mae_deg = np.degrees(mae_mlp) if name in ORIENTATION_DIMS else float("nan")

        results.append({
            "name": name, "r2": r2, "mae_mlp": mae_mlp, "mae_mean": mae_mean,
            "mae_pct_range": mae_pct_range, "mae_deg": mae_deg,
        })

        deg_str = f"{mae_deg:.2f} deg" if not np.isnan(mae_deg) else "n/a"
        print(f"{name:<10} {r2:>8.4f} {mae_mlp:>10.4f} {mae_mean:>10.4f} "
              f"{mae_pct_range:>11.1f}% {deg_str:>13}")

    print("=" * 70)

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    for r in results:
        if r["name"] not in ORIENTATION_DIMS:
            continue
        improvement_over_mean = (1 - r["mae_mlp"] / r["mae_mean"]) * 100 if r["mae_mean"] > 0 else 0
        print(f"\n{r['name']}:")
        print(f"  R2 = {r['r2']:.4f}  (looks {'bad' if r['r2'] < 0.4 else 'ok'} in isolation)")
        print(f"  MLP MAE = {r['mae_mlp']:.4f} rad = {r['mae_deg']:.2f} degrees")
        print(f"  Mean-baseline MAE = {r['mae_mean']:.4f} rad = {np.degrees(r['mae_mean']):.2f} degrees")
        print(f"  MLP improves over always-predicting-the-mean by {improvement_over_mean:.1f}%")
        print(f"  Error as % of total observed range: {r['mae_pct_range']:.1f}%")

        if improvement_over_mean > 40 and r["mae_deg"] < 5:
            print(f"  -> VERDICT: The MLP prediction is actually GOOD in absolute terms "
                  f"(~{r['mae_deg']:.1f} degree error) and substantially beats the "
                  f"mean baseline. R2 was MISLEADING here due to low target variance.")
        elif improvement_over_mean > 20:
            print(f"  -> VERDICT: The MLP is learning real signal (beats mean baseline "
                  f"meaningfully) but absolute error (~{r['mae_deg']:.1f} deg) may "
                  f"still be too large for precision manipulation tasks. Partial "
                  f"metric artifact, partial real limitation.")
        else:
            print(f"  -> VERDICT: The MLP barely beats the mean baseline. R2 being low "
                  f"appears to reflect a REAL limitation, not just a variance artifact.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Orientation Error: R2 vs Absolute Degree Error", fontsize=13, fontweight="bold")

    ax = axes[0]
    names = [r["name"] for r in results]
    r2_vals = [r["r2"] for r in results]
    colors = ["#e67e22" if n in ORIENTATION_DIMS else "#3498db" for n in names]
    ax.bar(names, r2_vals, color=colors, edgecolor="white")
    ax.axhline(0.4, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.0, color="black", alpha=0.3)
    ax.set_ylabel("R2 Score")
    ax.set_title("R2 by dimension\n(orange = orientation)")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    orient_results = [r for r in results if r["name"] in ORIENTATION_DIMS]
    orient_names = [r["name"] for r in orient_results]
    mlp_degs = [r["mae_deg"] for r in orient_results]
    mean_degs = [np.degrees(r["mae_mean"]) for r in orient_results]

    x = np.arange(len(orient_names))
    width = 0.35
    ax.bar(x - width/2, mlp_degs, width, label="MLP prediction", color="#2ecc71")
    ax.bar(x + width/2, mean_degs, width, label="Always predict mean", color="#95a5a6")
    ax.set_xticks(x)
    ax.set_xticklabels(orient_names)
    ax.set_ylabel("MAE (degrees)")
    ax.set_title("Absolute error: MLP vs trivial mean baseline")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    save_path = SAVE_DIR / "orientation_mae_recheck.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {save_path}")

    print("=" * 70)

    return results


if __name__ == "__main__":
    main()