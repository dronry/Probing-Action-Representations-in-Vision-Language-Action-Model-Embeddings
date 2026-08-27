"""
Experiment 1b: MLP Decodability Probe (nonlinear follow-up to 1a)
====================================================================
Purpose
-------
Experiment 1a found a STRUCTURED partial result:
    position (x,y,z) + gripper : R2 ~ 0.5-0.7   (decodes well, linearly)
    orientation (roll,pitch,yaw): R2 ~ 0.2-0.45  (decodes poorly, linearly)

This experiment asks: is the orientation failure a LINEARITY problem
(fixable with a small nonlinear decoder) or a REPRESENTATION problem
(the information genuinely isn't there in a usable form)?

We reuse the exact same frozen features already extracted in 1a
(features.npz) — NO GPU NEEDED, no re-running pi0. Only the probe
changes: Ridge regression -> small 2-layer MLP.

Interpretation guide
---------------------
If MLP R2 on roll/pitch >> Ridge R2 on roll/pitch:
    -> Linearity was the bottleneck. A nonlinear decoder head
       (small MLP) is the right architectural choice going forward.
       Core hypothesis (VLM embedding space is action-relevant)
       is SUPPORTED more broadly than 1a suggested.

If MLP R2 on roll/pitch is still low (close to Ridge, or close to 0):
    -> The orientation information is not linearly OR nonlinearly
       recoverable from these hidden states in a simple way.
       Either (a) orientation truly isn't well-represented in this
       layer/space, or (b) more context (e.g. more temporal frames,
       proprioception fused earlier) is needed.
       This would mean the flow-matching decoder needs orientation
       handled differently (e.g. a separate small expert, or fusing
       proprioceptive state more directly) rather than assuming the
       VLM embedding alone carries it.

If MLP R2 on position/gripper improves further:
    -> Bonus evidence the space is even richer than the linear probe
       suggested, useful supporting evidence for the paper regardless
       of the orientation outcome.

Requirements: torch, scikit-learn, matplotlib, numpy (all likely already
installed from Experiment 1a's session; MLP itself is tiny and trains
in seconds to low-minutes on CPU, no GPU required).
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

FEATURES_PATH   = "/kaggle/input/datasets/tahak9/boyfla/features.npz"   # from Experiment 1a
SAVE_DIR        = Path("/kaggle/working/probe_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PROBE_LAYERS    = [-1, -2, -3, -4]
ACTION_DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

# MLP hyperparameters — deliberately SMALL to avoid the comparison
# being unfair (a huge MLP could "win" just by memorizing, which would
# tell us nothing about whether the representation is genuinely
# nonlinearly decodable vs just overfit).
HIDDEN_DIM      = 256
DROPOUT         = 0.2
WEIGHT_DECAY    = 1e-4
LR              = 1e-3
EPOCHS          = 200
BATCH_SIZE      = 64
PATIENCE        = 20     # early stopping patience on val loss
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
SEED            = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device: {DEVICE}")
print(f"MLP architecture: [input] -> {HIDDEN_DIM} -> {HIDDEN_DIM // 2} -> [7]")
print(f"Dropout: {DROPOUT}, Weight decay: {WEIGHT_DECAY}, LR: {LR}\n")


# ──────────────────────────────────────────────
# MODEL: Small 2-layer MLP
# ──────────────────────────────────────────────

class SmallMLPProbe(nn.Module):
    """
    Deliberately small: 2 hidden layers, dropout, no attention/skip tricks.
    The point is to test "does SOME nonlinearity help" not "what's the
    best possible decoder" — a huge decoder would confound the question.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_probe(X_train, y_train, X_val, y_val, input_dim, output_dim):
    """Train one MLP probe with early stopping on validation loss."""
    model = SmallMLPProbe(input_dim, output_dim, HIDDEN_DIM, DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32, device=DEVICE)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32, device=DEVICE)

    n = len(X_train_t)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0

        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"    Early stopping at epoch {epoch} (best val loss: {best_val_loss:.6f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


# ──────────────────────────────────────────────
# MAIN EXPERIMENT
# ──────────────────────────────────────────────

def run_mlp_probes(features_path=FEATURES_PATH):
    print(f"Loading saved features from {features_path}...")
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

    print(f"  {len(actions_array)} samples loaded")
    print(f"  Layers available: {list(layer_arrays.keys())}\n")

    n = len(actions_array)
    act_dim = actions_array.shape[1]
    dim_names = ACTION_DIM_NAMES[:act_dim]

    # Same split as 1a for fair comparison (same seed, same test_size)
    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=42)
    # Further split train into train/val for early stopping
    subtrain_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)

    actions_train = actions_array[train_idx]
    actions_test  = actions_array[test_idx]

    results = {}

    layer_labels = {
        -1: "Layer -1 (last)",
        -2: "Layer -2",
        -3: "Layer -3",
        -4: "Layer -4",
        "avg": "Avg last 4 layers",
    }

    print("=" * 60)
    print("MLP PROBE RESULTS")
    print("=" * 60)

    for key in PROBE_LAYERS + ["avg"]:
        if key not in layer_arrays:
            continue

        X = layer_arrays[key]
        X_subtrain, X_val, X_test = X[subtrain_idx], X[val_idx], X[test_idx]
        y_subtrain, y_val = actions_array[subtrain_idx], actions_array[val_idx]

        # Standardize (fit on subtrain only, matching Ridge protocol from 1a)
        scaler = StandardScaler()
        X_subtrain_sc = scaler.fit_transform(X_subtrain)
        X_val_sc      = scaler.transform(X_val)
        X_test_sc     = scaler.transform(X_test)

        label = layer_labels.get(key, str(key))
        print(f"\nTraining MLP probe: {label}...")

        model = train_mlp_probe(
            X_subtrain_sc, y_subtrain, X_val_sc, y_val,
            input_dim=X.shape[1], output_dim=act_dim
        )

        with torch.no_grad():
            X_test_t = torch.tensor(X_test_sc, dtype=torch.float32, device=DEVICE)
            preds = model(X_test_t).cpu().numpy()

        r2_per_dim  = r2_score(actions_test, preds, multioutput="raw_values")
        r2_overall  = r2_score(actions_test, preds)
        mse_overall = mean_squared_error(actions_test, preds)

        verdict = (
            "PASS"    if r2_overall > 0.7 else
            "PARTIAL" if r2_overall > 0.4 else
            "FAIL"
        )

        results[key] = {
            "label": label,
            "r2_per_dim": r2_per_dim,
            "r2_overall": r2_overall,
            "mse_overall": mse_overall,
            "dim_names": dim_names,
        }

        print(f"  Overall R2:  {r2_overall:.4f}  [{verdict}]")
        print(f"  Overall MSE: {mse_overall:.6f}")
        print("  Per-dim R2:")
        for dname, r2 in zip(dim_names, r2_per_dim):
            bar = "#" * int(max(0, r2) * 20)
            print(f"    {dname:10s}: {r2:+.4f}  |{bar}")

    return results, dim_names


def compare_ridge_vs_mlp(features_path, mlp_results, dim_names):
    """
    Recompute Ridge on the exact same features and split, for a clean
    side-by-side (avoids parsing 1a's text summary — just recompute,
    it's cheap and guarantees an apples-to-apples comparison).
    """
    from sklearn.linear_model import Ridge

    print(f"\nLoading features again for Ridge comparison...")
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

    n = len(actions_array)
    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=42)
    actions_train = actions_array[train_idx]
    actions_test  = actions_array[test_idx]

    ridge_results = {}
    for key in PROBE_LAYERS + ["avg"]:
        if key not in layer_arrays:
            continue
        X = layer_arrays[key]
        X_train, X_test = X[train_idx], X[test_idx]
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)
        probe = Ridge(alpha=1.0)
        probe.fit(X_train_sc, actions_train)
        preds = probe.predict(X_test_sc)
        ridge_results[key] = {
            "r2_overall": r2_score(actions_test, preds),
            "r2_per_dim": r2_score(actions_test, preds, multioutput="raw_values"),
        }

    return ridge_results


def plot_comparison(ridge_results, mlp_results, dim_names):
    """Side-by-side bar chart: Ridge vs MLP, overall and per-dimension."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(
        "Experiment 1b: Linear (Ridge) vs Nonlinear (MLP) Decodability",
        fontsize=13, fontweight="bold"
    )

    ax = axes[0]
    layer_keys = [k for k in PROBE_LAYERS + ["avg"] if k in mlp_results and k in ridge_results]
    labels = [mlp_results[k]["label"] for k in layer_keys]
    ridge_scores = [ridge_results[k]["r2_overall"] for k in layer_keys]
    mlp_scores   = [mlp_results[k]["r2_overall"] for k in layer_keys]

    x = np.arange(len(labels))
    width = 0.35
    ax.barh(x - width/2, ridge_scores, width, label="Ridge (linear)", color="#3498db")
    ax.barh(x + width/2, mlp_scores,   width, label="MLP (nonlinear)", color="#e67e22")
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.axvline(0.7, color="green", linestyle="--", alpha=0.6, label="Pass (0.7)")
    ax.axvline(0.4, color="gray", linestyle="--", alpha=0.6, label="Partial (0.4)")
    ax.set_xlabel("R2 Score")
    ax.set_title("Overall R2: Ridge vs MLP")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    best_key = max(layer_keys, key=lambda k: mlp_results[k]["r2_overall"])
    ridge_dims = ridge_results[best_key]["r2_per_dim"]
    mlp_dims   = mlp_results[best_key]["r2_per_dim"]

    x = np.arange(len(dim_names))
    ax.bar(x - width/2, ridge_dims, width, label="Ridge", color="#3498db")
    ax.bar(x + width/2, mlp_dims,   width, label="MLP", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(dim_names)
    ax.axhline(0.7, color="green", linestyle="--", alpha=0.6)
    ax.axhline(0.4, color="gray", linestyle="--", alpha=0.6)
    ax.axhline(0.0, color="black", linestyle="-", alpha=0.3)
    ax.set_ylabel("R2 Score")
    ax.set_title(f"Per-Dimension: Ridge vs MLP\n({mlp_results[best_key]['label']})")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    save_path = SAVE_DIR / "probe_1b_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved -> {save_path}")

    try:
        from IPython.display import display, Image
        display(Image(str(save_path)))
    except Exception:
        pass

    return best_key


def save_comparison_summary(ridge_results, mlp_results, dim_names):
    path = SAVE_DIR / "probe_1b_summary.txt"
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("EXPERIMENT 1b: RIDGE vs MLP DECODABILITY COMPARISON\n")
        f.write("=" * 60 + "\n\n")

        for key in PROBE_LAYERS + ["avg"]:
            if key not in mlp_results or key not in ridge_results:
                continue
            label = mlp_results[key]["label"]
            r_r2 = ridge_results[key]["r2_overall"]
            m_r2 = mlp_results[key]["r2_overall"]
            delta = m_r2 - r_r2

            f.write(f"{label}\n")
            f.write(f"  Ridge R2: {r_r2:.4f}   MLP R2: {m_r2:.4f}   Delta: {delta:+.4f}\n")
            f.write("  Per-dim (Ridge -> MLP):\n")
            for dname, r_dim, m_dim in zip(
                dim_names, ridge_results[key]["r2_per_dim"], mlp_results[key]["r2_per_dim"]
            ):
                d_delta = m_dim - r_dim
                flag = " <-- notable gain" if d_delta > 0.15 else ""
                f.write(f"    {dname:10s}: {r_dim:+.4f} -> {m_dim:+.4f}  ({d_delta:+.4f}){flag}\n")
            f.write("\n")

    print(f"Comparison summary saved -> {path}")


def main():
    print("=" * 60)
    print("Experiment 1b: MLP Decodability Probe")
    print("Testing whether nonlinearity recovers orientation R2")
    print("=" * 60 + "\n")

    mlp_results, dim_names = run_mlp_probes()
    ridge_results = compare_ridge_vs_mlp(FEATURES_PATH, mlp_results, dim_names)
    best_key = plot_comparison(ridge_results, mlp_results, dim_names)
    save_comparison_summary(ridge_results, mlp_results, dim_names)

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    r_best = ridge_results[best_key]["r2_overall"]
    m_best = mlp_results[best_key]["r2_overall"]
    overall_gain = m_best - r_best

    ridge_dims = ridge_results[best_key]["r2_per_dim"]
    mlp_dims   = mlp_results[best_key]["r2_per_dim"]

    print(f"\nBest layer: {mlp_results[best_key]['label']}")
    print(f"  Ridge overall R2: {r_best:.4f}")
    print(f"  MLP overall R2:   {m_best:.4f}")
    print(f"  Overall gain:     {overall_gain:+.4f}\n")

    orientation_idxs = [i for i, n in enumerate(dim_names) if n in ("roll", "pitch", "yaw")]
    if orientation_idxs:
        orient_ridge_avg = np.mean([ridge_dims[i] for i in orientation_idxs])
        orient_mlp_avg   = np.mean([mlp_dims[i] for i in orientation_idxs])
        orient_gain = orient_mlp_avg - orient_ridge_avg

        print(f"Orientation dims (roll/pitch/yaw) average:")
        print(f"  Ridge: {orient_ridge_avg:.4f}")
        print(f"  MLP:   {orient_mlp_avg:.4f}")
        print(f"  Gain:  {orient_gain:+.4f}\n")

        if orient_gain > 0.2 and orient_mlp_avg > 0.4:
            print("VERDICT: Nonlinearity RECOVERS orientation signal.")
            print("  -> The failure in 1a was a LINEARITY bottleneck, not missing information.")
            print("  -> Recommend: use small MLP decoder instead of pure linear projection")
            print("     in the full architecture. Core hypothesis SUPPORTED more broadly.")
        elif orient_gain > 0.1:
            print("VERDICT: Nonlinearity partially helps orientation, but not enough.")
            print("  -> Some nonlinear structure exists, but orientation remains hard to decode.")
            print("  -> Recommend: hybrid decoder (linear for pos/gripper, larger MLP or")
            print("     auxiliary signal for orientation), or investigate if orientation")
            print("     needs additional temporal context not present in a single frame.")
        else:
            print("VERDICT: Nonlinearity does NOT meaningfully recover orientation.")
            print("  -> The information may genuinely be weakly represented in this layer's")
            print("     hidden state for a single frame, OR orientation requires signal")
            print("     from proprioception/temporal context not captured by this probe.")
            print("  -> Recommend: test whether concatenating proprioceptive state improves")
            print("     orientation R2, before concluding the embedding space itself is at fault.")

    print("=" * 60)

    return ridge_results, mlp_results


if __name__ == "__main__":
    main()