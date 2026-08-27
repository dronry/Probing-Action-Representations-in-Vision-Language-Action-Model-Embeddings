"""
Orientation-Only Expert: Design + Reference Implementation
==============================================================
See prior chat discussion for full context. Short version: position
and gripper decode well from frozen pi0/PaliGemma hidden states;
orientation (roll/pitch/yaw) caps around R2~0.25-0.42 across every
decoder variant tried, and this is NOT primarily a metric artifact
(confirmed via MAE recheck against a mean-baseline).

This script prototypes a SPLIT-HEAD architecture: a shared trunk plus
two heads with different conditioning emphasis -- one for position/
gripper (matches the original VLM-embedding-flow proposal), one for
orientation (gets a direct skip connection to the raw input, not just
the shared trunk, so it can use unmixed conditioning signal).

This is a diagnostic prototype using already-extracted features (an
MLP, not the real flow-matching decoder). If it shows a real gain here,
build the same conditioning-separation pattern into the actual
rectified-flow architecture next.

Requires: features.npz (from Experiment 1a/1b), optionally
vision_token_features.npz (from Experiment 1d) for richer conditioning.
No GPU strictly required.
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

LANG_FEATURES_PATH   = "/kaggle/input/datasets/tahak9/boyfla/features.npz"
VISION_FEATURES_PATH = "/kaggle/working/probe_results/vision_token_features.npz"
SAVE_DIR = Path("/kaggle/working/probe_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

ACTION_DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
POSITION_GRIPPER_IDXS = [0, 1, 2, 6]
ORIENTATION_IDXS = [3, 4, 5]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

TRUNK_HIDDEN = 256
HEAD_HIDDEN  = 128
DROPOUT = 0.2
WD, LR = 1e-4, 1e-3
EPOCHS, PATIENCE, BATCH = 250, 25, 64


class SplitHeadProbe(nn.Module):
    def __init__(self, input_dim, trunk_hidden=256, head_hidden=128, dropout=0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, trunk_hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(trunk_hidden, head_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 4),
        )
        self.orient_head = nn.Sequential(
            nn.Linear(trunk_hidden + input_dim, head_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, 3),
        )

    def forward(self, x):
        trunk_out = self.trunk(x)
        pos_pred = self.pos_head(trunk_out)
        orient_pred = self.orient_head(torch.cat([trunk_out, x], dim=-1))
        out = torch.zeros(x.shape[0], 7, device=x.device, dtype=x.dtype)
        out[:, POSITION_GRIPPER_IDXS] = pos_pred
        out[:, ORIENTATION_IDXS] = orient_pred
        return out


class BaselineJointProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 7),
        )
    def forward(self, x):
        return self.net(x)


def train_model(model, X_train, y_train, X_val, y_val):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)

    n = len(X_train_t)
    best_val_loss, best_state, patience_counter = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
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
        if patience_counter >= PATIENCE:
            print(f"    Early stopping at epoch {epoch} (best val loss: {best_val_loss:.6f})")
            break
    model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate(model, X_test, y_test):
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
        preds = model(X_test_t).cpu().numpy()
    r2_dims = r2_score(y_test, preds, multioutput="raw_values")
    r2_overall = r2_score(y_test, preds)
    mae_dims = [mean_absolute_error(y_test[:, i], preds[:, i]) for i in range(y_test.shape[1])]
    orient_r2 = np.mean([r2_dims[i] for i in ORIENTATION_IDXS])
    orient_mae_deg = np.mean([np.degrees(mae_dims[i]) for i in ORIENTATION_IDXS])
    return {
        "r2_overall": r2_overall, "r2_dims": r2_dims, "mae_dims": mae_dims,
        "orient_r2": orient_r2, "orient_mae_deg": orient_mae_deg,
    }


def load_and_assemble_features():
    print("Loading language-token features (required)...")
    lang_data = np.load(LANG_FEATURES_PATH)
    actions_array = lang_data["actions"]
    lang_key = "layer_-2" if "layer_-2" in lang_data else [
        k for k in lang_data.files if k.startswith("layer_")
    ][0]
    lang_best = lang_data[lang_key]
    print(f"  Language features: {lang_best.shape} ({lang_key})")

    feature_blocks = [lang_best]
    block_names = [f"language_{lang_key}"]

    try:
        vision_data = np.load(VISION_FEATURES_PATH)
        v_actions = vision_data["actions"]
        if len(v_actions) != len(actions_array):
            print(f"  WARNING: vision features have {len(v_actions)} samples, "
                  f"language features have {len(actions_array)}. Skipping vision features "
                  f"(different frame sets, cannot safely concatenate).")
        else:
            match = np.mean(np.isclose(v_actions, actions_array, atol=1e-4))
            if match < 0.95:
                print(f"  WARNING: vision/language actions only {match*100:.1f}% match. "
                      f"Skipping vision features.")
            else:
                vkeys = [k for k in vision_data.files if k != "actions"]
                vision_key = "-2_mean" if "-2_mean" in vkeys else vkeys[0]
                vision_best = vision_data[vision_key]
                feature_blocks.append(vision_best)
                block_names.append(f"vision_{vision_key}")
                print(f"  Vision features aligned and added: {vision_best.shape} ({vision_key})")
    except FileNotFoundError:
        print(f"  Vision-token features not found at {VISION_FEATURES_PATH} -- "
              f"proceeding WITHOUT vision conditioning. This likely understates the "
              f"split-head design's potential (Experiment 1d showed vision tokens "
              f"carry somewhat more orientation signal than language tokens).")

    print("\nAttempting to re-stream proprioception for full conditioning...")
    try:
        proprio = _restream_proprioception(actions_array)
        feature_blocks.append(proprio)
        block_names.append("proprioception")
        print(f"  Proprioception added: {proprio.shape}")
    except Exception as e:
        print(f"  Could not re-stream proprioception ({e}) -- proceeding without it.")

    X = np.concatenate(feature_blocks, axis=1)
    print(f"\nFinal assembled feature dim: {X.shape[1]}  (blocks: {block_names})")
    return X, actions_array, block_names


def _restream_proprioception(actions_array, dataset_id="lerobot/libero_spatial_image",
                               num_frames=1500, pool_mult=4):
    from datasets import load_dataset
    pool_size = num_frames * pool_mult
    raw_ds = load_dataset(dataset_id, split="train", streaming=True)
    pool = []
    for i, sample in enumerate(raw_ds):
        if i >= pool_size:
            break
        pool.append(sample)
    indices = np.linspace(0, len(pool) - 1, min(num_frames, len(pool)), dtype=int)

    proprio_list, action_check = [], []
    for idx in indices:
        s = pool[int(idx)]
        state = s["observation.state"]
        proprio_list.append(np.array(state, dtype=np.float32) if not isinstance(state, np.ndarray) else state)
        act = s.get("action", s.get("actions"))
        act = np.array(act, dtype=np.float32) if not isinstance(act, np.ndarray) else act
        if act.ndim > 1:
            act = act[0]
        action_check.append(act)

    proprio_array = np.stack(proprio_list, axis=0)
    action_check = np.stack(action_check, axis=0)

    if action_check.shape != actions_array.shape:
        raise RuntimeError(f"Shape mismatch: {action_check.shape} vs {actions_array.shape}")
    match = np.mean(np.isclose(action_check, actions_array, atol=1e-4))
    if match < 0.95:
        raise RuntimeError(f"Alignment only {match*100:.1f}% -- refusing to use.")

    return proprio_array


def main():
    print("=" * 70)
    print("Orientation-Only Expert: Split-Head Design Prototype")
    print("=" * 70 + "\n")

    X, actions_array, block_names = load_and_assemble_features()
    n = len(actions_array)
    dim_names = ACTION_DIM_NAMES

    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=SEED)
    subtrain_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=SEED)

    scaler = StandardScaler()
    X_subtrain_sc = scaler.fit_transform(X[subtrain_idx])
    X_val_sc = scaler.transform(X[val_idx])
    X_test_sc = scaler.transform(X[test_idx])

    y_subtrain = actions_array[subtrain_idx]
    y_val = actions_array[val_idx]
    y_test = actions_array[test_idx]

    input_dim = X.shape[1]

    print("\n" + "=" * 70)
    print("BASELINE: Joint MLP (no split, same total capacity)")
    print("=" * 70)
    baseline = BaselineJointProbe(input_dim, hidden_dim=TRUNK_HIDDEN, dropout=DROPOUT).to(DEVICE)
    baseline = train_model(baseline, X_subtrain_sc, y_subtrain, X_val_sc, y_val)
    baseline_metrics = evaluate(baseline, X_test_sc, y_test)
    print(f"\nBaseline overall R2: {baseline_metrics['r2_overall']:.4f}")
    print(f"Baseline orientation R2: {baseline_metrics['orient_r2']:.4f}")
    print(f"Baseline orientation MAE: {baseline_metrics['orient_mae_deg']:.2f} degrees")

    print("\n" + "=" * 70)
    print("SPLIT-HEAD MODEL (position/gripper head + orientation head)")
    print("=" * 70)
    split_model = SplitHeadProbe(input_dim, TRUNK_HIDDEN, HEAD_HIDDEN, DROPOUT).to(DEVICE)
    split_model = train_model(split_model, X_subtrain_sc, y_subtrain, X_val_sc, y_val)
    split_metrics = evaluate(split_model, X_test_sc, y_test)
    print(f"\nSplit-head overall R2: {split_metrics['r2_overall']:.4f}")
    print(f"Split-head orientation R2: {split_metrics['orient_r2']:.4f}")
    print(f"Split-head orientation MAE: {split_metrics['orient_mae_deg']:.2f} degrees")

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'Baseline (joint)':>18} {'Split-head':>14}")
    print(f"{'Overall R2':<30} {baseline_metrics['r2_overall']:>18.4f} {split_metrics['r2_overall']:>14.4f}")
    print(f"{'Orientation R2':<30} {baseline_metrics['orient_r2']:>18.4f} {split_metrics['orient_r2']:>14.4f}")
    print(f"{'Orientation MAE (deg)':<30} {baseline_metrics['orient_mae_deg']:>18.2f} {split_metrics['orient_mae_deg']:>14.2f}")

    print("\nPer-dimension R2:")
    print(f"{'Dim':<10} {'Baseline':>12} {'Split-head':>12}")
    for i, name in enumerate(dim_names):
        print(f"{name:<10} {baseline_metrics['r2_dims'][i]:>12.4f} {split_metrics['r2_dims'][i]:>12.4f}")

    orient_gain = split_metrics["orient_r2"] - baseline_metrics["orient_r2"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Split-Head Design vs Joint Baseline", fontsize=13, fontweight="bold")

    ax = axes[0]
    x = np.arange(len(dim_names))
    width = 0.35
    ax.bar(x - width/2, baseline_metrics["r2_dims"], width, label="Baseline (joint)", color="#95a5a6")
    ax.bar(x + width/2, split_metrics["r2_dims"], width, label="Split-head", color="#9b59b6")
    ax.set_xticks(x)
    ax.set_xticklabels(dim_names)
    ax.axhline(0.0, color="black", alpha=0.3)
    ax.set_ylabel("R2")
    ax.set_title("Per-Dimension R2")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    orient_names = [dim_names[i] for i in ORIENTATION_IDXS]
    baseline_orient_dims = [baseline_metrics["r2_dims"][i] for i in ORIENTATION_IDXS]
    split_orient_dims = [split_metrics["r2_dims"][i] for i in ORIENTATION_IDXS]
    x2 = np.arange(len(orient_names))
    ax.bar(x2 - width/2, baseline_orient_dims, width, label="Baseline", color="#95a5a6")
    ax.bar(x2 + width/2, split_orient_dims, width, label="Split-head", color="#9b59b6")
    ax.set_xticks(x2)
    ax.set_xticklabels(orient_names)
    ax.axhline(0.0, color="black", alpha=0.3)
    ax.set_ylabel("R2")
    ax.set_title("Orientation Dimensions: Baseline vs Split-Head")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    save_path = SAVE_DIR / "orientation_expert_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {save_path}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"\nOrientation R2 gain from split-head design: {orient_gain:+.4f}")

    if orient_gain > 0.15:
        print("\nThe split-head design with dedicated orientation conditioning "
              "produces a MEANINGFUL improvement. Recommend building this pattern "
              "into the actual flow-matching architecture.")
    elif orient_gain > 0.05:
        print("\nModest improvement. Some value in separating conditioning "
              "pathways, but orientation remains fundamentally harder. Consider "
              "whether the practical MAE (degrees) is acceptable regardless.")
    else:
        print("\nNo meaningful improvement from architectural restructuring alone. "
              "Consistent with earlier findings: the bottleneck is INFORMATION "
              "available in vision+language+proprioception, not decoder "
              "architecture. Further gains likely need more input signal (temporal "
              "history, force/torque sensing) rather than architecture changes.")
    print("=" * 70)

    return baseline_metrics, split_metrics


if __name__ == "__main__":
    main()