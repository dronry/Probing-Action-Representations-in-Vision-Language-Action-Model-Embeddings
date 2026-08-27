"""
Experiment 1c: Isolating the Orientation R2 Ceiling
======================================================
Experiment 1b found: even a nonlinear MLP probe caps orientation
(roll/pitch/yaw) around R2 ~ 0.23-0.35, while position/gripper reach
0.6-0.78. This experiment isolates WHY, via three independent tests
that each rule out a different explanation:

  TEST A — Bigger MLP capacity
      Question: was the 1b probe just too small/undertrained?
      Method: same frozen features, same layers, but a larger MLP
      (wider + deeper + longer training budget). If orientation R2
      does not move meaningfully, capacity was never the bottleneck.

  TEST B — Proprioception fusion
      Question: is the missing signal the robot's OWN current pose?
      Orientation targets (future roll/pitch/yaw) are naturally most
      predictable from CURRENT roll/pitch/yaw + vision/language intent,
      not from vision/language alone. A frozen VLM hidden state is
      conditioned on vision+language only — proprioceptive state may
      never have been "written into" that representation.
      Method: concatenate observation.state (proprioception, already
      saved alongside actions in 1a/1b — re-extracted here if not
      present) to the VLM hidden state before the MLP probe. Compare
      against Test A's MLP (same architecture, extra input dim).

  TEST C — Action chunks vs single-step
      Question: is single-step orientation just noisier / less
      predictable than a smoothed short-horizon target?
      Method: instead of predicting action[t], predict the MEAN of
      action[t : t+K] (a short chunk). This requires re-extracting
      chunk-level ground truth from the ORIGINAL dataset (not stored
      in 1a's features.npz, which only saved single-step actions) —
      this test therefore needs the dataset again, but NOT the model
      (no GPU / no pi0 forward passes needed, pure data reprocessing).

Design principle for all three: reuse Test A's MLP architecture and
training procedure as the shared baseline, changing exactly ONE
variable per test, so gains can be cleanly attributed.

Requires: features.npz from 1a (Tests A, B) AND access to
lerobot/libero_spatial_image via `datasets` (Test C only, for chunk
ground truth — no GPU needed, this is pure streaming + indexing).
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

FEATURES_PATH = "/kaggle/input/datasets/tahak9/boyfla/features.npz"
DATASET_ID    = "lerobot/libero_spatial_image"
SAVE_DIR      = Path("/kaggle/working/probe_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PROBE_LAYERS      = [-1, -2, -3, -4]
ACTION_DIM_NAMES  = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
ORIENTATION_IDXS  = [3, 4, 5]   # roll, pitch, yaw

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Test A: bigger MLP
BIG_HIDDEN      = 512
BIG_DEPTH_DIMS  = [512, 256, 128]   # 3 hidden layers vs 1b's 2
BIG_EPOCHS      = 400               # 2x longer budget
BIG_PATIENCE    = 40

# Shared training hyperparams (same as 1b baseline unless overridden)
DROPOUT      = 0.2
WEIGHT_DECAY = 1e-4
LR           = 1e-3
BATCH_SIZE   = 64

# Test C: chunk size
CHUNK_SIZE = 10   # predict mean of next 10 actions instead of just action[t]

print(f"Device: {DEVICE}\n")


# ──────────────────────────────────────────────
# SHARED: MLP definitions
# ──────────────────────────────────────────────

class SmallMLPProbe(nn.Module):
    """Same as Experiment 1b's probe — the fixed baseline for comparison."""
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    def forward(self, x):
        return self.net(x)


class BigMLPProbe(nn.Module):
    """Test A: larger capacity — 3 hidden layers, wider, more parameters."""
    def __init__(self, input_dim, output_dim, dims=(512, 256, 128), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for d in dims:
            layers += [nn.Linear(prev, d), nn.GELU(), nn.Dropout(dropout)]
            prev = d
        layers += [nn.Linear(prev, output_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


def train_probe(model, X_train, y_train, X_val, y_val, epochs, patience, lr=LR, wd=WEIGHT_DECAY):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32, device=DEVICE)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32, device=DEVICE)

    n = len(X_train_t)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch} (best val loss: {best_val_loss:.6f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


def eval_probe(model, X_test, y_test):
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
        preds = model(X_test_t).cpu().numpy()
    r2_per_dim = r2_score(y_test, preds, multioutput="raw_values")
    r2_overall = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    return r2_overall, r2_per_dim, mse, preds


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


def get_splits(n, actions_array):
    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=SEED)
    subtrain_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=SEED)
    return subtrain_idx, val_idx, test_idx


def orientation_summary(r2_per_dim, dim_names):
    idxs = [i for i, n in enumerate(dim_names) if n in ("roll", "pitch", "yaw")]
    return float(np.mean([r2_per_dim[i] for i in idxs])) if idxs else None


# ──────────────────────────────────────────────
# TEST A: BIGGER MLP CAPACITY
# ──────────────────────────────────────────────

def run_test_a():
    print("=" * 60)
    print("TEST A: Bigger MLP capacity")
    print("=" * 60)

    layer_arrays, actions_array = load_features()
    n = len(actions_array)
    act_dim = actions_array.shape[1]
    dim_names = ACTION_DIM_NAMES[:act_dim]
    subtrain_idx, val_idx, test_idx = get_splits(n, actions_array)

    results = {}
    for key in PROBE_LAYERS + ["avg"]:
        if key not in layer_arrays:
            continue
        X = layer_arrays[key]
        X_subtrain, X_val, X_test = X[subtrain_idx], X[val_idx], X[test_idx]
        y_subtrain, y_val, y_test = (
            actions_array[subtrain_idx], actions_array[val_idx], actions_array[test_idx]
        )

        scaler = StandardScaler()
        X_subtrain_sc = scaler.fit_transform(X_subtrain)
        X_val_sc      = scaler.transform(X_val)
        X_test_sc     = scaler.transform(X_test)

        print(f"\n  Layer {key}: training BIG MLP ({BIG_DEPTH_DIMS})...")
        model = BigMLPProbe(X.shape[1], act_dim, dims=BIG_DEPTH_DIMS, dropout=DROPOUT).to(DEVICE)
        model = train_probe(model, X_subtrain_sc, y_subtrain, X_val_sc, y_val,
                             epochs=BIG_EPOCHS, patience=BIG_PATIENCE)

        r2_overall, r2_per_dim, mse, _ = eval_probe(model, X_test_sc, y_test)
        orient_r2 = orientation_summary(r2_per_dim, dim_names)

        results[key] = {"r2_overall": r2_overall, "r2_per_dim": r2_per_dim,
                         "mse": mse, "orient_r2": orient_r2}
        print(f"    Overall R2: {r2_overall:.4f}   Orientation avg R2: {orient_r2:.4f}")

    return results, dim_names


# ──────────────────────────────────────────────
# TEST B: PROPRIOCEPTION FUSION
# ──────────────────────────────────────────────

def run_test_b():
    """
    Requires observation.state (proprioception) per sample. This was NOT
    saved in 1a's features.npz (only hidden states + actions were saved).
    We re-stream just the proprioception + action columns from the dataset
    (lightweight — no images decoded, no model forward pass, no GPU needed
    for this part) and align by frame index order, which MUST match 1a's
    sampling exactly (same NUM_FRAMES, same linspace-based subsampling
    logic) for this alignment to be valid.
    """
    print("\n" + "=" * 60)
    print("TEST B: Proprioception fusion")
    print("=" * 60)

    layer_arrays, actions_array = load_features()
    n_saved = len(actions_array)

    print(f"\n  Re-streaming proprioception (observation.state) to align with "
          f"{n_saved} saved samples...")
    print("  NOTE: this assumes the SAME sampling parameters as Experiment 1a "
          "(NUM_FRAMES, STREAM_POOL_MULT, same linspace subsampling). If 1a's "
          "config changed between runs, this alignment will be WRONG — verify "
          "the action arrays match before trusting Test B's results.")

    from datasets import load_dataset

    # These MUST match Experiment 1a's config exactly
    NUM_FRAMES       = 1500
    STREAM_POOL_MULT = 4
    pool_size = NUM_FRAMES * STREAM_POOL_MULT

    raw_ds = load_dataset(DATASET_ID, split="train", streaming=True)
    pool = []
    for i, sample in enumerate(raw_ds):
        if i >= pool_size:
            break
        pool.append(sample)

    indices = np.linspace(0, len(pool) - 1, min(NUM_FRAMES, len(pool)), dtype=int)

    proprio_list = []
    action_check_list = []
    for idx in indices:
        s = pool[int(idx)]
        state = s["observation.state"]
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        proprio_list.append(state)

        act = s.get("action", s.get("actions"))
        if not isinstance(act, np.ndarray):
            act = np.array(act, dtype=np.float32)
        if act.ndim > 1:
            act = act[0]
        action_check_list.append(act)

    proprio_array = np.stack(proprio_list, axis=0)
    action_check  = np.stack(action_check_list, axis=0)

    # Sanity check: does the re-streamed action array match the saved one?
    if action_check.shape != actions_array.shape:
        raise RuntimeError(
            f"Alignment check FAILED: re-streamed actions shape {action_check.shape} "
            f"!= saved actions shape {actions_array.shape}. Test B cannot proceed "
            "safely — the sampling parameters likely differ from Experiment 1a. "
            "Fix NUM_FRAMES/STREAM_POOL_MULT above to match 1a's actual config."
        )
    match_frac = np.mean(np.isclose(action_check, actions_array, atol=1e-4))
    print(f"  Alignment check: {match_frac*100:.1f}% of action values match within tolerance")
    if match_frac < 0.95:
        raise RuntimeError(
            f"Alignment check FAILED: only {match_frac*100:.1f}% of re-streamed actions "
            "match the saved features.npz actions. Proprioception would be fused with "
            "the WRONG frames. Do not trust results — fix the sampling alignment first."
        )
    print("  Alignment check PASSED — proceeding with fused features.\n")

    n = len(actions_array)
    act_dim = actions_array.shape[1]
    dim_names = ACTION_DIM_NAMES[:act_dim]
    subtrain_idx, val_idx, test_idx = get_splits(n, actions_array)

    results = {}
    for key in PROBE_LAYERS + ["avg"]:
        if key not in layer_arrays:
            continue

        # Fuse: [VLM hidden state | proprioception]
        X_fused = np.concatenate([layer_arrays[key], proprio_array], axis=1)

        X_subtrain, X_val, X_test = X_fused[subtrain_idx], X_fused[val_idx], X_fused[test_idx]
        y_subtrain, y_val, y_test = (
            actions_array[subtrain_idx], actions_array[val_idx], actions_array[test_idx]
        )

        scaler = StandardScaler()
        X_subtrain_sc = scaler.fit_transform(X_subtrain)
        X_val_sc      = scaler.transform(X_val)
        X_test_sc     = scaler.transform(X_test)

        print(f"  Layer {key} + proprio (dim {X_fused.shape[1]}): training MLP...")
        # Use the SAME architecture as 1b's baseline (SmallMLPProbe) so any
        # gain is attributable to the added proprioception input, not extra
        # capacity (that's Test A's job).
        model = SmallMLPProbe(X_fused.shape[1], act_dim, hidden_dim=256, dropout=DROPOUT).to(DEVICE)
        model = train_probe(model, X_subtrain_sc, y_subtrain, X_val_sc, y_val,
                             epochs=200, patience=20)

        r2_overall, r2_per_dim, mse, _ = eval_probe(model, X_test_sc, y_test)
        orient_r2 = orientation_summary(r2_per_dim, dim_names)

        results[key] = {"r2_overall": r2_overall, "r2_per_dim": r2_per_dim,
                         "mse": mse, "orient_r2": orient_r2}
        print(f"    Overall R2: {r2_overall:.4f}   Orientation avg R2: {orient_r2:.4f}")

    return results, dim_names, proprio_array.shape[1]


# ──────────────────────────────────────────────
# TEST C: ACTION CHUNKS VS SINGLE-STEP
# ──────────────────────────────────────────────

def run_test_c():
    """
    Requires per-episode, per-frame action sequences to build chunk targets
    (mean of action[t:t+K] within the SAME episode — must not cross episode
    boundaries). This needs episode_index and frame_index from the dataset,
    which were not saved in 1a's features.npz. Re-streams the dataset for
    this metadata (again, no GPU needed — this is pure data indexing).
    """
    print("\n" + "=" * 60)
    print(f"TEST C: Action chunks (K={CHUNK_SIZE}) vs single-step")
    print("=" * 60)

    layer_arrays, actions_array = load_features()
    n_saved = len(actions_array)

    print(f"\n  Re-streaming episode/frame metadata + building chunk targets...")
    from datasets import load_dataset

    NUM_FRAMES       = 1500
    STREAM_POOL_MULT = 4
    pool_size = NUM_FRAMES * STREAM_POOL_MULT

    raw_ds = load_dataset(DATASET_ID, split="train", streaming=True)
    pool = []
    for i, sample in enumerate(raw_ds):
        if i >= pool_size:
            break
        pool.append(sample)

    indices = np.linspace(0, len(pool) - 1, min(NUM_FRAMES, len(pool)), dtype=int)

    # Build a full per-episode action lookup so we can pull future actions
    # for chunking without re-streaming the whole dataset again per episode.
    episode_actions = {}   # episode_index -> {frame_index: action}
    for s in pool:
        ep = int(s["episode_index"])
        fr = int(s["frame_index"])
        act = s.get("action", s.get("actions"))
        if not isinstance(act, np.ndarray):
            act = np.array(act, dtype=np.float32)
        if act.ndim > 1:
            act = act[0]
        episode_actions.setdefault(ep, {})[fr] = act

    chunk_actions = []
    action_check_list = []
    valid_mask = []

    for idx in indices:
        s = pool[int(idx)]
        ep = int(s["episode_index"])
        fr = int(s["frame_index"])

        single_act = episode_actions[ep][fr]
        action_check_list.append(single_act)

        # Gather up to CHUNK_SIZE future actions within the SAME episode
        chunk = []
        for k in range(CHUNK_SIZE):
            if (fr + k) in episode_actions[ep]:
                chunk.append(episode_actions[ep][fr + k])
        if len(chunk) == 0:
            valid_mask.append(False)
            chunk_actions.append(single_act)   # placeholder, will be masked out
        else:
            valid_mask.append(True)
            chunk_actions.append(np.mean(np.stack(chunk, axis=0), axis=0))

    chunk_actions = np.stack(chunk_actions, axis=0)
    action_check  = np.stack(action_check_list, axis=0)
    valid_mask    = np.array(valid_mask)

    if action_check.shape != actions_array.shape:
        raise RuntimeError(
            f"Alignment check FAILED: re-streamed actions shape {action_check.shape} "
            f"!= saved actions shape {actions_array.shape}. Fix sampling params."
        )
    match_frac = np.mean(np.isclose(action_check, actions_array, atol=1e-4))
    print(f"  Alignment check: {match_frac*100:.1f}% match")
    if match_frac < 0.95:
        raise RuntimeError(
            f"Alignment check FAILED ({match_frac*100:.1f}% match) — do not trust "
            "Test C results. Fix sampling alignment first."
        )
    print(f"  Valid chunk targets (had >=1 future frame in same episode): "
          f"{valid_mask.sum()}/{len(valid_mask)}")
    print("  Alignment check PASSED — proceeding with chunk targets.\n")

    # Restrict to valid frames only (end-of-episode frames get dropped)
    keep = np.where(valid_mask)[0]
    layer_arrays_v = {k: v[keep] for k, v in layer_arrays.items()}
    chunk_actions_v = chunk_actions[keep]

    n = len(chunk_actions_v)
    act_dim = chunk_actions_v.shape[1]
    dim_names = ACTION_DIM_NAMES[:act_dim]
    subtrain_idx, val_idx, test_idx = get_splits(n, chunk_actions_v)

    results = {}
    for key in PROBE_LAYERS + ["avg"]:
        if key not in layer_arrays_v:
            continue
        X = layer_arrays_v[key]
        X_subtrain, X_val, X_test = X[subtrain_idx], X[val_idx], X[test_idx]
        y_subtrain, y_val, y_test = (
            chunk_actions_v[subtrain_idx], chunk_actions_v[val_idx], chunk_actions_v[test_idx]
        )

        scaler = StandardScaler()
        X_subtrain_sc = scaler.fit_transform(X_subtrain)
        X_val_sc      = scaler.transform(X_val)
        X_test_sc     = scaler.transform(X_test)

        print(f"  Layer {key}: training MLP on chunk targets...")
        model = SmallMLPProbe(X.shape[1], act_dim, hidden_dim=256, dropout=DROPOUT).to(DEVICE)
        model = train_probe(model, X_subtrain_sc, y_subtrain, X_val_sc, y_val,
                             epochs=200, patience=20)

        r2_overall, r2_per_dim, mse, _ = eval_probe(model, X_test_sc, y_test)
        orient_r2 = orientation_summary(r2_per_dim, dim_names)

        results[key] = {"r2_overall": r2_overall, "r2_per_dim": r2_per_dim,
                         "mse": mse, "orient_r2": orient_r2}
        print(f"    Overall R2: {r2_overall:.4f}   Orientation avg R2: {orient_r2:.4f}")

    return results, dim_names


# ──────────────────────────────────────────────
# COMPARISON PLOT + SUMMARY
# ──────────────────────────────────────────────

def plot_all_tests(baseline_1b_orient, test_a, test_b, test_c, dim_names):
    """
    baseline_1b_orient: dict {layer_key: orient_r2} from Experiment 1b's
    small-MLP results, hardcoded from the known 1b output for comparison
    (since 1b's raw per-layer results aren't repersisted as .npz — only
    text/plot — we pass the known numbers in directly).
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("Experiment 1c: Isolating the Orientation R2 Ceiling",
                  fontsize=13, fontweight="bold")

    layer_keys = [k for k in PROBE_LAYERS + ["avg"] if k in test_a[0]]
    labels = [str(k) for k in layer_keys]

    ax = axes[0]
    x = np.arange(len(labels))
    width = 0.2

    b_vals = [baseline_1b_orient.get(k, np.nan) for k in layer_keys]
    a_vals = [test_a[0][k]["orient_r2"] for k in layer_keys]
    p_vals = [test_b[0][k]["orient_r2"] if test_b else np.nan for k in layer_keys]
    c_vals = [test_c[0][k]["orient_r2"] if test_c else np.nan for k in layer_keys]

    ax.bar(x - 1.5*width, b_vals, width, label="1b: Small MLP (baseline)", color="#95a5a6")
    ax.bar(x - 0.5*width, a_vals, width, label="A: Big MLP", color="#3498db")
    if test_b:
        ax.bar(x + 0.5*width, p_vals, width, label="B: + Proprioception", color="#2ecc71")
    if test_c:
        ax.bar(x + 1.5*width, c_vals, width, label="C: Action chunks", color="#e67e22")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Orientation avg R2 (roll/pitch/yaw)")
    ax.set_title("Orientation R2 by Test, per Layer")
    ax.axhline(0.4, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    best_a = max(layer_keys, key=lambda k: test_a[0][k]["orient_r2"])
    ax.bar(dim_names[3:6], test_a[0][best_a]["r2_per_dim"][3:6],
           width=0.6, color="#3498db", label=f"Best layer ({best_a})")
    ax.axhline(0.0, color="black", alpha=0.3)
    ax.set_ylabel("R2")
    ax.set_title("Test A best-layer orientation breakdown")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    save_path = SAVE_DIR / "probe_1c_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved -> {save_path}")


def save_1c_summary(baseline_1b_orient, test_a, test_b, test_c, dim_names):
    path = SAVE_DIR / "probe_1c_summary.txt"
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("EXPERIMENT 1c: ISOLATING THE ORIENTATION R2 CEILING\n")
        f.write("=" * 60 + "\n\n")

        f.write("Orientation avg R2 (roll/pitch/yaw) by layer:\n\n")
        f.write(f"{'Layer':<10} {'1b (small MLP)':<16} {'A (big MLP)':<14} "
                f"{'B (+proprio)':<14} {'C (chunks)':<12}\n")
        for k in PROBE_LAYERS + ["avg"]:
            if k not in test_a[0]:
                continue
            b = baseline_1b_orient.get(k, float("nan"))
            a = test_a[0][k]["orient_r2"]
            p = test_b[0][k]["orient_r2"] if test_b else float("nan")
            c = test_c[0][k]["orient_r2"] if test_c else float("nan")
            f.write(f"{str(k):<10} {b:<16.4f} {a:<14.4f} {p:<14.4f} {c:<12.4f}\n")

        f.write("\n\nInterpretation:\n")
        f.write("  A >> 1b  -> capacity was the bottleneck (unlikely based on typical results)\n")
        f.write("  B >> 1b  -> proprioception was the missing signal\n")
        f.write("  C >> 1b  -> single-step targets were just noisier than chunk-level\n")
        f.write("  none >> 1b -> orientation may need temporal context beyond a single\n")
        f.write("                frame + proprioception (e.g. short observation history),\n")
        f.write("                or is inherently harder to recover from this VLM's\n")
        f.write("                pretrained representations regardless of decoder design.\n")

    print(f"Summary saved -> {path}")


def main():
    # Known baseline orientation R2 from Experiment 1b (small MLP), for
    # direct comparison — hardcoded from the actual 1b output you already have.
    baseline_1b_orient = {
        -1: np.mean([0.2331, 0.3100, 0.3485]),
        -2: np.mean([0.2334, 0.3263, 0.2147]),
        -3: np.mean([0.2502, 0.3140, 0.2259]),
        -4: np.mean([0.2497, 0.2746, 0.3444]),
        "avg": np.mean([0.1513, 0.1628, 0.2951]),
    }

    print("Experiment 1c: running Tests A, B, C\n")

    test_a = run_test_a()

    try:
        test_b = run_test_b()
    except Exception as e:
        print(f"\n  Test B FAILED: {e}")
        print("  Skipping Test B — check dataset streaming / alignment.")
        test_b = None

    try:
        test_c = run_test_c()
    except Exception as e:
        print(f"\n  Test C FAILED: {e}")
        print("  Skipping Test C — check dataset streaming / alignment.")
        test_c = None

    dim_names = test_a[1]
    plot_all_tests(baseline_1b_orient, test_a, test_b, test_c, dim_names)
    save_1c_summary(baseline_1b_orient, test_a, test_b, test_c, dim_names)

    # ── Final attribution ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL ATTRIBUTION")
    print("=" * 60)

    best_layer_a = max(test_a[0], key=lambda k: test_a[0][k]["orient_r2"])
    a_gain = test_a[0][best_layer_a]["orient_r2"] - baseline_1b_orient.get(best_layer_a, 0)
    print(f"\nTest A (capacity) best gain: {a_gain:+.4f} at layer {best_layer_a}")

    if test_b:
        best_layer_b = max(test_b[0], key=lambda k: test_b[0][k]["orient_r2"])
        b_gain = test_b[0][best_layer_b]["orient_r2"] - baseline_1b_orient.get(best_layer_b, 0)
        print(f"Test B (proprioception) best gain: {b_gain:+.4f} at layer {best_layer_b}")
    else:
        b_gain = None

    if test_c:
        best_layer_c = max(test_c[0], key=lambda k: test_c[0][k]["orient_r2"])
        c_gain = test_c[0][best_layer_c]["orient_r2"] - baseline_1b_orient.get(best_layer_c, 0)
        print(f"Test C (chunking) best gain: {c_gain:+.4f} at layer {best_layer_c}")
    else:
        c_gain = None

    print()
    gains = {"A (capacity)": a_gain, "B (proprioception)": b_gain, "C (chunking)": c_gain}
    valid_gains = {k: v for k, v in gains.items() if v is not None}
    if valid_gains:
        best_cause = max(valid_gains, key=valid_gains.get)
        print(f"Largest single gain: {best_cause} ({valid_gains[best_cause]:+.4f})")
        if valid_gains[best_cause] > 0.15:
            print(f"  -> {best_cause} is the primary lever for improving orientation decoding.")
        else:
            print("  -> No single factor produces a large gain. Orientation may need "
                  "multiple factors combined (proprio + chunking + capacity), or may "
                  "reflect a more fundamental limit of this VLM's representation for "
                  "wrist rotation specifically.")

    print("=" * 60)


if __name__ == "__main__":
    main()