"""
Experiment 3: Flow Trajectory Quality -- Embedding Space vs Raw Action Space
================================================================================
PURPOSE
----------
Experiments 1 and 2 tested whether the VLM embedding space CONTAINS
action-relevant structure (yes, for position/gripper) and whether newly
constructed action embeddings NATURALLY ORGANIZE near real language
tokens (no -- Experiment 2 found degenerate collapse).

This experiment tests a THIRD, independent question: even setting aside
whether embeddings land near real tokens, does running a rectified flow
THROUGH the VLM embedding space produce geometrically simpler (straighter,
lower-curvature) trajectories than running the same flow in raw action
space? Straight trajectories are rectified flow's core selling point.

WHY A SYNTHETIC 2D TASK, NOT LIBERO
---------------------------------------
This is about TRAJECTORY GEOMETRY, not task success:
  - RAW ACTION SPACE condition: targets are 2D goal points directly.
  - RANDOM PROJECTION condition (control): those same 2D goals mapped
    into 2048-dim space via a FIXED, isometric random projection.
    Isolates whether HIGH DIMENSIONALITY ALONE changes flow geometry,
    independent of any real semantic structure.
  - REAL VOCABULARY MANIFOLD condition (optional): targets placed near
    actual PaliGemma vocabulary embeddings (same extraction validated
    in Experiment 2). Tests whether REAL trained language geometry
    specifically helps beyond dimensionality alone.

All three conditions train an IDENTICAL rectified-flow architecture,
differing only in target space. We measure path curvature directly and
sampling efficiency (error vs number of Euler steps).

INTERPRETATION
------------------
If random projection ALSO straightens trajectories like the real vocab
condition: dimensionality is doing the work, not semantics -- the
manifold argument gets WEAKER on this axis.

If real vocab straightens trajectories MORE than random projection:
real language geometry specifically helps -- the manifold argument
gets STRONGER on this axis.

If neither straightens trajectories vs raw space: the whole geometric
efficiency motivation needs reconsideration independent of VLMs.

Requires: PyTorch (CPU fine for the raw + random-projection conditions;
GPU + pi0 checkpoint access needed for the optional real-vocab condition).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SAVE_DIR = Path("/kaggle/working/probe_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RUN_REAL_VOCAB_CONDITION = True
MODEL_ID = "lerobot/pi0_base"

N_SAMPLES = 2000
EMBED_DIM = 2048
RAW_DIM = 2

FLOW_HIDDEN = 256
FLOW_DEPTH = 4
FLOW_EPOCHS = 500
FLOW_LR = 1e-3
FLOW_BATCH = 128
FLOW_PATIENCE = 40

EULER_STEPS_TO_TEST = [1, 2, 4, 8, 16, 32, 64]

print(f"Device: {DEVICE}\n")


def make_synthetic_task(n_samples=N_SAMPLES, seed=SEED):
    rng = np.random.RandomState(seed)
    n_modes = 4
    mode_centers = rng.uniform(-3, 3, size=(n_modes, RAW_DIM))
    mode_assignments = rng.randint(0, n_modes, size=n_samples)
    noise = rng.randn(n_samples, RAW_DIM) * 0.3
    targets_2d = mode_centers[mode_assignments] + noise
    sources_2d = rng.randn(n_samples, RAW_DIM)
    return sources_2d.astype(np.float32), targets_2d.astype(np.float32), mode_assignments


def embed_via_random_projection(targets_2d, embed_dim=EMBED_DIM, seed=SEED):
    rng = np.random.RandomState(seed)
    random_matrix = rng.randn(embed_dim, RAW_DIM)
    q, _ = np.linalg.qr(random_matrix)
    projection = q[:, :RAW_DIM]
    embedded = targets_2d @ projection.T
    return embedded.astype(np.float32), projection


def embed_via_real_vocab(targets_2d, mode_assignments, model_id=MODEL_ID, seed=SEED):
    vocab_embeddings = _extract_vocab_embeddings(model_id)
    n_modes = mode_assignments.max() + 1
    rng = np.random.RandomState(seed)
    chosen_token_idxs = rng.choice(len(vocab_embeddings), size=n_modes, replace=False)
    mode_anchor_embeddings = vocab_embeddings[chosen_token_idxs]
    anchor_norm_scale = np.linalg.norm(mode_anchor_embeddings, axis=1).mean()
    local_noise_scale = 0.05 * anchor_norm_scale
    embedded = mode_anchor_embeddings[mode_assignments] + \
               rng.randn(len(targets_2d), vocab_embeddings.shape[1]).astype(np.float32) * local_noise_scale
    return embedded.astype(np.float32), mode_anchor_embeddings


def _extract_vocab_embeddings(model_id):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import glob

    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        hf_token = os.environ.get("HF_TOKEN", "")

    print("  Extracting real vocabulary embeddings for condition 3b...")
    ckpt_dir = snapshot_download(model_id, token=hf_token or None,
                                   ignore_patterns=["*.msgpack", "*.h5", "flax_model*"])
    shards = sorted(glob.glob(f"{ckpt_dir}/*.safetensors")) or \
             sorted(glob.glob(f"{ckpt_dir}/**/*.safetensors", recursive=True))

    EXCLUDE = "gemma_expert"
    all_keys = []
    for shard_path in shards:
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for key in shard.keys():
                all_keys.append(key)

    candidates = [k for k in all_keys
                  if ("embed_tokens.weight" in k or "lm_head.weight" in k)
                  and EXCLUDE not in k]
    if not candidates:
        raise RuntimeError("No embedding/lm_head candidates found (excluding gemma_expert).")

    target_key = candidates[0]
    print(f"  Using key: '{target_key}' (matches Experiment 2's validated approach)")

    for shard_path in shards:
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            if target_key in shard.keys():
                tensor = shard.get_tensor(target_key).float().numpy()
                if tensor.shape[1] != EMBED_DIM:
                    raise RuntimeError(f"Extracted table has dim {tensor.shape[1]}, expected {EMBED_DIM}.")
                return tensor

    raise RuntimeError(f"Key '{target_key}' found in listing but not in re-scan.")


class RectifiedFlowNet(nn.Module):
    def __init__(self, dim, hidden=256, depth=4):
        super().__init__()
        layers = [nn.Linear(dim + 1, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        t_expand = t.view(-1, 1).expand(x.shape[0], 1)
        inp = torch.cat([x, t_expand], dim=-1)
        return self.net(inp)


def train_rectified_flow(x0, x1, dim, epochs=FLOW_EPOCHS, patience=FLOW_PATIENCE):
    model = RectifiedFlowNet(dim, FLOW_HIDDEN, FLOW_DEPTH).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FLOW_LR)
    loss_fn = nn.MSELoss()

    x0_t = torch.tensor(x0, dtype=torch.float32, device=DEVICE)
    x1_t = torch.tensor(x1, dtype=torch.float32, device=DEVICE)

    n = len(x0_t)
    n_train = int(n * 0.85)
    train_idx = torch.arange(n_train)
    val_idx = torch.arange(n_train, n)

    best_val_loss, best_state, patience_counter = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        for i in range(0, len(perm), FLOW_BATCH):
            batch_idx = perm[i:i + FLOW_BATCH]
            x0_b = x0_t[batch_idx]
            x1_b = x1_t[batch_idx]
            t = torch.rand(len(batch_idx), device=DEVICE)

            xt = (1 - t.view(-1, 1)) * x0_b + t.view(-1, 1) * x1_b
            target_velocity = x1_b - x0_b

            optimizer.zero_grad()
            pred_velocity = model(xt, t)
            loss = loss_fn(pred_velocity, target_velocity)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            x0_v, x1_v = x0_t[val_idx], x1_t[val_idx]
            t_v = torch.rand(len(val_idx), device=DEVICE)
            xt_v = (1 - t_v.view(-1, 1)) * x0_v + t_v.view(-1, 1) * x1_v
            target_v = x1_v - x0_v
            val_loss = loss_fn(model(xt_v, t_v), target_v).item()

        if val_loss < best_val_loss:
            best_val_loss, best_state, patience_counter = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch} (best val loss: {best_val_loss:.6f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def integrate_trajectory(model, x0, n_steps, dim):
    x0_t = torch.tensor(x0, dtype=torch.float32, device=DEVICE)
    dt = 1.0 / n_steps
    x = x0_t.clone()
    trajectory = [x.cpu().numpy().copy()]

    for step in range(n_steps):
        t = torch.full((x.shape[0],), step * dt, device=DEVICE)
        v = model(x, t)
        x = x + v * dt
        trajectory.append(x.cpu().numpy().copy())

    return np.stack(trajectory, axis=0)


def compute_path_curvature(trajectory):
    straight_line_dist = np.linalg.norm(trajectory[-1] - trajectory[0], axis=-1)
    segment_lengths = np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=-1)
    path_length = segment_lengths.sum(axis=0)
    curvature_ratio = path_length / np.maximum(straight_line_dist, 1e-6)
    return curvature_ratio, path_length, straight_line_dist


def compute_endpoint_error_vs_steps(model, x0, x1_true, dim, steps_list=EULER_STEPS_TO_TEST):
    errors = []
    x1_true_t = torch.tensor(x1_true, dtype=torch.float32, device=DEVICE)
    for n_steps in steps_list:
        trajectory = integrate_trajectory(model, x0, n_steps, dim)
        final = torch.tensor(trajectory[-1], dtype=torch.float32, device=DEVICE)
        mse = torch.mean((final - x1_true_t) ** 2).item()
        errors.append(mse)
    return errors


def run_condition(name, x0, x1, dim):
    print(f"\n{'='*70}")
    print(f"CONDITION: {name}  (dim={dim})")
    print(f"{'='*70}")

    print("Training rectified flow...")
    model = train_rectified_flow(x0, x1, dim)

    print("\nMeasuring path curvature (64-step high-resolution integration)...")
    trajectory = integrate_trajectory(model, x0, n_steps=64, dim=dim)
    curvature_ratio, path_length, straight_dist = compute_path_curvature(trajectory)

    print(f"  Mean curvature ratio (path_length / straight_line_dist): {curvature_ratio.mean():.4f}")
    print(f"  Median curvature ratio: {np.median(curvature_ratio):.4f}")
    print(f"  (1.0 = perfectly straight; higher = more curved)")

    print("\nMeasuring endpoint error vs number of integration steps...")
    errors = compute_endpoint_error_vs_steps(model, x0, x1, dim)
    for n_steps, err in zip(EULER_STEPS_TO_TEST, errors):
        print(f"  {n_steps:3d} steps: MSE = {err:.6f}")

    return {
        "name": name, "model": model, "trajectory": trajectory,
        "curvature_ratio": curvature_ratio, "path_length": path_length,
        "straight_dist": straight_dist, "step_errors": errors,
    }


def plot_comparison(results_raw, results_random_proj, results_real_vocab=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Experiment 3: Flow Trajectory Quality Across Target Spaces",
                  fontsize=13, fontweight="bold")

    all_results = [results_raw, results_random_proj] + ([results_real_vocab] if results_real_vocab else [])
    colors = ["#3498db", "#e67e22", "#2ecc71"]

    ax = axes[0]
    for r, c in zip(all_results, colors):
        ax.hist(r["curvature_ratio"], bins=40, alpha=0.5, color=c, label=r["name"])
    ax.axvline(1.0, color="black", linestyle="--", alpha=0.5, label="Perfectly straight")
    ax.set_xlabel("Curvature ratio (path length / straight-line distance)")
    ax.set_ylabel("Count")
    ax.set_title("Path Curvature Distribution")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for r, c in zip(all_results, colors):
        ax.plot(EULER_STEPS_TO_TEST, r["step_errors"], marker="o", color=c, label=r["name"])
    ax.set_xlabel("Number of Euler integration steps")
    ax.set_ylabel("Endpoint MSE")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title("Sampling Efficiency: Error vs Steps")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    names = [r["name"] for r in all_results]
    means = [r["curvature_ratio"].mean() for r in all_results]
    ax.bar(names, means, color=colors[:len(all_results)], edgecolor="white")
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_ylabel("Mean curvature ratio")
    ax.set_title("Mean Curvature by Condition\n(lower = straighter = better)")
    ax.tick_params(axis="x", rotation=15)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    save_path = SAVE_DIR / "experiment3_flow_trajectory_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved -> {save_path}")


def plot_2d_trajectories_sample(results_raw, sources_2d, n_show=30):
    trajectory = results_raw["trajectory"]
    fig, ax = plt.subplots(figsize=(7, 7))

    idxs = np.random.RandomState(SEED).choice(trajectory.shape[1], n_show, replace=False)
    for idx in idxs:
        path = trajectory[:, idx, :]
        ax.plot(path[:, 0], path[:, 1], alpha=0.4, color="#3498db", linewidth=1)
        ax.scatter(*path[0], color="gray", s=15, zorder=3)
        ax.scatter(*path[-1], color="#e74c3c", s=15, zorder=3)

    ax.scatter([], [], color="gray", label="Start (noise)")
    ax.scatter([], [], color="#e74c3c", label="End (target)")
    ax.set_title(f"Sample Trajectories: Raw 2D Action Space\n(n={n_show} paths shown)")
    ax.legend()
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    plt.tight_layout()
    save_path = SAVE_DIR / "experiment3_raw2d_trajectories.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"2D trajectory visualization saved -> {save_path}")


def main():
    print("=" * 70)
    print("Experiment 3: Flow Trajectory Quality")
    print("Embedding Space vs Raw Action Space")
    print("=" * 70)

    sources_2d, targets_2d, mode_assignments = make_synthetic_task()
    print(f"\nSynthetic task: {N_SAMPLES} samples, {RAW_DIM}D, 4-mode target distribution")

    rng = np.random.RandomState(SEED)
    noise_highdim = rng.randn(N_SAMPLES, EMBED_DIM).astype(np.float32)

    results_raw = run_condition("Raw 2D action space", sources_2d, targets_2d, RAW_DIM)

    targets_random_proj, _ = embed_via_random_projection(targets_2d)
    results_random_proj = run_condition(
        "Random projection (2048D control)", noise_highdim, targets_random_proj, EMBED_DIM
    )

    results_real_vocab = None
    if RUN_REAL_VOCAB_CONDITION:
        try:
            targets_real_vocab, _ = embed_via_real_vocab(targets_2d, mode_assignments)
            results_real_vocab = run_condition(
                "Real PaliGemma vocab manifold", noise_highdim, targets_real_vocab, EMBED_DIM
            )
        except Exception as e:
            print(f"\n  Condition 3b (real vocab) FAILED: {type(e).__name__}: {e}")
            print("  Continuing with conditions 1 and 2 only.")

    plot_comparison(results_raw, results_random_proj, results_real_vocab)
    plot_2d_trajectories_sample(results_raw, sources_2d)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    raw_curv = results_raw["curvature_ratio"].mean()
    rand_curv = results_random_proj["curvature_ratio"].mean()
    print(f"\nMean curvature -- Raw 2D: {raw_curv:.4f}  |  Random projection (2048D): {rand_curv:.4f}")

    if results_real_vocab:
        vocab_curv = results_real_vocab["curvature_ratio"].mean()
        print(f"Mean curvature -- Real vocab manifold: {vocab_curv:.4f}")

        dim_effect = rand_curv - raw_curv
        semantic_effect = vocab_curv - rand_curv

        print(f"\nDimensionality effect (random proj vs raw): {dim_effect:+.4f}")
        print(f"Semantic-structure effect (real vocab vs random proj): {semantic_effect:+.4f}")

        threshold = 0.05 * abs(dim_effect) if dim_effect != 0 else 0.02
        if abs(semantic_effect) < threshold:
            print("\nReal vocabulary geometry produces NO meaningful difference beyond what "
                  "dimensionality alone explains. Any straightening/curving effect is a "
                  "DIMENSIONALITY artifact, not evidence of privileged semantic structure. "
                  "This weakens the manifold hypothesis on the trajectory-quality axis.")
        elif semantic_effect < -0.05:
            print("\nReal vocabulary geometry produces MEANINGFULLY STRAIGHTER trajectories "
                  "than random projection at the same dimensionality. Evidence FOR the "
                  "privileged manifold hypothesis on the trajectory-quality axis -- real "
                  "language structure specifically helps, not just high dimensionality.")
        else:
            print("\nReal vocabulary geometry produces MEANINGFULLY MORE CURVED trajectories "
                  "than random projection. Real language token geometry may be a HARDER "
                  "target manifold to flow through than an arbitrary high-dim space -- "
                  "argues AGAINST routing actions through it for efficiency reasons.")
    else:
        print("\n(Condition 3b was skipped or failed -- only dimensionality effect measured.)")
        if rand_curv < raw_curv - 0.05:
            print("Higher dimensionality alone straightens trajectories in this synthetic task.")
        else:
            print("No strong dimensionality effect observed in this synthetic task.")

    print("=" * 70)

    return results_raw, results_random_proj, results_real_vocab


if __name__ == "__main__":
    main()