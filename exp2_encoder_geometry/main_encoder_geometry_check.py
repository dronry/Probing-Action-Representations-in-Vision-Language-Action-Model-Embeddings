"""
Experiment 2: Encoder Geometry Check
========================================
Everything in Experiment 1 (1a-1e) tested whether FROZEN VLM hidden
states already contain action-relevant structure. This experiment
tests something different: if we TRAIN an encoder E_phi that maps
actions into the VLM's embedding space (the thing that would define
flow-matching targets during training), does it actually land near
real, semantically meaningful language token embeddings -- or does it
drift into an arbitrary, unused corner of the space that happens to be
high-dimensional and easy to fit into?

This matters because the "privileged manifold" argument only holds if
the flow's targets are geometrically anchored to the VLM's actual
pretrained structure. If E_phi can achieve perfect reconstruction while
landing nowhere near real token embeddings, then using this embedding
space bought nothing over training a flow in an arbitrary learned
latent space.

TWO CONDITIONS
-----------------
(a) PLAIN: E_phi trained with reconstruction loss only.
(b) REGULARIZED: E_phi trained with reconstruction loss PLUS a cosine
    proximity term pulling E_phi(action) toward its nearest real
    vocabulary token embedding (extracted from the frozen pi0 model's
    checkpoint).

METRICS
----------
1. Nearest-neighbor cosine distance to real vocab embeddings.
2. Reconstruction quality (R2/MSE).
3. PCA visualization of encoded actions vs a sample of real vocab tokens.
4. Density check: how many unique vocab tokens are the nearest neighbor.

Requires: features.npz (for actions), GPU access to pull the embedding
table from pi0's checkpoint. Extraction reads only the safetensors
shards directly -- it never constructs the full PI0Policy, so it does
not require GPU memory at all (this was the root cause of the prior
OOM: the old code's exact-key match failed silently and fell back to
building the entire policy and moving it to the GPU, which a T4 cannot
hold alongside inference/training memory for the rest of the script).

KEY-DISCOVERY FIX
-----------------
The previous version hardcoded one exact dotted key path taken from a
single prior run's log and treated any mismatch as fatal-but-recoverable
(fall back to full model). That was wrong on two counts:
  1. It assumed key naming is stable across lerobot/HF cache versions,
     which the actual failure (zero keys containing "embed_tokens" at
     all) shows is false for this checkpoint snapshot.
  2. Its fallback path (full policy construction + .to(cuda)) is not a
     safe recovery -- it's a much heavier operation that predictably
     OOMs on a T4, so "falling back" just traded a clean, fast failure
     for a slow, expensive crash.

This version instead does real key discovery: it dumps all safetensors
keys once, searches with a broader set of candidate substrings
(embed_tokens, tok_embeddings, wte, embed_tokens_weight, shared.weight),
excludes gemma_expert, and picks the candidate whose tensor shape has
hidden dim 2048 (PaliGemma's confirmed dim from Experiments 1a-1d). If
nothing matches, it fails loudly with the full key dump so the actual
naming can be inspected by hand -- no silent fallback to a GPU-heavy
path.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

MODEL_ID = "lerobot/pi0_base"
FEATURES_PATH = "/kaggle/input/datasets/tahak9/boyfla/features.npz"
SAVE_DIR = Path("/kaggle/working/probe_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DTYPE = torch.bfloat16
ACTION_DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

EMB_DIM = 2048
ENCODER_HIDDEN = 256
RECON_EPOCHS = 300
RECON_PATIENCE = 30
BATCH_SIZE = 64
LR = 1e-3
WD = 1e-4
COSINE_REG_WEIGHT = 0.3

EXPECTED_HIDDEN_DIM = 2048   # PaliGemma's confirmed dim from Experiments 1a-1d
EXCLUDE_SUBSTRING = "gemma_expert"
# Broader set of candidate substrings -- do NOT assume one exact dotted
# path is stable across checkpoint snapshots/versions. Ordered roughly
# by how likely they are to be the real language-model embedding table.
CANDIDATE_SUBSTRINGS = [
    "language_model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "embed_tokens.weight",
    "tok_embeddings.weight",
    "wte.weight",
    "shared.weight",
    # lerobot/pi0_base does NOT ship a separate embed_tokens.weight tensor
    # -- confirmed bug/behavior, see huggingface/lerobot#1529 and #2151.
    # Gemma ties input embeddings to the output projection, so lm_head.weight
    # (paligemma's own, NOT gemma_expert's) IS the embedding table, same
    # shape [vocab_size, hidden_dim]. This must stay LAST in priority order
    # since a real embed_tokens key should always be preferred if present.
    "lm_head.weight",
]

try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

print(f"Device: {DEVICE}")
print(f"HF_TOKEN: {'loaded' if HF_TOKEN else 'NOT SET'}\n")


def _dump_all_keys(shards):
    """Read every key from every shard, without loading tensors, so we
    can inspect actual naming before guessing at a match."""
    from safetensors import safe_open
    key_to_shard = {}
    for shard_path in shards:
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for key in shard.keys():
                key_to_shard[key] = shard_path
    return key_to_shard


def extract_vocab_embeddings():
    """
    Extracts PaliGemma's real language-token embedding table directly
    from the safetensors shards -- no full model construction, so this
    never touches the GPU and cannot OOM.

    pi0 has TWO Gemma-style sub-networks: the main PaliGemma VLM
    (hidden dim 2048, confirmed in every Experiment 1 run) and a
    separate `gemma_expert` used for flow-matching action generation
    (hidden dim 1024). We must get PaliGemma's table, not the expert's.
    """
    print("=" * 60)
    print("STEP 0: Extracting vocabulary embedding table")
    print("=" * 60)

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import glob

    ckpt_dir = snapshot_download(MODEL_ID, token=HF_TOKEN or None,
                                   ignore_patterns=["*.msgpack", "*.h5", "flax_model*"])
    shards = sorted(glob.glob(f"{ckpt_dir}/*.safetensors")) or \
             sorted(glob.glob(f"{ckpt_dir}/**/*.safetensors", recursive=True))
    if not shards:
        raise RuntimeError(f"No safetensors found in {ckpt_dir}")

    print("  Dumping all keys from all shards for real key discovery "
          "(no guessed exact path)...")
    key_to_shard = _dump_all_keys(shards)
    all_keys = list(key_to_shard.keys())
    print(f"  Found {len(all_keys)} total keys across {len(shards)} shard(s).")

    # Build the candidate list: any key matching one of CANDIDATE_SUBSTRINGS,
    # excluding the action-expert network, preserving substring priority
    # order so the most specific/likely match is tried first.
    candidates = []
    for substr in CANDIDATE_SUBSTRINGS:
        for key in all_keys:
            if EXCLUDE_SUBSTRING in key:
                continue
            if substr in key and key not in candidates:
                candidates.append(key)

    if not candidates:
        # Fail loudly, but show USEFUL keys, not an arbitrary alphabetical
        # slice. all_keys[:50] is misleading here: gemma_expert.* sorts
        # before paligemma_with_expert.paligemma.* alphabetically, so a
        # blind prefix slice never reaches the keys that actually matter.
        non_expert_keys = [k for k in all_keys if EXCLUDE_SUBSTRING not in k]
        relevant = [k for k in all_keys if "embed" in k.lower() or "lm_head" in k.lower()]
        top_level_groups = sorted({".".join(k.split(".")[:3]) for k in non_expert_keys})

        raise RuntimeError(
            "Could not find any embedding-table candidate key.\n"
            f"Searched substrings: {CANDIDATE_SUBSTRINGS}\n"
            f"Excluded substring: '{EXCLUDE_SUBSTRING}'\n"
            f"Total keys found: {len(all_keys)}. Non-expert keys: {len(non_expert_keys)}.\n\n"
            f"Top-level non-expert key groups ({len(top_level_groups)}):\n"
            + "\n".join(top_level_groups[:80]) +
            f"\n\nKeys containing 'embed' or 'lm_head' anywhere (expert or not) -- "
            f"{len(relevant)} found:\n"
            + ("\n".join(relevant) if relevant else "  (none)") +
            "\n\nUpdate CANDIDATE_SUBSTRINGS with the real naming and rerun. "
            "Do not fall back to full model construction -- it will OOM on "
            "a T4 and does not fix the underlying key-naming mismatch."
        )

    print(f"  Candidate keys found (priority order): {candidates[:5]}"
          f"{' ...' if len(candidates) > 5 else ''}")

    # Try each candidate in priority order, checking hidden dim, until
    # one passes the sanity check. This handles the case where a lower-
    # priority substring (e.g. "shared.weight") matches something with
    # the wrong shape before a correct one is found.
    tensor_np = None
    matched_key = None
    for key in candidates:
        shard_path = key_to_shard[key]
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            tensor = shard.get_tensor(key)
        candidate_shape = tuple(tensor.shape)
        print(f"  Checking '{key}' -> shape {candidate_shape}")
        if len(candidate_shape) == 2 and candidate_shape[1] == EXPECTED_HIDDEN_DIM:
            tensor_np = tensor.float().numpy()
            matched_key = key
            break
        else:
            print(f"    Shape mismatch (expected hidden dim {EXPECTED_HIDDEN_DIM}), "
                  f"trying next candidate...")

    if tensor_np is None:
        raise RuntimeError(
            f"None of the {len(candidates)} candidate keys had hidden dim "
            f"{EXPECTED_HIDDEN_DIM}. Candidates tried: {candidates}\n"
            "Inspect shapes manually and update EXPECTED_HIDDEN_DIM or "
            "CANDIDATE_SUBSTRINGS if PaliGemma's dim differs in this checkpoint."
        )

    print(f"  MATCHED: '{matched_key}'")
    if "lm_head" in matched_key:
        print("  NOTE: matched via lm_head.weight, not a dedicated embed_tokens tensor. "
              "This checkpoint (lerobot/pi0_base) does not ship a separate input-embedding "
              "table -- Gemma ties input embeddings to the output projection, so this IS "
              "the same vector space (see huggingface/lerobot#1529, #2151), but call this "
              "out explicitly in the writeup rather than calling it 'the embedding table' "
              "unqualified.")
    print(f"  Shape: {tensor_np.shape}")
    print(f"  Sanity check PASSED: hidden dim {tensor_np.shape[1]} matches PaliGemma's "
          f"confirmed dimension from Experiments 1a-1d.")

    return tensor_np


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, emb_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, emb_dim),
        )
    def forward(self, a):
        return self.net(a)


class ActionDecoder(nn.Module):
    def __init__(self, emb_dim, action_dim):
        super().__init__()
        self.proj = nn.Linear(emb_dim, action_dim)
    def forward(self, z):
        return self.proj(z)


def nearest_neighbor_cosine_distance(encoded, vocab_embeddings, batch_size=256):
    encoded_norm = encoded / (np.linalg.norm(encoded, axis=1, keepdims=True) + 1e-8)
    vocab_norm = vocab_embeddings / (np.linalg.norm(vocab_embeddings, axis=1, keepdims=True) + 1e-8)

    min_distances = np.zeros(len(encoded))
    nearest_idxs = np.zeros(len(encoded), dtype=int)

    for i in range(0, len(encoded), batch_size):
        batch = encoded_norm[i:i + batch_size]
        sims = batch @ vocab_norm.T
        max_sims = sims.max(axis=1)
        max_idxs = sims.argmax(axis=1)
        min_distances[i:i + batch_size] = 1 - max_sims
        nearest_idxs[i:i + batch_size] = max_idxs

    return min_distances, nearest_idxs


def cosine_proximity_loss(encoded, vocab_embeddings_t):
    encoded_norm = torch.nn.functional.normalize(encoded, dim=-1)
    vocab_norm = torch.nn.functional.normalize(vocab_embeddings_t, dim=-1)

    with torch.no_grad():
        sims = encoded_norm @ vocab_norm.T
        nearest_idx = sims.argmax(dim=-1)
        nearest_targets = vocab_norm[nearest_idx]

    cos_sim = (encoded_norm * nearest_targets).sum(dim=-1)
    return (1 - cos_sim).mean()


def train_encoder_decoder(actions_train, actions_val, vocab_embeddings,
                            use_cosine_reg, cosine_weight=COSINE_REG_WEIGHT):
    action_dim = actions_train.shape[1]
    encoder = ActionEncoder(action_dim, EMB_DIM, ENCODER_HIDDEN).to(DEVICE)
    decoder = ActionDecoder(EMB_DIM, action_dim).to(DEVICE)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    recon_loss_fn = nn.MSELoss()

    vocab_for_reg = vocab_embeddings
    if len(vocab_embeddings) > 20000:
        rng = np.random.RandomState(SEED)
        sub_idx = rng.choice(len(vocab_embeddings), 20000, replace=False)
        vocab_for_reg = vocab_embeddings[sub_idx]
    vocab_t = torch.tensor(vocab_for_reg, dtype=torch.float32, device=DEVICE)

    actions_train_t = torch.tensor(actions_train, dtype=torch.float32, device=DEVICE)
    actions_val_t = torch.tensor(actions_val, dtype=torch.float32, device=DEVICE)

    n = len(actions_train_t)
    best_val_loss, best_state, patience_counter = float("inf"), None, 0

    for epoch in range(RECON_EPOCHS):
        encoder.train(); decoder.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            batch = actions_train_t[idx]

            optimizer.zero_grad()
            z = encoder(batch)
            recon = decoder(z)
            recon_loss = recon_loss_fn(recon, batch)

            if use_cosine_reg:
                cos_loss = cosine_proximity_loss(z, vocab_t)
                loss = recon_loss + cosine_weight * cos_loss
            else:
                loss = recon_loss

            loss.backward()
            optimizer.step()

        encoder.eval(); decoder.eval()
        with torch.no_grad():
            z_val = encoder(actions_val_t)
            recon_val = decoder(z_val)
            val_loss = recon_loss_fn(recon_val, actions_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "encoder": {k: v.clone() for k, v in encoder.state_dict().items()},
                "decoder": {k: v.clone() for k, v in decoder.state_dict().items()},
            }
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= RECON_PATIENCE:
            print(f"    Early stopping at epoch {epoch} (best val recon loss: {best_val_loss:.6f})")
            break

    encoder.load_state_dict(best_state["encoder"])
    decoder.load_state_dict(best_state["decoder"])
    encoder.eval(); decoder.eval()
    return encoder, decoder


def evaluate_condition(name, encoder, decoder, actions_test, vocab_embeddings):
    with torch.no_grad():
        actions_test_t = torch.tensor(actions_test, dtype=torch.float32, device=DEVICE)
        z_test = encoder(actions_test_t).cpu().numpy()
        recon_test = decoder(torch.tensor(z_test, dtype=torch.float32, device=DEVICE)).cpu().numpy()

    recon_r2 = r2_score(actions_test, recon_test)
    recon_mse = mean_squared_error(actions_test, recon_test)

    nn_distances, nn_idxs = nearest_neighbor_cosine_distance(z_test, vocab_embeddings)
    mean_nn_dist = nn_distances.mean()
    median_nn_dist = np.median(nn_distances)
    unique_nn_tokens = len(np.unique(nn_idxs))

    print(f"\n{name}")
    print(f"  Reconstruction R2: {recon_r2:.4f}")
    print(f"  Reconstruction MSE: {recon_mse:.6f}")
    print(f"  Mean nearest-neighbor cosine distance to vocab: {mean_nn_dist:.4f}")
    print(f"  Median nearest-neighbor cosine distance: {median_nn_dist:.4f}")
    print(f"  Unique vocab tokens used as nearest-neighbor: {unique_nn_tokens} / {len(actions_test)} samples")

    # Degenerate-collapse check: if nearly all test samples map to a
    # tiny handful of unique tokens, this is NOT evidence of meaningful
    # geometric alignment -- it's the regularizer finding a trivial
    # shortcut (collapsing everything onto 1-2 arbitrary points), which
    # would trivially minimize cosine-proximity loss without reflecting
    # any real structure in how actions relate to language semantics.
    is_degenerate = unique_nn_tokens <= 5 and len(actions_test) >= 50
    if is_degenerate:
        print(f"  WARNING: DEGENERATE COLLAPSE DETECTED -- only {unique_nn_tokens} unique "
              f"token(s) account for ALL {len(actions_test)} nearest-neighbor matches. "
              f"This is a trivial shortcut, not evidence of meaningful geometric "
              f"alignment with language structure. Treat low nn_distance from this "
              f"condition with suspicion, not as a positive finding.")

    return {
        "name": name, "z": z_test, "recon_r2": recon_r2, "recon_mse": recon_mse,
        "nn_distances": nn_distances, "mean_nn_dist": mean_nn_dist,
        "median_nn_dist": median_nn_dist, "unique_nn_tokens": unique_nn_tokens,
        "is_degenerate": is_degenerate,
    }


def plot_geometry(results_plain, results_reg, vocab_embeddings, vocab_sample_size=3000):
    rng = np.random.RandomState(SEED)
    if len(vocab_embeddings) > vocab_sample_size:
        sub_idx = rng.choice(len(vocab_embeddings), vocab_sample_size, replace=False)
        vocab_sample = vocab_embeddings[sub_idx]
    else:
        vocab_sample = vocab_embeddings

    combined = np.concatenate([vocab_sample, results_plain["z"], results_reg["z"]], axis=0)
    pca = PCA(n_components=2, random_state=SEED)
    combined_2d = pca.fit_transform(combined)

    n_vocab = len(vocab_sample)
    n_plain = len(results_plain["z"])
    vocab_2d = combined_2d[:n_vocab]
    plain_2d = combined_2d[n_vocab:n_vocab + n_plain]
    reg_2d = combined_2d[n_vocab + n_plain:]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Encoder Geometry: Where Do Encoded Actions Land? (PCA projection)",
                  fontsize=13, fontweight="bold")

    for ax, title in zip(axes, ["Vocabulary embeddings (reference)",
                                   "PLAIN encoder (reconstruction only)",
                                   "REGULARIZED encoder (+ cosine proximity)"]):
        ax.scatter(vocab_2d[:, 0], vocab_2d[:, 1], s=3, alpha=0.15, color="gray", label="Vocab tokens")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    axes[1].scatter(plain_2d[:, 0], plain_2d[:, 1], s=8, alpha=0.6, color="#e74c3c", label="Encoded actions")
    axes[2].scatter(reg_2d[:, 0], reg_2d[:, 1], s=8, alpha=0.6, color="#2ecc71", label="Encoded actions")
    for ax in axes[1:]:
        ax.legend(fontsize=9)

    plt.tight_layout()
    save_path = SAVE_DIR / "experiment2_encoder_geometry.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nGeometry plot saved -> {save_path}")

    fig2, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(results_plain["nn_distances"], bins=40, alpha=0.6, color="#e74c3c", label="PLAIN encoder")
    ax.hist(results_reg["nn_distances"], bins=40, alpha=0.6, color="#2ecc71", label="REGULARIZED encoder")
    ax.set_xlabel("Cosine distance to nearest vocabulary token")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Nearest-Neighbor Distances")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_path2 = SAVE_DIR / "experiment2_distance_distribution.png"
    plt.savefig(save_path2, dpi=150, bbox_inches="tight")
    print(f"Distance distribution plot saved -> {save_path2}")


def run_weight_sweep(actions_train, actions_val, actions_test, vocab_embeddings,
                       weights=(0.3, 0.1, 0.05, 0.02, 0.01, 0.005)):
    """
    Tests whether the degenerate collapse seen at COSINE_REG_WEIGHT=0.3 is a
    function of the weight magnitude, or a structural property of the loss
    itself (nearest-neighbor reassignment creating a rich-get-richer dynamic
    that collapses regardless of weight). Trains one regularized encoder per
    weight and reports unique_nn_tokens + mean_nn_dist at each.

    If collapse persists (very low unique_nn_tokens) across a wide range of
    weights -- including weights small enough that recon_r2 barely moves --
    that's strong evidence the collapse is inherent to per-sample nearest-
    neighbor argmax reassignment, not a tuning artifact. If collapse eases
    at lower weights (unique_nn_tokens climbs substantially, ideally into
    the dozens+), that's evidence proximity CAN be achieved without
    degenerating, just not at weight=0.3.
    """
    print("\n" + "=" * 70)
    print("WEIGHT SWEEP: testing whether collapse is weight-invariant")
    print("=" * 70)

    sweep_results = []
    for w in weights:
        print(f"\n--- COSINE_REG_WEIGHT = {w} ---")
        encoder_w, decoder_w = train_encoder_decoder(
            actions_train, actions_val, vocab_embeddings,
            use_cosine_reg=True, cosine_weight=w
        )
        res_w = evaluate_condition(f"REGULARIZED (weight={w})", encoder_w, decoder_w,
                                     actions_test, vocab_embeddings)
        sweep_results.append({"weight": w, **res_w})

    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'weight':>8} | {'recon_r2':>9} | {'mean_nn_dist':>13} | {'unique_tokens':>13} | degenerate")
    print("-" * 70)
    for r in sweep_results:
        print(f"{r['weight']:>8} | {r['recon_r2']:>9.4f} | {r['mean_nn_dist']:>13.4f} | "
              f"{r['unique_nn_tokens']:>13} | {r['is_degenerate']}")

    all_degenerate = all(r["is_degenerate"] for r in sweep_results)
    any_recovered = any(not r["is_degenerate"] and r["unique_nn_tokens"] >= 20 for r in sweep_results)

    print()
    if all_degenerate:
        print("RESULT: Collapse persists across the ENTIRE weight range tested. This "
              "supports the structural-dynamics explanation over a tuning explanation -- "
              "the per-sample nearest-neighbor argmax reassignment in cosine_proximity_loss "
              "creates a rich-get-richer dynamic (whichever token is locally denser attracts "
              "more samples next batch) that collapses regardless of regularization strength. "
              "Lowering the weight further would likely just slow convergence to the same "
              "collapsed state, not prevent it. This strengthens the negative-result claim: "
              "proximity to real vocabulary structure is NOT achievable through this "
              "regularization approach without destroying representational diversity.")
    elif any_recovered:
        print("RESULT: Collapse EASES at lower weights -- at least one weight setting "
              "achieved proximity without degenerate collapse (>=20 unique tokens). "
              "The collapse at weight=0.3 was a tuning artifact, not a structural property "
              "of the loss. The manifold argument may be buildable at a lower weight -- "
              "inspect the lowest-weight non-degenerate result to see if proximity and "
              "diversity are both meaningfully present, and check reconstruction cost there too."
              )
    else:
        print("RESULT: Mixed -- collapse severity varies with weight but doesn't clearly "
              "resolve into diverse assignment at any tested value. Inspect the full table "
              "above; consider testing weights below the smallest one tried here.")

    return sweep_results


def main():
    print("=" * 70)
    print("Experiment 2: Encoder Geometry Check")
    print("=" * 70 + "\n")

    vocab_embeddings = extract_vocab_embeddings()
    print(f"\nVocabulary embedding table shape: {vocab_embeddings.shape}\n")

    print("Loading actions from Experiment 1a features...")
    data = np.load(FEATURES_PATH)
    actions_array = data["actions"]
    print(f"  Loaded {len(actions_array)} actions, dim={actions_array.shape[1]}\n")

    n = len(actions_array)
    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=SEED)
    subtrain_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=SEED)

    actions_train = actions_array[subtrain_idx]
    actions_val = actions_array[val_idx]
    actions_test = actions_array[test_idx]

    print("=" * 70)
    print("CONDITION (a): PLAIN encoder (reconstruction loss only)")
    print("=" * 70)
    encoder_plain, decoder_plain = train_encoder_decoder(
        actions_train, actions_val, vocab_embeddings, use_cosine_reg=False
    )
    results_plain = evaluate_condition("PLAIN", encoder_plain, decoder_plain, actions_test, vocab_embeddings)

    print("\n" + "=" * 70)
    print("CONDITION (b): REGULARIZED encoder (+ cosine proximity)")
    print("=" * 70)
    encoder_reg, decoder_reg = train_encoder_decoder(
        actions_train, actions_val, vocab_embeddings, use_cosine_reg=True
    )
    results_reg = evaluate_condition("REGULARIZED", encoder_reg, decoder_reg, actions_test, vocab_embeddings)

    sweep_results = run_weight_sweep(actions_train, actions_val, actions_test, vocab_embeddings)

    plot_geometry(results_plain, results_reg, vocab_embeddings)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    recon_cost = results_plain["recon_r2"] - results_reg["recon_r2"]
    proximity_gain = results_plain["mean_nn_dist"] - results_reg["mean_nn_dist"]

    print(f"\nReconstruction R2: PLAIN={results_plain['recon_r2']:.4f}  "
          f"REGULARIZED={results_reg['recon_r2']:.4f}  (cost: {recon_cost:+.4f})")
    print(f"Mean cosine distance to vocab: PLAIN={results_plain['mean_nn_dist']:.4f}  "
          f"REGULARIZED={results_reg['mean_nn_dist']:.4f}  (improvement: {proximity_gain:+.4f})")

    if results_reg.get("is_degenerate", False):
        print("\nDEGENERATE COLLAPSE detected in the regularized condition -- nearly all "
              "encoded actions mapped to a handful of tokens. This low nn_distance is NOT "
              "valid evidence for the manifold hypothesis; it's a trivial optimization "
              "shortcut. Before drawing any conclusion, rerun with a LOWER "
              "COSINE_REG_WEIGHT (e.g. 0.05-0.1) and check whether unique_nn_tokens "
              "increases substantially -- a meaningful finding should show actions "
              "distributed across many semantically-plausible tokens, not collapsed onto 1-2.")
    elif results_plain["mean_nn_dist"] < 0.3:
        print("\nPLAIN encoder ALREADY lands close to real vocabulary tokens without "
              "being told to. STRONG support for the privileged manifold hypothesis.")
    elif proximity_gain > 0.15 and recon_cost < 0.1:
        print("\nPLAIN encoder drifts from real tokens, but REGULARIZATION successfully "
              "pulls encoded actions close to real vocabulary WITHOUT much reconstruction "
              "cost. The manifold argument is BUILDABLE.")
    elif proximity_gain > 0.15 and recon_cost >= 0.1:
        print("\nRegularization CAN pull actions close to real tokens, but at a real "
              "reconstruction cost. TENSION between representing actions well and "
              "anchoring to real token geometry -- needs an explicit tradeoff decision.")
    else:
        print("\nRegularization does NOT meaningfully pull actions closer to real "
              "vocabulary tokens even when explicitly optimized for. Evidence AGAINST "
              "the privileged manifold hypothesis as literally stated.")

    print("=" * 70)

    return results_plain, results_reg, sweep_results


if __name__ == "__main__":
    main()