"""
Experiment 1d: Vision-Token Decodability Probe
=================================================
Every probe so far (1a, 1b, 1c) reads the hidden state at the LAST
INSTRUCTION TOKEN position -- a pooled, language-centric summary of the
whole sequence. Orientation (roll/pitch/yaw) capped around R2~0.25-0.38
across every variant tried (linear/nonlinear decoder, bigger capacity,
+proprioception, action chunking).

UNTESTED HYPOTHESIS: orientation information may live in the VISION
token hidden states (the image patch tokens PaliGemma/SigLIP produce
per image, which sit earlier in the sequence, before the language
tokens) rather than in the pooled language-token summary that every
previous experiment read from. Wrist orientation is a visual-geometric
quantity -- how the gripper looks relative to the object -- so it's
plausible it's represented in the visual tokens' hidden states but gets
diluted/lost by the time information is pooled into the final language
token via attention.

WHAT CHANGES FROM PREVIOUS EXPERIMENTS
-----------------------------------------
- Hooks now capture ALL image-patch token positions at each layer, not
  just the last token.
- We extract THREE pooling variants of the vision tokens per layer:
    (a) MEAN pool over all vision tokens
    (b) MAX pool over all vision tokens (captures peak/salient signal)
    (c) mean pool of the SECOND HALF of image tokens only, as a proxy
        for the wrist camera block IF the two camera views are stacked
        as [base_tokens][wrist_tokens] in the sequence -- verify this
        assumption against Step 0's diagnostic before trusting the
        "wrist-only" result specifically.
- Everything else (frozen model, real select_action pipeline, Ridge +
  MLP probes, same train/test split protocol) is unchanged from 1a/1b,
  so results are directly comparable.

IMPORTANT CAVEAT -- VERIFY BEFORE TRUSTING RESULTS
----------------------------------------------------
The exact image-token layout (how many tokens per camera, in what order,
whether they're contiguous blocks) is PaliGemma/pi0-version-dependent
and NOT something we can hardcode confidently without inspection. This
script includes a MANDATORY diagnostic step (Step 0) that runs on the
first frame, inspects the actual sequence length and attention mask, and
PRINTS its best-guess token layout before extracting anything at scale.
Read that printout carefully -- if the guessed layout looks wrong (e.g.
image token count isn't a clean multiple of 256), stop and adjust
pool_vision_tokens()'s assumptions before running the full extraction.

Requires: GPU (same as Experiment 1a), lerobot/pi0_base, HF_TOKEN.
"""

import os
import gc
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ══════════════════════════════════════════════════════════════════════
# CONFIG (mirrors Experiment 1a's working config)
# ══════════════════════════════════════════════════════════════════════

MODEL_ID   = "lerobot/pi0_base"
DATASET_ID = "lerobot/libero_spatial_image"

PROBE_LAYERS = [-1, -2, -3, -4]
NUM_FRAMES   = 1500         # smaller than 1a's 1500 -- this extracts MORE data
                              # per frame (full token sequences, not just one
                              # vector), so keep this leaner to fit memory/time
STREAM_POOL_MULT = 4

MODEL_DTYPE = torch.bfloat16
ACTION_DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
ORIENTATION_DIMS = ["roll", "pitch", "yaw"]

SAVE_DIR = Path("/kaggle/working/probe_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

RIDGE_ALPHA  = 1.0
MLP_HIDDEN   = 256
MLP_DROPOUT  = 0.2
MLP_WD       = 1e-4
MLP_LR       = 1e-3
MLP_EPOCHS   = 200
MLP_PATIENCE = 20
MLP_BATCH    = 64

try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected.")

_free, _total = torch.cuda.mem_get_info(0)
PRIMARY_DEVICE = "cuda:0"
print(f"Device: {PRIMARY_DEVICE}  ({_free/1e9:.2f}GB free / {_total/1e9:.2f}GB total)")
print(f"HF_TOKEN: {'loaded' if HF_TOKEN else 'NOT SET -- this will fail, see 1a notes'}\n")


# ══════════════════════════════════════════════════════════════════════
# MODEL LOADING (same approach as Experiment 1a's working script)
# ══════════════════════════════════════════════════════════════════════

def load_pi0_config_safely(model_id, token=None):
    import json, tempfile
    from lerobot.policies.pi0 import PI0Config
    import draccus
    from huggingface_hub import hf_hub_download
    try:
        return PI0Config.from_pretrained(model_id, token=token)
    except Exception:
        pass
    config_path = hf_hub_download(repo_id=model_id, filename="config.json", token=token)
    with open(config_path) as f:
        config_dict = json.load(f)
    for bad_key in ("type", "_class_name", "_diffusers_version"):
        config_dict.pop(bad_key, None)
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "config.json")
    with open(tmp_path, "w") as f:
        json.dump(config_dict, f)
    with draccus.config_type("json"):
        return draccus.parse(PI0Config, tmp_path, args=[])


def gpu_alloc_monkeypatch(device, dtype):
    import contextlib
    import torch.nn as nn

    @contextlib.contextmanager
    def _ctx():
        _real_to = nn.Module.to
        _real_empty, _real_zeros = torch.empty, torch.zeros
        _real_ones, _real_full = torch.ones, torch.full
        _real_arange = torch.arange

        def _gpu_empty(*a, **k): k["device"] = device; k["dtype"] = dtype; return _real_empty(*a, **k)
        def _gpu_zeros(*a, **k): k["device"] = device; k["dtype"] = dtype; return _real_zeros(*a, **k)
        def _gpu_ones(*a, **k):  k["device"] = device; k["dtype"] = dtype; return _real_ones(*a, **k)
        def _gpu_full(size, fv, **k): k["device"] = device; k.setdefault("dtype", dtype); return _real_full(size, fv, **k)
        def _gpu_arange(*a, **k): k["device"] = device; return _real_arange(*a, **k)
        def _noop_to(self, *a, **k): return self

        torch.empty, torch.zeros, torch.ones, torch.full, torch.arange = (
            _gpu_empty, _gpu_zeros, _gpu_ones, _gpu_full, _gpu_arange
        )
        nn.Module.to = _noop_to
        try:
            yield
        finally:
            torch.empty, torch.zeros, torch.ones, torch.full, torch.arange = (
                _real_empty, _real_zeros, _real_ones, _real_full, _real_arange
            )
            nn.Module.to = _real_to

    return _ctx()


def _force_all_buffers_to_device(model, device):
    target = torch.device(device)
    moved = []
    for name, buf in model.named_buffers():
        if buf is not None and buf.device != target:
            moved.append(name)
            *parent_path, leaf = name.split(".")
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)
            parent._buffers[leaf] = buf.to(target)
    if moved:
        print(f"  [buffer sweep] moved {len(moved)} buffer(s) to {target}")
    return moved


def load_task_instructions(dataset_id, token=None):
    import json
    from huggingface_hub import hf_hub_download
    task_map = {}
    for filename, kind in (("meta/tasks.parquet", "parquet"), ("meta/tasks.jsonl", "jsonl")):
        try:
            path = hf_hub_download(repo_id=dataset_id, repo_type="dataset",
                                     filename=filename, token=token)
        except Exception:
            continue
        if kind == "parquet":
            import pyarrow.parquet as pq
            table = pq.read_table(path).to_pydict()
            keys = list(table.keys())
            idx_col = next((k for k in keys if "index" in k.lower()), None)
            text_col = next((k for k in keys if k.lower() in ("task", "text", "instruction")), None)
            for i, t in zip(table[idx_col], table[text_col]):
                task_map[int(i)] = t
        else:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    idx = row.get("task_index", row.get("index"))
                    txt = row.get("task", row.get("text"))
                    if idx is not None and txt is not None:
                        task_map[int(idx)] = txt
        if task_map:
            break
    if not task_map:
        raise RuntimeError(f"Could not load task instructions for {dataset_id}.")
    return task_map


def load_model_and_dataset():
    import glob
    from lerobot.policies.pi0 import PI0Policy
    from lerobot.policies.factory import make_pre_post_processors
    from huggingface_hub import snapshot_download, login, whoami
    from safetensors import safe_open

    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=False)

    print("Loading PI0Config...")
    config = load_pi0_config_safely(MODEL_ID, HF_TOKEN or None)
    if hasattr(config, "dtype"): config.dtype = "bfloat16"
    if hasattr(config, "device"): config.device = PRIMARY_DEVICE

    print("Downloading checkpoint...")
    ckpt_dir = snapshot_download(MODEL_ID, token=HF_TOKEN or None,
                                   ignore_patterns=["*.msgpack", "*.h5", "flax_model*"])
    shards = sorted(glob.glob(f"{ckpt_dir}/*.safetensors")) or \
             sorted(glob.glob(f"{ckpt_dir}/**/*.safetensors", recursive=True))
    if not shards:
        raise RuntimeError(f"No safetensors found in {ckpt_dir}")

    print("Constructing model skeleton on GPU (bf16)...")
    with gpu_alloc_monkeypatch(PRIMARY_DEVICE, MODEL_DTYPE):
        policy = PI0Policy(config)
    _force_all_buffers_to_device(policy, PRIMARY_DEVICE)

    param_map = dict(policy.named_parameters())
    buffer_map = dict(policy.named_buffers())
    all_tensors = {**param_map, **buffer_map}

    with safe_open(shards[0], framework="pt", device="cpu") as s:
        ckpt_sample = list(s.keys())[0]
    model_sample = list(param_map.keys())[0] if param_map else ""
    if model_sample.startswith("model.") and not ckpt_sample.startswith("model."):
        model_prefix, ckpt_strip = "model.", ""
    elif ckpt_sample.startswith("model.") and not model_sample.startswith("model."):
        model_prefix, ckpt_strip = "", "model."
    else:
        model_prefix, ckpt_strip = "", ""

    print("Streaming weights into GPU params...")
    loaded_keys = set()
    for shard_path in shards:
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for ck in shard.keys():
                target = None
                for candidate in (ck, ck[len(ckpt_strip):] if ckpt_strip else None,
                                   model_prefix + ck if model_prefix else None):
                    if candidate and candidate in all_tensors:
                        target = candidate
                        break
                if target is None:
                    continue
                cpu_bf16 = shard.get_tensor(ck).to(dtype=MODEL_DTYPE)
                all_tensors[target].data.copy_(cpu_bf16.to(PRIMARY_DEVICE))
                loaded_keys.add(target)
        gc.collect(); torch.cuda.empty_cache()

    print(f"  Loaded {len(loaded_keys)} tensors")
    policy = policy.eval()
    for p in policy.parameters():
        p.requires_grad = False
    _force_all_buffers_to_device(policy, PRIMARY_DEVICE)

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN required for PaliGemma tokenizer (gated repo).")
    whoami(token=HF_TOKEN)

    print("Building processors...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": PRIMARY_DEVICE}},
    )

    print("Loading task instructions...")
    task_map = load_task_instructions(DATASET_ID, HF_TOKEN or None)

    print(f"Streaming dataset pool ({NUM_FRAMES * STREAM_POOL_MULT} frames)...")
    from datasets import load_dataset
    raw_ds = load_dataset(DATASET_ID, split="train", streaming=True)
    frames = []
    for i, sample in enumerate(raw_ds):
        if i >= NUM_FRAMES * STREAM_POOL_MULT:
            break
        frames.append(sample)
    print(f"  Loaded {len(frames)} frames")

    return policy, preprocessor, task_map, frames


# ══════════════════════════════════════════════════════════════════════
# STEP 0: DIAGNOSTIC -- inspect actual token layout on frame 0
# ══════════════════════════════════════════════════════════════════════

def _pil_to_chw_float_tensor(pil_img, device):
    arr = np.asarray(pil_img, dtype=np.uint8).copy()
    if arr.ndim == 2:
        arr = arr[:, :, None]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t.float().div(255.0).to(device)


def _build_raw_batch(sample, task_map, device, model_dtype):
    task_index = sample.get("task_index")
    instruction = task_map.get(int(task_index), "perform the task") if task_index is not None else "perform the task"
    state_raw = sample["observation.state"]
    state_tensor = (
        state_raw.to(dtype=model_dtype, device=device) if isinstance(state_raw, torch.Tensor)
        else torch.tensor(state_raw, dtype=model_dtype, device=device)
    )
    return {
        "observation.images.base_0_rgb": _pil_to_chw_float_tensor(sample["observation.images.image"], device),
        "observation.images.left_wrist_0_rgb": _pil_to_chw_float_tensor(sample["observation.images.wrist_image"], device),
        "observation.state": state_tensor,
        "task": instruction,
    }


def _find_lm_layers(policy):
    candidates = [
        lambda p: p.model.paligemma_with_expert.paligemma.model.language_model.layers,
        lambda p: p.model.paligemma_with_expert.paligemma.language_model.model.layers,
        lambda p: p.model.paligemma.model.language_model.layers,
    ]
    for attempt in candidates:
        try:
            layers = attempt(policy)
            if layers is not None and len(layers) > 0:
                path = "model.paligemma_with_expert.paligemma...language_model.layers"
                return layers, path
        except AttributeError:
            continue
    best_name, best_mod = None, None
    for name, mod in policy.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 10:
            if best_mod is None or len(mod) > len(best_mod):
                best_name, best_mod = name, mod
    if best_mod is not None:
        return best_mod, best_name
    raise RuntimeError("Cannot find PaliGemma LM layers.")


def diagnose_token_layout(policy, preprocessor, frames, task_map):
    """
    MANDATORY first step. Runs one forward pass, inspects the actual
    sequence length, and prints a best-guess split between image tokens
    and language tokens based on standard PaliGemma conventions (image
    tokens first, contiguous block(s), followed by text tokens).
    """
    device = next(policy.parameters()).device
    sample = frames[0]
    raw_batch = _build_raw_batch(sample, task_map, device, MODEL_DTYPE)
    policy_input = preprocessor(raw_batch)

    print("\n" + "=" * 60)
    print("STEP 0: TOKEN LAYOUT DIAGNOSTIC (frame 0)")
    print("=" * 60)

    for k, v in policy_input.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    lang_tokens = policy_input.get("observation.language.tokens")
    lang_mask   = policy_input.get("observation.language.attention_mask")
    if lang_tokens is not None:
        n_lang_tokens = int(lang_mask.sum().item()) if lang_mask is not None else lang_tokens.shape[1]
        print(f"\n  Language tokens (from tokenizer): {n_lang_tokens} valid tokens "
              f"(padded shape: {lang_tokens.shape[1]})")
    else:
        n_lang_tokens = None
        print("\n  WARNING: could not find observation.language.tokens in preprocessor "
              "output -- check key names printed above.")

    lm_layers, path = _find_lm_layers(policy)
    seq_len_holder = {}

    def probe_hook(module, args, kwargs):
        hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
        if hs is not None:
            seq_len_holder["seq_len"] = hs.shape[1]
        return None

    h = lm_layers[0].register_forward_pre_hook(probe_hook, with_kwargs=True)
    with torch.no_grad():
        policy.reset()
        with torch.autocast(device_type="cuda", dtype=MODEL_DTYPE):
            policy.select_action(policy_input)
    h.remove()

    total_seq_len = seq_len_holder.get("seq_len")
    print(f"\n  ACTUAL total sequence length entering LM layer 0: {total_seq_len}")

    result = None
    if total_seq_len is not None and n_lang_tokens is not None:
        n_image_tokens = total_seq_len - n_lang_tokens
        print(f"  Inferred image token count (total - language): {n_image_tokens}")

        if n_image_tokens % 256 == 0:
            n_cameras_guess = n_image_tokens // 256
            print(f"  {n_image_tokens} / 256 = {n_cameras_guess} -- consistent with "
                  f"{n_cameras_guess} camera view(s) at standard 256-token encoding.")
        else:
            print(f"  {n_image_tokens} is NOT a clean multiple of 256 -- the standard "
                  f"assumption may not hold for this config. INSPECT MANUALLY before "
                  f"trusting pool_vision_tokens() below.")

        result = {
            "total_seq_len": total_seq_len,
            "n_lang_tokens": n_lang_tokens,
            "n_image_tokens": n_image_tokens,
        }
    else:
        print("  Could not fully determine layout -- inspect policy_input keys above "
              "and adjust manually.")

    print("=" * 60 + "\n")
    return result, path


# ══════════════════════════════════════════════════════════════════════
# HOOKS: capture FULL hidden state sequence (not just last token)
# ══════════════════════════════════════════════════════════════════════

def register_full_sequence_hooks(policy):
    hidden_states_cache = {li: None for li in PROBE_LAYERS}
    hooks = []
    lm_layers, path = _find_lm_layers(policy)
    num_layers = len(lm_layers)

    def make_hook(layer_idx):
        def hook_fn(module, input, kwargs, output):
            hs = output[0] if isinstance(output, tuple) else output
            hidden_states_cache[layer_idx] = hs.detach().float().cpu()
        return hook_fn

    for li in PROBE_LAYERS:
        abs_idx = num_layers + li
        hook = lm_layers[abs_idx].register_forward_hook(make_hook(li), with_kwargs=True)
        hooks.append(hook)

    return hidden_states_cache, hooks


def pool_vision_tokens(full_hs, n_image_tokens, n_lang_tokens):
    """
    full_hs: [1, seq_len, d_emb] tensor (or [seq_len, d_emb] after squeeze)
    Assumes image tokens occupy the FIRST n_image_tokens positions of the
    sequence (standard PaliGemma convention), followed by language tokens.
    """
    hs = full_hs.squeeze(0) if full_hs.dim() == 3 else full_hs
    image_hs = hs[:n_image_tokens]

    mean_pool = image_hs.mean(dim=0).numpy()
    max_pool  = image_hs.max(dim=0).values.numpy()

    half = n_image_tokens // 2
    wrist_proxy_pool = image_hs[half:].mean(dim=0).numpy()

    return mean_pool, max_pool, wrist_proxy_pool


# ══════════════════════════════════════════════════════════════════════
# EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def extract_vision_features(policy, preprocessor, frames, task_map,
                              hidden_states_cache, n_image_tokens, n_lang_tokens):
    device = next(policy.parameters()).device
    total_pool = len(frames)
    n_target = min(NUM_FRAMES, total_pool)
    indices = np.linspace(0, total_pool - 1, n_target, dtype=int)

    pooling_variants = ["mean", "max", "wrist_proxy"]
    all_feats = {li: {p: [] for p in pooling_variants} for li in PROBE_LAYERS}
    all_actions = []
    failed = 0

    print(f"\nExtracting vision-token features: {len(indices)} frames...")

    with torch.no_grad():
        for i, raw_idx in enumerate(indices):
            idx = int(raw_idx)
            if i % 100 == 0:
                print(f"  Frame {i}/{len(indices)}")

            for li in PROBE_LAYERS:
                hidden_states_cache[li] = None

            try:
                sample = frames[idx]
                raw_batch = _build_raw_batch(sample, task_map, device, MODEL_DTYPE)
                policy_input = preprocessor(raw_batch)

                policy.reset()
                with torch.autocast(device_type="cuda", dtype=MODEL_DTYPE):
                    policy.select_action(policy_input)
            except Exception as e:
                if i == 0:
                    raise RuntimeError(f"Failed on first frame: {e}")
                failed += 1
                continue

            captured = True
            frame_feats = {li: {} for li in PROBE_LAYERS}
            for li in PROBE_LAYERS:
                hs = hidden_states_cache[li]
                if hs is None:
                    captured = False
                    break
                mean_p, max_p, wrist_p = pool_vision_tokens(hs, n_image_tokens, n_lang_tokens)
                frame_feats[li]["mean"] = mean_p
                frame_feats[li]["max"] = max_p
                frame_feats[li]["wrist_proxy"] = wrist_p

            if not captured:
                failed += 1
                continue

            action_raw = sample.get("action", sample.get("actions"))
            if action_raw is None:
                failed += 1
                continue
            action = action_raw.numpy() if isinstance(action_raw, torch.Tensor) else np.array(action_raw, dtype=np.float32)
            if action.ndim > 1:
                action = action[0]

            for li in PROBE_LAYERS:
                for p in pooling_variants:
                    all_feats[li][p].append(frame_feats[li][p])
            all_actions.append(action)

    n_samples = len(all_actions)
    print(f"\n  Extracted {n_samples} samples, {failed} failed")
    if n_samples == 0:
        raise RuntimeError("No samples collected.")

    actions_array = np.stack(all_actions, axis=0)
    feature_arrays = {}
    for li in PROBE_LAYERS:
        for p in pooling_variants:
            key = f"{li}_{p}"
            feats = all_feats[li][p]
            if len(feats) == n_samples:
                feature_arrays[key] = np.stack(feats, axis=0)

    return feature_arrays, actions_array


# ══════════════════════════════════════════════════════════════════════
# PROBES (Ridge + small MLP, same protocol as 1a/1b)
# ══════════════════════════════════════════════════════════════════════

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
    model = SmallMLPProbe(input_dim, output_dim, MLP_HIDDEN, MLP_DROPOUT).to(PRIMARY_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    loss_fn = nn.MSELoss()
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=PRIMARY_DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=PRIMARY_DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=PRIMARY_DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=PRIMARY_DEVICE)

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


def run_probes(feature_arrays, actions_array):
    n = len(actions_array)
    act_dim = actions_array.shape[1]
    dim_names = ACTION_DIM_NAMES[:act_dim]
    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=42)
    subtrain_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)

    actions_train = actions_array[train_idx]
    actions_test = actions_array[test_idx]

    results = {}
    print("\n" + "=" * 60)
    print("VISION-TOKEN PROBE RESULTS (Ridge + MLP)")
    print("=" * 60)

    for key, X in feature_arrays.items():
        X_train, X_test = X[train_idx], X[test_idx]
        X_subtrain, X_val = X[subtrain_idx], X[val_idx]
        y_subtrain, y_val = actions_array[subtrain_idx], actions_array[val_idx]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(X_train_sc, actions_train)
        ridge_preds = ridge.predict(X_test_sc)
        ridge_r2 = r2_score(actions_test, ridge_preds)
        ridge_r2_dims = r2_score(actions_test, ridge_preds, multioutput="raw_values")

        scaler2 = StandardScaler()
        X_subtrain_sc = scaler2.fit_transform(X_subtrain)
        X_val_sc = scaler2.transform(X_val)
        X_test_sc2 = scaler2.transform(X_test)
        mlp = train_mlp(X_subtrain_sc, y_subtrain, X_val_sc, y_val, X.shape[1], act_dim)
        with torch.no_grad():
            mlp_preds = mlp(torch.tensor(X_test_sc2, dtype=torch.float32, device=PRIMARY_DEVICE)).cpu().numpy()
        mlp_r2 = r2_score(actions_test, mlp_preds)
        mlp_r2_dims = r2_score(actions_test, mlp_preds, multioutput="raw_values")

        orient_idxs = [i for i, n in enumerate(dim_names) if n in ORIENTATION_DIMS]
        ridge_orient = np.mean([ridge_r2_dims[i] for i in orient_idxs])
        mlp_orient = np.mean([mlp_r2_dims[i] for i in orient_idxs])

        results[key] = {
            "ridge_r2": ridge_r2, "ridge_r2_dims": ridge_r2_dims, "ridge_orient": ridge_orient,
            "mlp_r2": mlp_r2, "mlp_r2_dims": mlp_r2_dims, "mlp_orient": mlp_orient,
        }
        print(f"\n{key}")
        print(f"  Ridge: overall={ridge_r2:.4f}  orientation={ridge_orient:.4f}")
        print(f"  MLP:   overall={mlp_r2:.4f}  orientation={mlp_orient:.4f}")

    return results, dim_names


def plot_and_save(results, dim_names, known_best_baseline_orient=0.38):
    keys = list(results.keys())
    orient_mlp = [results[k]["mlp_orient"] for k in keys]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#e67e22" if "wrist" in k else "#3498db" if "mean" in k else "#2ecc71" for k in keys]
    bars = ax.barh(keys, orient_mlp, color=colors, edgecolor="white")
    ax.axvline(known_best_baseline_orient, color="red", linestyle="--",
               label=f"Best language-token result ({known_best_baseline_orient:.2f})")
    for bar, v in zip(bars, orient_mlp):
        ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Orientation avg R2 (MLP probe)")
    ax.set_title("Vision-Token Pooling vs Language-Token Baseline: Orientation R2")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_path = SAVE_DIR / "probe_1d_vision_tokens.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {save_path}")

    summary_path = SAVE_DIR / "probe_1d_summary.txt"
    with open(summary_path, "w") as f:
        f.write("EXPERIMENT 1d: VISION-TOKEN DECODABILITY\n")
        f.write("=" * 60 + "\n\n")
        for k in keys:
            r = results[k]
            f.write(f"{k}\n")
            f.write(f"  Ridge overall={r['ridge_r2']:.4f}  orientation={r['ridge_orient']:.4f}\n")
            f.write(f"  MLP   overall={r['mlp_r2']:.4f}  orientation={r['mlp_orient']:.4f}\n\n")
    print(f"Summary saved -> {summary_path}")


def main():
    print("=" * 60)
    print("Experiment 1d: Vision-Token Decodability Probe")
    print("=" * 60 + "\n")

    policy, preprocessor, task_map, frames = load_model_and_dataset()

    layout, path = diagnose_token_layout(policy, preprocessor, frames, task_map)
    if layout is None:
        raise RuntimeError(
            "Could not determine token layout automatically. Inspect the "
            "printed policy_input keys and sequence length above, then hardcode "
            "n_image_tokens/n_lang_tokens manually before proceeding."
        )

    n_image_tokens = layout["n_image_tokens"]
    n_lang_tokens = layout["n_lang_tokens"]

    print(f"Proceeding with n_image_tokens={n_image_tokens}, n_lang_tokens={n_lang_tokens}")
    print("(If this looks wrong based on the Step 0 diagnostic above, stop and fix "
          "pool_vision_tokens()'s assumptions before trusting results.)\n")

    hidden_states_cache, hooks = register_full_sequence_hooks(policy)

    feature_arrays, actions_array = extract_vision_features(
        policy, preprocessor, frames, task_map, hidden_states_cache,
        n_image_tokens, n_lang_tokens
    )

    for h in hooks:
        h.remove()

    np.savez(SAVE_DIR / "vision_token_features.npz", actions=actions_array, **feature_arrays)
    print(f"\nFeatures saved -> {SAVE_DIR / 'vision_token_features.npz'}")

    results, dim_names = run_probes(feature_arrays, actions_array)
    plot_and_save(results, dim_names)

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    best_key = max(results, key=lambda k: results[k]["mlp_orient"])
    best_orient = results[best_key]["mlp_orient"]
    print(f"\nBest vision-token orientation R2: {best_orient:.4f}  (via {best_key})")
    print(f"Best language-token orientation R2 (from 1b/1c): ~0.30-0.38")

    if best_orient > 0.45:
        print("\nVision tokens carry substantially MORE orientation signal than "
              "the pooled language token. This changes the architecture: condition "
              "the flow on vision-token features (or a pooled version), not just "
              "the last language token.")
    elif best_orient > 0.38:
        print("\nModest improvement over language-token baseline. Vision tokens "
              "help a little but don't resolve the ceiling.")
    else:
        print("\nVision tokens do NOT outperform the language-token baseline. "
              "Orientation appears genuinely hard to linearly/nonlinearly recover "
              "from ANY single-frame PaliGemma representation, not just the pooled "
              "language summary. This strengthens the case that orientation needs "
              "a fundamentally different mechanism (e.g. dedicated small expert "
              "trained on raw action space) rather than routing through VLM embeddings.")
    print("=" * 60)


if __name__ == "__main__":
    main()