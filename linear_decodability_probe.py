"""
Experiment 1: Linear Decodability Probe  (Kaggle — single/dual T4 GPU version)
================================================================================
Tests whether PaliGemma (pi0's VLM backbone) hidden states linearly predict
ground truth robot actions, using LIBERO Spatial data.

WHY THE PREVIOUS VERSION FAILED
--------------------------------
`_forward_pass` hand-called `paligemma(input_ids=..., pixel_values=..., attention_mask=...)`
directly. That is not how PI0Policy's real inference path works, and the batch
being passed in (`observation.images.image`, `observation.state`, etc.) never
contained `input_ids` in the first place — there was no tokenization step at
all. PaliGemma's text tower requires tokenized language instructions
(`observation.language.tokens` / `observation.language.attention_mask`), which
in the real LeRobot pipeline are produced by the *preprocessor* returned from
`make_pre_post_processors`, not by hand.

FIXES IN THIS VERSION
----------------------
FIX 1 — Use the REAL inference pipeline instead of hand-calling the backbone:
    preprocessor, postprocessor = make_pre_post_processors(policy.config, MODEL_ID, ...)
    policy_input = preprocessor(raw_batch)
    policy.select_action(policy_input)     # fires hooks internally
    This guarantees `input_ids`/`pixel_values`/`attention_mask`-equivalent
    tensors are built correctly (tokenization + normalization), because we're
    using LeRobot's own code path instead of reconstructing it.

FIX 2 — Language instruction is fetched from dataset task metadata:
    lerobot/libero_spatial_image does NOT carry a per-frame "task" string.
    It stores task_index per frame and a separate tasks table
    (meta/tasks.parquet or meta/tasks.jsonl) mapping task_index -> instruction.
    We download and parse that mapping once, then look up per-frame.

FIX 3 — select_action's internal action queue is bypassed for probing:
    select_action() only runs a *real* forward pass when its internal action
    queue is empty (it caches `n_action_steps` actions per real inference
    call). For a per-frame linear probe we need a fresh forward pass on every
    frame, so we call policy.reset() before every single frame. This clears
    the queue and guarantees every frame triggers model.sample_actions(...)
    (and therefore fires the hooks) exactly once.

FIX 4 — Hidden state check now happens strictly AFTER the forward pass,
    inside a try/except that surrounds the whole frame, so a bad frame can
    never silently corrupt the cache used by the next frame.

FIX 5 — action key / dtype handling unchanged from before (still correct):
    None-check before .numpy()/.ndim, handles torch.Tensor / np.ndarray / list.

FIX 6 — GPU monkey-patch context is now a context manager (was a bare
    try/finally in a helper function before) so it can't leak state if an
    exception occurs during weight streaming, and is easier to audit.

FIX 7 — Kaggle has 2x T4 by default. This script defaults to using GPU 0 only
    (single-GPU path, matching the original design intent) but includes a
    DUAL_GPU flag + guidance if you want to split image encoder / LM across
    both — off by default because pi0's forward pass is not natively
    device_map-aware and naive splitting will break the hook layer paths.

FIX 8 — Dataset streaming now retries once on transient HF network errors
    (Kaggle's network can be flakier than Colab's for HF Hub streaming).

FIX 9 — `NUM_FRAMES` sampling now happens on the *episode-aware* stream
    (previous version implicitly assumed frames arrive already shuffled /
    representative — LIBERO parquet files are stored episode-contiguous, so
    naively taking the first N frames means "the first ~1-2 episodes
    repeated", not a spread across tasks). We now stream a larger pool and
    subsample uniformly by index across the pool, same as before, but we
    increase the default pool multiplier and document the caveat clearly.

FIX 16 (NEW) — position_ids device bug in SigLIP vision embeddings:
    The previous run crashed with:
        RuntimeError: Expected all tensors to be on the same device, but got
        index is on cpu, different from other tensors on cuda:0
    inside SiglipVisionEmbeddings.forward(), at:
        embeddings = embeddings + self.position_embedding(self.position_ids)
    Root cause: `position_ids` is a non-persistent buffer created internally
    by transformers via `torch.arange(...).expand(...)`. Our GPU monkeypatch
    (gpu_alloc_monkeypatch) only intercepted torch.empty/zeros/ones/full — it
    never intercepted torch.arange, so this buffer was built on CPU (the
    default device) during model construction. Worse, because the same patch
    also no-ops nn.Module.to() (to stop huge accidental CPU->GPU copies during
    construction), nothing ever moved it to GPU afterward either. It is also
    NOT saved in the safetensors checkpoint (persistent=False), so the shard-
    streaming loop in load_model_and_dataset() has no key to restore it from
    — it just silently stays wherever it was created.
    Fix (two layers of defense):
      (a) gpu_alloc_monkeypatch now also patches torch.arange to force
          `device=<target>` (but deliberately does NOT force `dtype` to
          bf16 — arange output here is used as an integer index into an
          nn.Embedding, so it must stay long/int, not bf16).
      (b) A new belt-and-suspenders sweep, _force_all_buffers_to_device(),
          runs immediately after `PI0Policy(config)` is constructed. It walks
          every named buffer (persistent or not) and moves any stragglers
          left on the wrong device — this catches position_ids-style buffers
          even if some other, still-unpatched factory function (torch.linspace,
          `.new_zeros()`, etc.) created them on CPU in some future transformers
          version.
      (c) `position_ids` (and other `*_ids`-style index buffers) are added to
          a new _KNOWN_DEVICE_ONLY_BUFFER_SUFFIXES allowlist so Step 4's
          "unexpected missing tensor" warning doesn't cry wolf about them —
          they don't need a checkpoint VALUE (they're deterministic), they
          only needed to be on the right DEVICE, which is now handled above.

FIX 17 (NEW) — state dtype mismatch:
    Once FIX 16 got past construction, the first real forward pass raised:
        RuntimeError: mat1 and mat2 must have the same dtype, but got Float
        and BFloat16
    inside `self.state_proj(state)` — a plain nn.Linear with bf16 weights
    (streamed in from the checkpoint), but `observation.state` was hardcoded
    to float32 in _build_raw_batch(). The vision tower tolerates fp32 input
    because SigLIP/PaliGemma upcast pixel_values internally before their own
    forward; state_proj has no such internal cast.
    Fix: build `observation.state` in MODEL_DTYPE (bf16) instead of a
    hardcoded float32, in _build_raw_batch().

FIX 18 (NEW) — internally-generated noise tensor dtype mismatch:
    Fixing FIX 17 got past state_proj, but hit the identical class of error
    one step further into the pipeline, inside sample_actions() ->
    denoise_step() -> embed_suffix() -> `self.action_in_proj(noisy_actions)`.
    `noisy_actions` (x_t) is diffusion noise that pi0's OWN sample_actions()
    generates internally (we never pass a `noise=` kwarg), almost certainly
    via something like torch.randn(..., device=...) with no explicit dtype,
    defaulting to float32 regardless of the model's bf16 weights. This
    tensor is not something _build_raw_batch() controls, so it can't be
    point-fixed the way FIX 17 fixed `state`.
    Fix: wrap the `policy.select_action(policy_input)` call in
    `torch.autocast(device_type="cuda", dtype=MODEL_DTYPE)`. Autocast
    automatically casts fp32 activations to bf16 at matmul/linear op
    boundaries without requiring us to know in advance which internal
    lerobot tensor needs casting, and does not alter the dtype of the
    bf16 parameters we streamed in manually — it only affects the dtype
    tensors are cast to going INTO an op.

KNOWN REMAINING UNCERTAINTY (verify before trusting downstream results)
------------------------------------------------------------------------
The exact keys returned by `preprocessor(raw_batch)` are lerobot-version-
dependent (this has changed across releases: `observation.language.tokens`
vs nested dicts, etc.). CELL 5B below prints the full key/shape structure on
the very first frame and RAISES if anything looks obviously wrong, rather
than silently proceeding — read that printout before trusting the rest of
the run.
"""

# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 1 ── INSTALL  (run once per session; restart runtime after)
# ══════════════════════════════════════════════════════════════════════════════
"""
Kaggle notebooks: after this cell finishes, use Run All from the next cell on,
or manually re-run from Cell 2 (Kaggle doesn't have a one-click "restart
runtime" the way Colab does — use Factory Reset / Restart Session from the
notebook menu, then re-run from Cell 2).
"""

INSTALL_SCRIPT = r"""
set -e
echo "=== Step 1/6: pinned av ==="
pip install "av==16.1.0" --quiet --no-deps

echo "=== Step 2/6: accelerate ==="
pip install "accelerate>=0.26.0" --quiet

echo "=== Step 3/6: safetensors ==="
pip install "safetensors>=0.4.0" --quiet

echo "=== Step 4/6: lerobot[pi] from git ==="
pip install "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git" \
    --quiet --extra-index-url https://pypi.org/simple

echo "=== Step 5/6: scikit-learn ==="
pip install "scikit-learn>=1.3.0" --quiet

echo "=== Step 6/6: Pillow + pyarrow (needed for tasks.parquet lookup) ==="
pip install "Pillow>=9.0.0" "pyarrow>=14.0.0" --quiet

echo ""
echo "All installs done. Restart the Kaggle session, then run from Cell 2."
"""

# Uncomment and run to install:
# import subprocess
# subprocess.run(["bash", "-c", INSTALL_SCRIPT], check=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 2 ── IMPORTS & SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

import os
import gc
import json
import time
import tempfile
import warnings
import contextlib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print("Python packages OK")
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"GPU count: {n_gpus}")
    for gi in range(n_gpus):
        props = torch.cuda.get_device_properties(gi)
        free, total = torch.cuda.mem_get_info(gi)
        print(f"  GPU {gi}: {props.name}  {props.total_memory / 1e9:.1f} GB total, "
              f"{free/1e9:.1f} GB free")

try:
    import sklearn; print(f"scikit-learn : {sklearn.__version__}")
except ImportError:
    raise ImportError("scikit-learn not found — did you run Cell 1 and restart the session?")

try:
    import lerobot; print(f"lerobot      : {lerobot.__version__}")
except ImportError:
    raise ImportError("lerobot not found — did you run Cell 1 and restart the session?")

try:
    import av; print(f"av           : {av.__version__}")
except ImportError:
    raise ImportError("av not found — did you run Cell 1 and restart the session?")

try:
    import draccus; print("draccus      : OK")
except ImportError:
    raise ImportError("draccus not found — lerobot install may have failed")

try:
    import accelerate; print(f"accelerate   : {accelerate.__version__}")
except ImportError:
    raise ImportError("accelerate not found — did you run Cell 1 and restart the session?")

try:
    from safetensors import safe_open; print("safetensors  : OK")
except ImportError:
    raise ImportError("safetensors not found — did you run Cell 1 and restart the session?")

try:
    import pyarrow; print(f"pyarrow      : {pyarrow.__version__}")
except ImportError:
    raise ImportError("pyarrow not found — needed to read tasks.parquet. Run Cell 1 again.")

print("\nAll imports OK — safe to continue.\n")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 3 ── CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# ── Hard GPU memory reset (FIX 10) ────────────────────────────────────────────
for _leftover_name in ("policy", "preprocessor", "postprocessor", "frames",
                       "task_map", "hidden_states_cache", "hooks", "results"):
    if _leftover_name in dir():
        del globals()[_leftover_name]

gc.collect()
if torch.cuda.is_available():
    for _gi in range(torch.cuda.device_count()):
        with torch.cuda.device(_gi):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
gc.collect()

if torch.cuda.is_available():
    _alloc0 = torch.cuda.memory_allocated(0) / 1e9
    _free0, _total0 = torch.cuda.mem_get_info(0)
    print(f"GPU 0 after reset: alloc={_alloc0:.2f}GB, free={_free0/1e9:.2f}GB / {_total0/1e9:.2f}GB")
    if _alloc0 > 0.5:
        print("  WARNING: significant memory still allocated after reset.")
        print("  If this is a fresh kernel start (not a re-run), another process/kernel")
        print("  may be sharing this GPU. Consider: Kaggle menu > Session > Restart & Run All.")

# ── HF Token ─────────────────────────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

if HF_TOKEN:
    print(f"HF_TOKEN : loaded ({HF_TOKEN[:8]}...)")
else:
    print("HF_TOKEN : NOT SET. This experiment WILL FAIL without it, because")
    print("           pi0's tokenizer processor requires the gated")
    print("           google/paligemma-3b-pt-224 repo. Set it in Kaggle:")
    print("           Add-ons > Secrets > HF_TOKEN, and accept the license at")
    print("           https://huggingface.co/google/paligemma-3b-pt-224 first.")

# ── Model & Dataset ───────────────────────────────────────────────────────────
MODEL_ID   = "lerobot/pi0_base"
DATASET_ID = "lerobot/libero_spatial_image"   # public, no auth needed

# ── Device ────────────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected. Enable a GPU accelerator in Kaggle's notebook settings.")

_free_by_gpu = []
for _gi in range(torch.cuda.device_count()):
    _free, _total = torch.cuda.mem_get_info(_gi)
    _free_by_gpu.append(_free)
    print(f"  GPU {_gi}: {_free/1e9:.2f}GB free / {_total/1e9:.2f}GB total")

_best_gpu = int(np.argmax(_free_by_gpu)) if _free_by_gpu else 0
PRIMARY_DEVICE = f"cuda:{_best_gpu}"
MODEL_DTYPE    = torch.bfloat16
print(f"  -> Selected {PRIMARY_DEVICE} ({_free_by_gpu[_best_gpu]/1e9:.2f}GB free)")

MIN_FREE_GB_REQUIRED = 11.0
if _free_by_gpu[_best_gpu] / 1e9 < MIN_FREE_GB_REQUIRED:
    raise RuntimeError(
        f"Best available GPU ({PRIMARY_DEVICE}) only has "
        f"{_free_by_gpu[_best_gpu]/1e9:.2f}GB free, need ~{MIN_FREE_GB_REQUIRED}GB. "
        "This usually means a previous run left tensors allocated in this kernel "
        "session. Restart the Kaggle session (not just re-run cells) and try again."
    )

DUAL_GPU = False

# ── Experiment ────────────────────────────────────────────────────────────────
NUM_FRAMES        = 1500
STREAM_POOL_MULT  = 4
PROBE_LAYERS      = [-1, -2, -3, -4]
RIDGE_ALPHA       = 1.0
CHECKPOINT_EVERY  = 500

# ── Output ────────────────────────────────────────────────────────────────────
SAVE_DIR = Path("/kaggle/working/probe_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

ACTION_DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

_KNOWN_ABSENT_PREFIXES = (
    "model.action_in_proj",
    "model.action_out_proj",
    "model.action_time_mlp",
    "model.paligemma_with_expert.attn_pool",
    "model.paligemma_with_expert.action_expert",
)

# FIX 16(c): buffers that are legitimately absent from the checkpoint because
# they're deterministic index buffers (e.g. torch.arange-based position_ids),
# not learned weights. They only need to be on the right DEVICE (handled by
# _force_all_buffers_to_device), not restored from a checkpoint VALUE.
_KNOWN_DEVICE_ONLY_BUFFER_SUFFIXES = (
    "position_ids",
    "cache_position",
)

print(f"Device        : {PRIMARY_DEVICE}")
print(f"Model dtype   : {MODEL_DTYPE}")
print(f"Target frames : {NUM_FRAMES}  (streaming pool: {NUM_FRAMES * STREAM_POOL_MULT})")
print(f"Layers        : {PROBE_LAYERS}")
print(f"Save dir      : {SAVE_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 4 ── HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def print_gpu_mem(tag=""):
    if not torch.cuda.is_available():
        return
    alloc    = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    free, total = torch.cuda.mem_get_info(0)
    print(f"  [mem/{tag}] alloc={alloc:.2f}GB  reserved={reserved:.2f}GB  "
          f"free={free/1e9:.2f}GB / {total/1e9:.2f}GB total")


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def save_checkpoint(layer_arrays_partial, actions_partial, frame_idx):
    if len(actions_partial) == 0:
        return
    path = SAVE_DIR / "features_checkpoint.npz"
    save_dict = {
        f"layer_{k}": np.stack(v)
        for k, v in layer_arrays_partial.items()
        if len(v) > 0
    }
    save_dict["actions"]   = np.stack(actions_partial)
    save_dict["frame_idx"] = np.array([frame_idx])
    np.savez(path, **save_dict)
    print(f"  Checkpoint saved at frame {frame_idx} -> {path}")


def _is_known_absent(key):
    if any(key.startswith(p) for p in _KNOWN_ABSENT_PREFIXES):
        return True
    # FIX 16(c): also treat known device-only index buffers as expected-absent
    # from the checkpoint (they were never meant to be restored from a value).
    if any(key.endswith(suffix) for suffix in _KNOWN_DEVICE_ONLY_BUFFER_SUFFIXES):
        return True
    return False


@contextlib.contextmanager
def gpu_alloc_monkeypatch(device, dtype):
    """
    Context manager that forces newly-constructed tensors to be allocated
    directly on `device` (and, for float factories, in `dtype`) during model
    construction, instead of building on CPU first and copying over later
    (which would transiently double memory and is slow).

    FIX 16(a): previously this patched torch.empty/zeros/ones/full but NOT
    torch.arange. transformers builds several deterministic index buffers
    (most notably SigLIP's `position_ids` in SiglipVisionEmbeddings) via
    `torch.arange(...).expand(...)` at __init__ time. Since that call wasn't
    intercepted, those buffers were built on CPU and — because nn.Module.to()
    is also no-op'd below to avoid huge accidental CPU<->GPU copies during
    construction — nothing ever moved them to GPU afterward either. That
    silently broke the vision tower's forward pass the first time it ran
    (RuntimeError: index is on cpu, different from other tensors on cuda:0).

    torch.arange output is deliberately NOT forced to `dtype` here (only to
    `device`): these buffers are used as integer indices into nn.Embedding,
    so forcing them to bf16 would break embedding lookups just as badly as
    leaving them on the wrong device.
    """
    import torch.nn as nn

    _real_to     = nn.Module.to
    _real_empty  = torch.empty
    _real_zeros  = torch.zeros
    _real_ones   = torch.ones
    _real_full   = torch.full
    _real_arange = torch.arange   # FIX 16(a): capture original for restore

    def _gpu_empty(*args, **kwargs):
        kwargs["device"] = device
        kwargs["dtype"]  = dtype
        return _real_empty(*args, **kwargs)

    def _gpu_zeros(*args, **kwargs):
        kwargs["device"] = device
        kwargs["dtype"]  = dtype
        return _real_zeros(*args, **kwargs)

    def _gpu_ones(*args, **kwargs):
        kwargs["device"] = device
        kwargs["dtype"]  = dtype
        return _real_ones(*args, **kwargs)

    def _gpu_full(size, fill_value, **kwargs):
        kwargs["device"] = device
        kwargs.setdefault("dtype", dtype)
        return _real_full(size, fill_value, **kwargs)

    def _gpu_arange(*args, **kwargs):
        # FIX 16(a): force device only. Do NOT touch dtype — arange is used
        # for integer index buffers (e.g. SigLIP position_ids); forcing bf16
        # here would break embedding lookups (needs long/int dtype).
        kwargs["device"] = device
        return _real_arange(*args, **kwargs)

    def _noop_to(self, *args, **kwargs):
        return self

    try:
        torch.empty  = _gpu_empty
        torch.zeros  = _gpu_zeros
        torch.ones   = _gpu_ones
        torch.full   = _gpu_full
        torch.arange = _gpu_arange   # FIX 16(a): apply patch
        nn.Module.to = _noop_to
        yield
    finally:
        torch.empty  = _real_empty
        torch.zeros  = _real_zeros
        torch.ones   = _real_ones
        torch.full   = _real_full
        torch.arange = _real_arange  # FIX 16(a): restore original
        nn.Module.to = _real_to


def _force_all_buffers_to_device(model, device):
    """
    FIX 16(b): belt-and-suspenders sweep, run immediately after model
    construction (still inside/just after the gpu_alloc_monkeypatch block).

    Walks every registered buffer (persistent AND non-persistent — the
    position_ids bug was specifically a non-persistent buffer, which is
    excluded from named_buffers()'s default behavior in older torch versions
    but included in modern ones; we don't rely on that and instead just
    check every buffer we can enumerate) and force-moves any stragglers left
    on the wrong device onto `device` in place.

    This is deliberately independent of the torch.arange patch above: it's
    a safety net in case some *other*, currently-unpatched factory function
    (torch.linspace, tensor.new_zeros(), raw torch.tensor(...) literals,
    etc.) creates a buffer on the wrong device in some future transformers/
    lerobot version. Cheap to run once at construction time, and prints a
    clear diagnostic naming exactly which buffers it had to move, so if this
    fires again in the future you'll see it immediately in the logs instead
    of hitting a bare device-mismatch RuntimeError deep in a forward pass.
    """
    target = torch.device(device)
    moved = []

    for name, buf in model.named_buffers():
        if buf is None:
            continue
        if buf.device != target:
            moved.append((name, str(buf.device)))
            *parent_path, leaf = name.split(".")
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)
            # Reassign in place so every reference (including ones already
            # cached elsewhere, e.g. in a submodule's __init__ locals) still
            # points at a valid, correctly-placed tensor going forward.
            parent._buffers[leaf] = buf.to(target)

    if moved:
        print(f"  [buffer sweep] moved {len(moved)} stray buffer(s) to {target}:")
        for n, old_dev in moved[:20]:
            print(f"    {n}  (was on {old_dev})")
        if len(moved) > 20:
            print(f"    ... and {len(moved) - 20} more")
    else:
        print(f"  [buffer sweep] all buffers already on {target}")

    return moved


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 5 ── LOAD MODEL, PROCESSORS, TASK METADATA & DATASET
# ══════════════════════════════════════════════════════════════════════════════

def load_pi0_config_safely(model_id, token=None):
    from lerobot.policies.pi0 import PI0Config
    import draccus
    from huggingface_hub import hf_hub_download

    try:
        cfg = PI0Config.from_pretrained(model_id, token=token)
        print("  Config: standard path OK")
        return cfg
    except Exception as e:
        print(f"  Config: standard path failed ({type(e).__name__}) — using patched loader")

    config_path = hf_hub_download(repo_id=model_id, filename="config.json", token=token)
    with open(config_path) as f:
        config_dict = json.load(f)

    removed = []
    for bad_key in ("type", "_class_name", "_diffusers_version"):
        if bad_key in config_dict:
            del config_dict[bad_key]
            removed.append(bad_key)
    if removed:
        print(f"  Config: removed unknown fields {removed}")

    tmp_dir  = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "config.json")
    with open(tmp_path, "w") as f:
        json.dump(config_dict, f)

    with draccus.config_type("json"):
        cfg = draccus.parse(PI0Config, tmp_path, args=[])

    print("  Config: patched path OK")
    return cfg


def _detect_key_prefix(shards, param_map):
    from safetensors import safe_open

    with safe_open(shards[0], framework="pt", device="cpu") as s:
        ckpt_keys = list(s.keys())

    if not ckpt_keys:
        return "", ""

    ckpt_sample  = ckpt_keys[0]
    model_keys   = list(param_map.keys())
    model_sample = model_keys[0] if model_keys else ""

    if ckpt_sample.startswith("model.") and not model_sample.startswith("model."):
        print("  Key prefix: ckpt has 'model.' prefix, model does not -> stripping from ckpt")
        return "model.", ""

    if model_sample.startswith("model.") and not ckpt_sample.startswith("model."):
        print("  Key prefix: model has 'model.' prefix, ckpt does not -> adding to lookup")
        return "", "model."

    if ckpt_sample in param_map:
        print("  Key prefix: keys match directly — no remapping needed")
        return "", ""

    print(f"  Key prefix: could not auto-detect (ckpt='{ckpt_sample}', model='{model_sample}')")
    return "", ""


def load_task_instructions(dataset_id, token=None):
    """
    FIX 2: lerobot/libero_spatial_image stores task instructions in a separate
    metadata table (meta/tasks.parquet in v3.0 datasets, meta/tasks.jsonl in
    older ones) mapping task_index -> natural-language instruction. Per-frame
    samples only carry `task_index`, not the instruction text itself.
    Tries both filenames since this varies by dataset codebase_version.
    """
    from huggingface_hub import hf_hub_download

    task_map = {}

    for filename, kind in (("meta/tasks.parquet", "parquet"), ("meta/tasks.jsonl", "jsonl")):
        try:
            path = hf_hub_download(
                repo_id=dataset_id, repo_type="dataset", filename=filename, token=token
            )
        except Exception:
            continue

        if kind == "parquet":
            import pyarrow.parquet as pq
            table = pq.read_table(path).to_pydict()
            keys = list(table.keys())
            idx_col  = next((k for k in keys if "index" in k.lower()), None)
            text_col = next((k for k in keys if k.lower() in ("task", "text", "instruction")), None)
            if idx_col is None or text_col is None:
                print(f"  WARNING: unexpected columns in tasks.parquet: {keys}")
                continue
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
            print(f"  Loaded {len(task_map)} task instructions from {filename}")
            break

    if not task_map:
        raise RuntimeError(
            f"Could not load task instructions for {dataset_id}. "
            "Checked meta/tasks.parquet and meta/tasks.jsonl, both missing/unparseable. "
            "Inspect the dataset repo file list on huggingface.co/datasets/"
            f"{dataset_id}/tree/main/meta to find the correct filename/schema."
        )

    return task_map


def load_model_and_dataset():
    import glob
    import torch.nn as nn
    from lerobot.policies.pi0 import PI0Policy
    from lerobot.policies.factory import make_pre_post_processors
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import huggingface_hub

    # ── HF login ─────────────────────────────────────────────────────────────
    if HF_TOKEN:
        print("Logging in to Hugging Face Hub...")
        huggingface_hub.login(token=HF_TOKEN, add_to_git_credential=False)
        print("  Logged in")
    else:
        print("No HF_TOKEN — skipping login (fine for public repos)")

    aggressive_cleanup()
    print_gpu_mem("before model load")

    # ── Config ───────────────────────────────────────────────────────────────
    print("\nLoading PI0Config...")
    config = load_pi0_config_safely(MODEL_ID, HF_TOKEN or None)
    if hasattr(config, "dtype"):
        config.dtype = "bfloat16"
    if hasattr(config, "device"):
        config.device = PRIMARY_DEVICE
    print(f"  config.dtype={getattr(config,'dtype','N/A')}  "
          f"config.device={getattr(config,'device','N/A')}")

    # ── Download checkpoint ───────────────────────────────────────────────────
    print("\nStep 1/5: Downloading / verifying checkpoint cache...")
    ckpt_dir = snapshot_download(
        MODEL_ID,
        token=HF_TOKEN or None,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    )
    print(f"  Checkpoint dir: {ckpt_dir}")

    shards = sorted(glob.glob(f"{ckpt_dir}/*.safetensors"))
    if not shards:
        shards = sorted(glob.glob(f"{ckpt_dir}/**/*.safetensors", recursive=True))
    if not shards:
        raise RuntimeError(f"No .safetensors files found in {ckpt_dir}.")
    print(f"  Found {len(shards)} shard(s)")

    # ── Build model skeleton directly on GPU in bf16 ──────────────────────────
    print("\nStep 2/5: Constructing model skeleton (GPU bf16 allocators)...")
    _free_before_build, _ = torch.cuda.mem_get_info(int(PRIMARY_DEVICE.split(":")[1]))
    print(f"  Free memory immediately before model construction: {_free_before_build/1e9:.2f}GB")
    if _free_before_build / 1e9 < MIN_FREE_GB_REQUIRED:
        raise RuntimeError(
            f"Only {_free_before_build/1e9:.2f}GB free right before construction "
            f"(need ~{MIN_FREE_GB_REQUIRED}GB). Something allocated GPU memory between "
            "the Cell 3 check and here — restart the Kaggle session and re-run from Cell 2."
        )

    with gpu_alloc_monkeypatch(PRIMARY_DEVICE, MODEL_DTYPE):
        policy = PI0Policy(config)

    sample_param = next(policy.parameters())
    print(f"  Skeleton on GPU. param device={sample_param.device}, dtype={sample_param.dtype}")

    # FIX 16(b): sweep for any buffers (e.g. SigLIP's position_ids, built via
    # torch.arange) that ended up on the wrong device despite the patch above.
    # Run this even though torch.arange is now patched — it's a zero-cost
    # safety net against any other factory function we haven't anticipated.
    print("  Sweeping buffers for device mismatches...")
    _force_all_buffers_to_device(policy, PRIMARY_DEVICE)

    print_gpu_mem("after skeleton")
    aggressive_cleanup()

    # ── Build lookup maps ─────────────────────────────────────────────────────
    param_map   = dict(policy.named_parameters())
    buffer_map  = dict(policy.named_buffers())
    all_tensors = {**param_map, **buffer_map}
    print(f"  Model has {len(all_tensors)} named tensors (params + buffers)")

    # ── Key prefix detection ──────────────────────────────────────────────────
    print("\nStep 3/5: Detecting checkpoint key alignment...")
    ckpt_strip, model_prefix = _detect_key_prefix(shards, param_map)

    if ckpt_strip:
        remap = {k[len(ckpt_strip):]: v for k, v in all_tensors.items()
                 if not k.startswith(ckpt_strip)}
        remap.update(all_tensors)
        lookup = remap
    elif model_prefix:
        lookup = {(model_prefix + k): v for k, v in all_tensors.items()}
        lookup.update(all_tensors)
    else:
        lookup = all_tensors

    # ── Stream weights shard-by-shard ─────────────────────────────────────────
    print("Step 3/5 (cont.): Streaming checkpoint weights into GPU params...")
    loaded_keys       = set()
    skipped_ckpt_keys = set()

    for shard_path in shards:
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for ck in shard.keys():
                target = None
                for candidate in (
                    ck,
                    ck[len(ckpt_strip):]  if ckpt_strip    else None,
                    model_prefix + ck     if model_prefix   else None,
                ):
                    if candidate and candidate in all_tensors:
                        target = candidate
                        break

                if target is None:
                    skipped_ckpt_keys.add(ck)
                    continue

                cpu_bf16 = shard.get_tensor(ck).to(dtype=MODEL_DTYPE)
                all_tensors[target].data.copy_(cpu_bf16.to(PRIMARY_DEVICE))
                loaded_keys.add(target)
                del cpu_bf16

        aggressive_cleanup()

    # ── Verify coverage ───────────────────────────────────────────────────────
    print("\nStep 4/5: Verifying weight coverage...")
    missing           = set(all_tensors.keys()) - loaded_keys
    expected_absent   = {k for k in missing if _is_known_absent(k)}
    unexpected_absent = missing - expected_absent

    if unexpected_absent:
        print(f"  WARNING: {len(unexpected_absent)} UNEXPECTED missing tensor(s):")
        for k in sorted(unexpected_absent)[:10]:
            print(f"    {k}")
        if len(unexpected_absent) > 10:
            print(f"    ... and {len(unexpected_absent) - 10} more")
    else:
        print("  No unexpected missing tensors")

    if expected_absent:
        print(f"  {len(expected_absent)} flow-matching-head / device-only-buffer "
              f"tensors absent from checkpoint (expected)")

    if skipped_ckpt_keys:
        print(f"  {len(skipped_ckpt_keys)} ckpt key(s) not in model (normal)")

    print(f"  Loaded {len(loaded_keys)} tensors -> GPU bf16")

    if len(loaded_keys) == 0:
        with safe_open(shards[0], framework="pt", device="cpu") as s:
            ckpt_sample_keys = list(s.keys())[:5]
        raise RuntimeError(
            "CRITICAL: 0 tensors loaded.\n"
            f"Checkpoint keys (first 5): {ckpt_sample_keys}\n"
            f"Model keys     (first 5): {list(all_tensors.keys())[:5]}"
        )

    policy = policy.eval()
    for p in policy.parameters():
        p.requires_grad = False

    print(f"  dtype={next(policy.parameters()).dtype}  "
          f"device={next(policy.parameters()).device}")

    # FIX 16(b) again: run the sweep a second time after weight-streaming.
    # Streaming only touches tensors present in all_tensors/loaded via
    # .data.copy_(), so it should not disturb buffer devices, but this is a
    # cheap final check before we start doing real forward passes — much
    # cheaper than discovering a device mismatch mid-extraction on frame 0.
    print("  Re-checking buffer devices after weight streaming...")
    _force_all_buffers_to_device(policy, PRIMARY_DEVICE)

    print_gpu_mem("after model load")

    # ── Build pre/post processors (FIX 1: real tokenization + normalization) ──
    print("\nStep 5/5: Building pre/post processors (tokenizer, normalization)...")
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set, but building the pi0 processors requires "
            "downloading the tokenizer config from google/paligemma-3b-pt-224, "
            "which is a GATED repo on Hugging Face.\n"
            "Fix:\n"
            "  1. Visit https://huggingface.co/google/paligemma-3b-pt-224 while logged "
            "in and click 'Agree and access repository'.\n"
            "  2. Create a read-access token at https://huggingface.co/settings/tokens\n"
            "  3. On Kaggle: Add-ons > Secrets > add HF_TOKEN > enable it for this notebook.\n"
            "  4. Restart the session and re-run."
        )

    try:
        from huggingface_hub import whoami
        _who = whoami(token=HF_TOKEN)
        print(f"  HF identity verified: {_who.get('name', '<unknown>')}")
    except Exception as e:
        raise RuntimeError(
            f"HF_TOKEN is set but failed validation ({type(e).__name__}: {e}). "
            "Check that the token value in Kaggle Secrets is correct and not expired."
        )

    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            MODEL_ID,
            preprocessor_overrides={"device_processor": {"device": PRIMARY_DEVICE}},
        )
    except ValueError as e:
        if "paligemma" in str(e).lower() and "gated" in str(e).lower():
            raise RuntimeError(
                "Failed to load the PaliGemma tokenizer because the gated repo "
                "google/paligemma-3b-pt-224 hasn't been accepted by this HF account "
                "(HF_TOKEN is set and valid, but license acceptance is separate).\n"
                "Fix: visit https://huggingface.co/google/paligemma-3b-pt-224 while "
                "logged in as the SAME account that owns this token, and click "
                "'Agree and access repository'. Then re-run.\n"
                f"Original error: {e}"
            ) from e
        raise
    print("  Processors built.")

    # ── Task instruction lookup table (FIX 2) ─────────────────────────────────
    print("\nLoading task instruction metadata...")
    task_map = load_task_instructions(DATASET_ID, HF_TOKEN or None)
    print(f"  Sample task_index -> instruction: {next(iter(task_map.items()))}")

    # ── Stream dataset (no disk write) ────────────────────────────────────────
    print(f"\nStreaming dataset: {DATASET_ID} ...")
    from datasets import load_dataset

    pool_size = NUM_FRAMES * STREAM_POOL_MULT

    def _stream_with_retry(max_retries=1):
        for attempt in range(max_retries + 1):
            try:
                raw_ds = load_dataset(DATASET_ID, split="train", streaming=True)
                out = []
                for i, sample in enumerate(raw_ds):
                    if i >= pool_size:
                        break
                    out.append(sample)
                return out
            except Exception as e:
                if attempt == max_retries:
                    raise
                print(f"  Streaming attempt {attempt+1} failed ({type(e).__name__}: {e}), retrying...")
                time.sleep(3)

    frames = _stream_with_retry(max_retries=1)

    print(f"  Loaded {len(frames)} frames into pool "
          f"(will subsample {NUM_FRAMES} uniformly across this pool in extract_features)")
    print(f"  Sample keys: {list(frames[0].keys())}")

    for action_key in ("action", "actions"):
        if action_key in frames[0]:
            val = frames[0][action_key]
            print(f"  Action key: '{action_key}', type={type(val).__name__}, "
                  f"value={np.array(val).shape if hasattr(val, '__len__') else val}")
            break
    else:
        print(f"  WARNING: no 'action' or 'actions' key found! Keys: {list(frames[0].keys())}")

    if "task_index" not in frames[0]:
        print("  WARNING: no 'task_index' key found in frames — language "
              "instruction lookup will fail. Check dataset schema.")

    return policy, preprocessor, postprocessor, task_map, frames


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 6 ── FORWARD HOOKS
# ══════════════════════════════════════════════════════════════════════════════

def _find_lm_layers(policy):
    candidates = [
        lambda p: (
            p.model.paligemma_with_expert.paligemma.model.language_model.layers,
            "model.paligemma_with_expert.paligemma.model.language_model.layers"
        ),
        lambda p: (
            p.model.paligemma_with_expert.paligemma.language_model.model.layers,
            "model.paligemma_with_expert.paligemma.language_model.model.layers"
        ),
        lambda p: (
            p.model.paligemma.model.language_model.layers,
            "model.paligemma.model.language_model.layers"
        ),
    ]

    for attempt in candidates:
        try:
            layers, path = attempt(policy)
            if layers is not None and len(layers) > 0:
                return layers, path
        except AttributeError:
            continue

    print("  Standard paths failed — scanning full module tree for LM layers...")
    best_name, best_mod = None, None
    for name, mod in policy.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 10:
            if best_mod is None or len(mod) > len(best_mod):
                best_name, best_mod = name, mod
    if best_mod is not None:
        print(f"  Found candidate ModuleList at '{best_name}' ({len(best_mod)} blocks)")
        return best_mod, best_name

    raise RuntimeError(
        "Cannot find PaliGemma LM layers. "
        "Run inspect_model_tree(policy) to find the correct path."
    )


def register_layer_hooks(policy):
    hidden_states_cache = {li: None for li in PROBE_LAYERS}
    hooks = []

    lm_layers, path = _find_lm_layers(policy)
    num_layers = len(lm_layers)
    print(f"  PaliGemma LM: {num_layers} layers at '{path}'")

    def make_hook(layer_idx):
        def hook_fn(module, input, kwargs, output):
            hs = output[0] if isinstance(output, tuple) else output
            last_valid_idx = hs.shape[1] - 1

            mask = kwargs.get("attention_mask", None)
            if mask is None and len(input) > 1 and isinstance(input[1], torch.Tensor):
                mask = input[1]

            if mask is not None:
                try:
                    if mask.dim() == 2:
                        valid = torch.nonzero(mask[0] > 0).flatten()
                        if len(valid) > 0:
                            last_valid_idx = valid[-1].item()
                    elif mask.dim() == 4:
                        row_sums = mask[0, 0, :, :].float().sum(dim=-1)
                        valid = torch.nonzero(row_sums > -1e6).flatten()
                        if len(valid) > 0:
                            last_valid_idx = valid[-1].item()
                except Exception:
                    pass

            hidden_states_cache[layer_idx] = (
                hs[:, last_valid_idx, :].detach().float().cpu()
            )
        return hook_fn

    for li in PROBE_LAYERS:
        abs_idx = num_layers + li
        hook = lm_layers[abs_idx].register_forward_hook(make_hook(li), with_kwargs=True)
        hooks.append(hook)
        print(f"  Hook: layer {abs_idx} (relative {li})")

    return hidden_states_cache, hooks


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 7 ── EXTRACT FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def _pil_to_chw_float_tensor(pil_img, device):
    """
    Convert a PIL Image to a [0,1]-range float32 CHW tensor on `device` — the
    standard LeRobotDataset image convention. _preprocess_images() inside pi0
    itself handles resizing to 224x224 and rescaling [0,1] -> [-1,1]
    internally, so we only need to get as far as a plain [0,1] float CHW
    tensor on the right device here; do NOT resize or renormalize further, or
    you'll double-apply what the model does.
    """
    arr = np.asarray(pil_img, dtype=np.uint8).copy()   # HWC, uint8, [0,255]
    # NOTE: .copy() above avoids the "given NumPy array is not writable"
    # UserWarning seen in the previous run (PIL images can return read-only
    # array views) — harmless before, but fixed for cleanliness.
    if arr.ndim == 2:                                   # grayscale safety net
        arr = arr[:, :, None]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW, uint8, CPU
    return t.float().div(255.0).to(device)                    # CHW, float32, [0,1], on device


def _build_raw_batch(sample, task_map, device):
    """
    Build the raw (unbatched) observation dict that the real LeRobot
    preprocessor expects.

    FIX 13 — CAMERA KEY RENAMING: renames LIBERO's native `image`/
    `wrist_image` keys to pi0_base's expected `base_0_rgb`/`left_wrist_0_rgb`
    slot names (see full rationale in the version history above).

    FIX 14 — PIL/list -> TENSOR CONVERSION: converts raw PIL images and
    Python-list state into torch tensors, since model._preprocess_images()
    assumes tensors already (only handles device moves/resizing internally,
    not PIL decoding).

    FIX 15 — EXPLICIT DEVICE PLACEMENT: image/state tensors are built
    directly on `device` (the same PRIMARY_DEVICE the model was constructed
    on), rather than relying on the preprocessor to move them for us — the
    preprocessor's device_processor only moves the tokenized language
    fields, not arbitrary extra batch keys we add ourselves.

    NOTE (FIX 16 context): the device error you previously saw at this layer
    ("index is on cpu, different from other tensors on cuda:0") was NOT
    actually caused by this function — raw_batch/policy_input tensors here
    were already correctly placed. It was caused by an internal SigLIP
    `position_ids` buffer never being moved to GPU during model construction.
    See FIX 16 in the module docstring and gpu_alloc_monkeypatch /
    _force_all_buffers_to_device above for the actual fix.

    FIX 17 (NEW) — state dtype mismatch:
    `observation.state` was previously hardcoded to float32. That's fine for
    the vision tower (SigLIP/PaliGemma upcast pixel_values internally before
    their own forward, so fp32 in is tolerated), but pi0's action-expert
    `state_proj` is a bare nn.Linear with no internal dtype cast — it just
    calls F.linear(input, self.weight, self.bias) directly. Since
    state_proj.weight was streamed in as bf16 (MODEL_DTYPE), handing it a
    fp32 activation raised:
        RuntimeError: mat1 and mat2 must have the same dtype, but got Float
        and BFloat16
    Fix: build `observation.state` in MODEL_DTYPE (bf16) up front, matching
    every other weight tensor in the model, instead of hardcoding float32.
    """
    task_index = sample.get("task_index")
    if task_index is None:
        instruction = "perform the task"  # last-resort fallback, should not trigger normally
    else:
        instruction = task_map.get(int(task_index), "perform the task")

    state_raw = sample["observation.state"]
    if isinstance(state_raw, torch.Tensor):
        # FIX 17: cast to MODEL_DTYPE (bf16), not a hardcoded float32 — must
        # match state_proj.weight's dtype or F.linear raises a dtype-mismatch
        # RuntimeError (state_proj has no internal autocast, unlike the
        # vision tower which upcasts pixel_values internally).
        state_tensor = state_raw.to(dtype=MODEL_DTYPE, device=device)
    else:
        state_tensor = torch.tensor(state_raw, dtype=MODEL_DTYPE, device=device)

    raw_batch = {
        "observation.images.base_0_rgb":       _pil_to_chw_float_tensor(sample["observation.images.image"], device),
        "observation.images.left_wrist_0_rgb": _pil_to_chw_float_tensor(sample["observation.images.wrist_image"], device),
        "observation.state":                   state_tensor,
        "task":                                instruction,
    }
    return raw_batch


def extract_features(policy, preprocessor, frames, task_map, hidden_states_cache):
    """
    Run frozen pi0 forward passes over frames via the REAL select_action path.

    FIX 3: policy.reset() is called before every frame to clear select_action's
    internal action-chunk queue, so every single frame forces a genuine forward
    pass through the model (and therefore fires the registered hooks). Without
    this, most frames would just pop a cached action from the queue with NO
    forward pass and NO hook firing, silently producing far fewer valid samples
    than frames processed.
    """
    device = next(policy.parameters()).device

    total_pool = len(frames)
    n_target   = min(NUM_FRAMES, total_pool)
    indices    = np.linspace(0, total_pool - 1, n_target, dtype=int)

    all_layer_feats = {li: [] for li in PROBE_LAYERS}
    all_actions     = []
    all_proprio     = []   # NEW
    all_episode_idx = []   # NEW
    all_task_idx    = []   # NEW
    failed_frames   = 0

    print(f"\nExtracting features: {len(indices)} frames from pool of {total_pool}...")
    print_gpu_mem("before extraction")

    first_frame_debug_printed = False

    with torch.no_grad():
        for i, raw_idx in enumerate(indices):
            idx = int(raw_idx)

            if i % 100 == 0:
                print(f"  Frame {i}/{len(indices)}  (pool idx {idx})")
            if i > 0 and i % CHECKPOINT_EVERY == 0:
                save_checkpoint(all_layer_feats, all_actions, i)
                print_gpu_mem(f"frame {i}")

            # ── 1. Clear cache ────────────────────────────────────────────────
            for li in PROBE_LAYERS:
                hidden_states_cache[li] = None

            # ── 2. Get sample ─────────────────────────────────────────────────
            try:
                sample = frames[idx]
            except Exception:
                if i == 0:
                    raise
                failed_frames += 1
                continue

            # ── 3. Build raw batch + resolve language instruction ─────────────
            try:
                raw_batch = _build_raw_batch(sample, task_map, device)
            except Exception as e:
                if i == 0:
                    raise RuntimeError(f"Failed to build raw batch on first frame: {e}")
                failed_frames += 1
                continue

            # ── 4. Preprocess (tokenize + normalize) ───────────────────────────
            try:
                policy_input = preprocessor(raw_batch)
            except Exception as e:
                if i == 0:
                    raise RuntimeError(
                        f"Preprocessor failed on first frame: {e}\n"
                        f"raw_batch keys: {list(raw_batch.keys())}"
                    )
                failed_frames += 1
                continue

            if not first_frame_debug_printed:
                print("\n  --- First-frame preprocessor output (verify before trusting run) ---")
                for k, v in policy_input.items():
                    if hasattr(v, "shape"):
                        print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype} "
                              f"device={getattr(v, 'device', 'n/a')}")
                    else:
                        print(f"    {k}: {type(v).__name__} = {v!r}")
                print("  --- end first-frame debug ---\n")
                first_frame_debug_printed = True

            # ── 5. Clear the action queue, then run select_action ──────────────
            # (FIX 3: forces a genuine forward pass -> hooks fire -> cache filled)
            #
            # FIX 18: select_action() is wrapped in torch.autocast(bf16). This
            # is necessary because sample_actions()/denoise_step() internally
            # generates its own diffusion noise tensor `x_t` (via something
            # like torch.randn(..., device=...) with no explicit dtype) when
            # we don't pass a `noise=` kwarg ourselves. That tensor defaults
            # to float32 and is fed straight into `action_in_proj`, a bf16
            # nn.Linear, which raises the same "mat1 and mat2 must have the
            # same dtype" error FIX 17 fixed for `state_proj` — except this
            # time the offending tensor isn't one we build in
            # _build_raw_batch(), so we can't point-fix it the same way.
            # autocast casts fp32 activations to bf16 automatically at
            # matmul/linear op boundaries; it does not alter the dtype of the
            # already-streamed bf16 parameters, so this is safe to combine
            # with the manual bf16 weight-streaming done in
            # load_model_and_dataset().
            try:
                policy.reset()
                with torch.autocast(device_type="cuda", dtype=MODEL_DTYPE):
                    policy.select_action(policy_input)
                success = True
            except Exception as e:
                success = False
                if i == 0:
                    err_str = str(e).lower()
                    hint = (
                        "If this is a device-mismatch error mentioning 'position_ids' or "
                        "an internal buffer being on cpu, see FIX 16 — run "
                        "_force_all_buffers_to_device(policy, PRIMARY_DEVICE) again and "
                        "re-check that gpu_alloc_monkeypatch's torch.arange patch is active.\n"
                        "If this is a dtype-mismatch error ('must have the same dtype', "
                        "Float vs BFloat16) that STILL occurs even with the FIX 18 autocast "
                        "wrapper active, it likely means a tensor is passed to an op autocast "
                        "doesn't cover (e.g. an elementwise add/sub outside autocast's "
                        "op allowlist, or a tensor created/consumed inside a context that "
                        "exited the `with torch.autocast(...)` block early). Print "
                        "policy_input dtypes above and compare against the dtype of the "
                        "nn.Linear/nn.Conv weight named in the traceback (e.g. state_proj, "
                        "action_in_proj, action_out_proj) to isolate exactly which tensor "
                        "still needs an explicit .to(MODEL_DTYPE) cast at its creation site.\n"
                        "Otherwise run inspect_model_tree(policy) to check architecture, or "
                        "verify the preprocessor output keys above match what "
                        "predict_action_chunk() expects for your installed lerobot version."
                    )
                    raise RuntimeError(
                        f"select_action failed on first frame: {type(e).__name__}: {e}\n"
                        f"policy_input keys: {list(policy_input.keys())}\n"
                        f"{hint}"
                    )

            if not success:
                failed_frames += 1
                continue

            # ── 6. Collect hidden states from cache ───────────────────────────
            captured = True
            row_hs   = []
            for li in PROBE_LAYERS:
                hs = hidden_states_cache[li]
                if hs is None:
                    captured = False
                    break
                row_hs.append(hs.squeeze(0).numpy())

            if not captured:
                if i == 0:
                    raise RuntimeError(
                        "Hooks did not fire on first frame even though select_action "
                        "succeeded. The forward pass may be bypassing the hooked "
                        "layer path (e.g. cached KV path). Check _find_lm_layers() "
                        "and confirm register_layer_hooks() found the right module."
                    )
                failed_frames += 1
                continue

            # ── 7. Collect ground-truth action ────────────────────────────────
            action_raw = sample.get("action", sample.get("actions"))

            if action_raw is None:
                if i == 0:
                    print(f"  WARNING: no action key. Sample keys: {list(sample.keys())}")
                failed_frames += 1
                continue

            if isinstance(action_raw, torch.Tensor):
                action = action_raw.numpy()
            elif isinstance(action_raw, np.ndarray):
                action = action_raw
            else:
                try:
                    action = np.array(action_raw, dtype=np.float32)
                except Exception:
                    failed_frames += 1
                    continue

            if action.ndim == 0:
                action = action.reshape(1)
            elif action.ndim > 1:
                action = action[0]

            # ── 7b. Collect proprio + episode/task indices (NEW) ────────────────
            # Pulled from the SAME `sample` dict, same iteration, so alignment
            # with `action` and `row_hs` is guaranteed by construction --
            # no separate re-streaming pass, no risk of the Exp1c-style
            # cross-run misalignment bug.
            state_raw = sample.get("observation.state")
            if state_raw is None:
                if i == 0:
                    print(f"  WARNING: no observation.state key. Sample keys: {list(sample.keys())}")
                failed_frames += 1
                continue
            if isinstance(state_raw, torch.Tensor):
                proprio = state_raw.float().cpu().numpy()
            else:
                proprio = np.array(state_raw, dtype=np.float32)
            if proprio.ndim > 1:
                proprio = proprio[0]

            episode_idx = int(sample.get("episode_index", -1))
            task_idx    = int(sample.get("task_index", -1))
            
            # ── 8. Commit this frame ──────────────────────────────────────────
            for li, hs in zip(PROBE_LAYERS, row_hs):
                all_layer_feats[li].append(hs)
            all_actions.append(action)
            all_proprio.append(proprio)          # NEW
            all_episode_idx.append(episode_idx)  # NEW
            all_task_idx.append(task_idx) 

    n_samples = len(all_actions)
    print(f"\n  Extraction complete: {n_samples} samples collected, "
          f"{failed_frames} skipped")

    if n_samples == 0:
        raise RuntimeError(
            "No samples collected. Check the first-frame debug printout above — "
            "most likely cause is a mismatch between preprocessor output keys and "
            "what predict_action_chunk()/sample_actions() expects for your "
            "installed lerobot version, or hooks not firing on the actual forward path."
        )

    # FIX 19(a) — explicit failed_frames threshold check:
    # A large skip fraction means the R2 you compute later is being fit on a
    # biased subsample (whatever frames happened to succeed), not a
    # representative sample of the streamed pool. Silently proceeding on,
    # say, 60% failures would make the final R2 essentially meaningless
    # without the reader knowing to distrust it. This makes that risk loud
    # and unmissable instead of something you have to notice buried in logs.
    total_attempted = n_samples + failed_frames
    fail_frac = failed_frames / total_attempted if total_attempted > 0 else 0.0
    print(f"  Failure rate  : {fail_frac*100:.1f}% ({failed_frames}/{total_attempted} attempted)")
    if fail_frac > 0.30:
        print(
            f"  \033[91mWARNING: {fail_frac*100:.1f}% of attempted frames failed to "
            f"produce a sample. The resulting R2 is fit on a biased subsample — "
            f"whatever frames happened to succeed, not a representative draw from "
            f"the streamed pool. Investigate the failure cause (re-run with a small "
            f"NUM_FRAMES and inspect which exceptions are being swallowed by the "
            f"try/except blocks in this function) before trusting downstream "
            f"results.\033[0m"
        )
    elif fail_frac > 0.10:
        print(
            f"  NOTE: {fail_frac*100:.1f}% failure rate is non-trivial. Results are "
            f"probably still usable but double-check nothing systematic (e.g. one "
            f"particular task_index or episode range) is being dropped."
        )

    actions_array = np.stack(all_actions, axis=0)
    proprio_array  = np.stack(all_proprio, axis=0)              # NEW
    episode_array  = np.array(all_episode_idx)                  # NEW
    task_array     = np.array(all_task_idx)
    layer_arrays  = {}

    for li in PROBE_LAYERS:
        feats = all_layer_feats[li]
        if len(feats) == n_samples:
            layer_arrays[li] = np.stack(feats, axis=0)
        else:
            print(f"  WARNING: Layer {li} has {len(feats)} samples vs "
                  f"{n_samples} actions — skipping layer")

    valid_layers = [li for li in PROBE_LAYERS if li in layer_arrays]
    if len(valid_layers) > 1:
        layer_arrays["avg"] = np.mean(
            [layer_arrays[li] for li in valid_layers], axis=0
        )

    # FIX 19(b) — explicit NaN/Inf audit on the collected arrays.
    # bf16 forward passes (especially with autocast, FIX 18) can silently
    # produce NaN/Inf in edge cases (e.g. overflow in intermediate bf16
    # accumulation) that won't raise an exception — they'll just poison the
    # Ridge regression later with no clear error message pointing back here.
    # Check now, right after collection, so a corrupted run is caught at the
    # source instead of surfacing as a confusing sklearn warning or a
    # nonsensical R2 several cells later.
    def _nan_inf_report(name, arr):
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        if n_nan or n_inf:
            print(f"  \033[91mWARNING: {name} contains {n_nan} NaN and {n_inf} Inf "
                  f"value(s) out of {arr.size} total. This will corrupt the Ridge fit "
                  f"downstream. Likely cause: bf16 overflow during the forward pass, "
                  f"or a genuinely bad frame that slipped past the per-frame "
                  f"try/except. Consider filtering these rows out before "
                  f"train_probes(), or re-running with float32 to see if the issue "
                  f"is precision-related.\033[0m")
            return True
        return False

    any_bad = _nan_inf_report("actions_array", actions_array)
    for li in valid_layers:
        any_bad = _nan_inf_report(f"layer_arrays[{li}]", layer_arrays[li]) or any_bad
    if not any_bad:
        print("  NaN/Inf check: clean — no NaN or Inf values in collected actions or features")

    print(f"  Action shape  : {actions_array.shape}")
    if valid_layers:
        print(f"  Embedding dim : {layer_arrays[valid_layers[0]].shape[1]}")
    print_gpu_mem("after extraction")

    assert proprio_array.shape[0] == actions_array.shape[0], "proprio/action count mismatch!"
    assert episode_array.shape[0] == actions_array.shape[0], "episode_index/action count mismatch!"
    assert task_array.shape[0] == actions_array.shape[0], "task_index/action count mismatch!"

    return layer_arrays, actions_array, proprio_array, episode_array, task_array


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 8 ── TRAIN LINEAR PROBES
# ══════════════════════════════════════════════════════════════════════════════

def train_probes(layer_arrays, actions_array):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split

    results = {}
    n = len(actions_array)

    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=42)
    actions_train = actions_array[train_idx]
    actions_test  = actions_array[test_idx]

    act_dim   = actions_array.shape[1]
    dim_names = ACTION_DIM_NAMES[:act_dim]
    if len(dim_names) < act_dim:
        dim_names += [f"dim_{i}" for i in range(len(dim_names), act_dim)]

    layer_labels = {
        -1:    "Layer -1 (last)",
        -2:    "Layer -2",
        -3:    "Layer -3",
        -4:    "Layer -4",
        "avg": "Avg last 4 layers",
    }

    print("\n" + "=" * 60)
    print("PROBE RESULTS")
    print("=" * 60)

    for key in PROBE_LAYERS + ["avg"]:
        if key not in layer_arrays:
            continue

        X = layer_arrays[key]
        X_train, X_test = X[train_idx], X[test_idx]

        scaler     = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        probe = Ridge(alpha=RIDGE_ALPHA)
        probe.fit(X_train_sc, actions_train)
        preds = probe.predict(X_test_sc)

        r2_per_dim  = r2_score(actions_test, preds, multioutput="raw_values")
        r2_overall  = r2_score(actions_test, preds)
        mse_overall = mean_squared_error(actions_test, preds)

        label = layer_labels.get(key, str(key))
        results[key] = {
            "label":       label,
            "r2_per_dim":  r2_per_dim,
            "r2_overall":  r2_overall,
            "mse_overall": mse_overall,
            "probe":       probe,
            "scaler":      scaler,
            "dim_names":   dim_names,
        }

        verdict = (
            "PASS"    if r2_overall > 0.7 else
            "PARTIAL" if r2_overall > 0.4 else
            "FAIL"
        )
        print(f"\n{label}")
        print(f"  Overall R2:  {r2_overall:.4f}  [{verdict}]")
        print(f"  Overall MSE: {mse_overall:.6f}")
        print("  Per-dim R2:")
        for dname, r2 in zip(dim_names, r2_per_dim):
            bar = "#" * int(max(0, r2) * 20)
            print(f"    {dname:10s}: {r2:+.4f}  |{bar}")

    # Random baseline
    print("\n--- BASELINES ---")
    rng       = np.random.RandomState(42)
    feat_dim  = layer_arrays[PROBE_LAYERS[0]].shape[1]
    X_rand_tr = rng.randn(len(train_idx), feat_dim)
    X_rand_te = rng.randn(len(test_idx),  feat_dim)
    probe_rand = Ridge(alpha=RIDGE_ALPHA)
    probe_rand.fit(X_rand_tr, actions_train)
    r2_rand = r2_score(actions_test, probe_rand.predict(X_rand_te))
    print(f"\nRandom Gaussian features -> R2: {r2_rand:.4f}")
    results["random"] = {"label": "Random baseline", "r2_overall": r2_rand}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 9 ── PLOT & SAVE
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Experiment 1: Linear Decodability of pi0 VLM Hidden States -> Actions",
        fontsize=13, fontweight="bold"
    )

    ax        = axes[0]
    layer_keys = [k for k in PROBE_LAYERS + ["avg"] if k in results]
    labels    = [results[k]["label"]      for k in layer_keys]
    r2_scores = [results[k]["r2_overall"] for k in layer_keys]
    colors    = [
        "#2ecc71" if r > 0.7 else "#f39c12" if r > 0.4 else "#e74c3c"
        for r in r2_scores
    ]

    bars = ax.barh(labels, r2_scores, color=colors, edgecolor="white", height=0.5)
    ax.axvline(0.7, color="green",  linestyle="--", alpha=0.7, label="Pass (0.7)")
    ax.axvline(0.4, color="orange", linestyle="--", alpha=0.7, label="Partial (0.4)")
    rand_r2 = results.get("random", {}).get("r2_overall", 0)
    ax.axvline(rand_r2, color="red", linestyle=":", alpha=0.7, label="Random baseline")
    for bar, score in zip(bars, r2_scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=10)
    ax.set_xlim(-0.1, 1.05)
    ax.set_xlabel("R2 Score")
    ax.set_title("Overall R2 by Layer")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax      = axes[1]
    best_key = max(
        [k for k in PROBE_LAYERS + ["avg"] if k in results],
        key=lambda k: results[k]["r2_overall"]
    )
    r2_dims   = results[best_key]["r2_per_dim"]
    dim_names = results[best_key].get("dim_names", ACTION_DIM_NAMES[:len(r2_dims)])
    colors_d  = [
        "#2ecc71" if r > 0.7 else "#f39c12" if r > 0.4 else "#e74c3c"
        for r in r2_dims
    ]

    ax.bar(dim_names[:len(r2_dims)], r2_dims, color=colors_d, edgecolor="white")
    ax.axhline(0.7, color="green",  linestyle="--", alpha=0.7, label="Pass (0.7)")
    ax.axhline(0.4, color="orange", linestyle="--", alpha=0.7)
    ax.axhline(0.0, color="black",  linestyle="-",  alpha=0.3)
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel("R2 Score")
    ax.set_title(f"Per-Dimension R2\n({results[best_key]['label']})")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    save_path = SAVE_DIR / "probe_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {save_path}")

    try:
        from IPython.display import display, Image
        display(Image(str(save_path)))
    except Exception:
        pass


def save_summary(results):
    summary_path = SAVE_DIR / "probe_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("EXPERIMENT 1: LINEAR DECODABILITY PROBE\n")
        f.write(f"Model  : {MODEL_ID}\n")
        f.write(f"Dataset: {DATASET_ID}\n")
        f.write(f"Frames : {NUM_FRAMES}\n")
        f.write("=" * 60 + "\n\n")

        for key in PROBE_LAYERS + ["avg"]:
            if key not in results:
                continue
            r = results[key]
            verdict = (
                "PASS — linear decoder valid"    if r["r2_overall"] > 0.7 else
                "PARTIAL — consider MLP decoder" if r["r2_overall"] > 0.4 else
                "FAIL — geometry not action-relevant"
            )
            f.write(f"{r['label']}\n")
            f.write(f"  R2 overall : {r['r2_overall']:.4f}  [{verdict}]\n")
            f.write(f"  MSE overall: {r['mse_overall']:.6f}\n")
            if "r2_per_dim" in r:
                dim_names = r.get("dim_names", ACTION_DIM_NAMES)
                dims = ", ".join(
                    f"{n}={v:.3f}" for n, v in zip(dim_names, r["r2_per_dim"])
                )
                f.write(f"  Per-dim R2 : {dims}\n")
            f.write("\n")

        f.write(f"Random baseline R2: "
                f"{results.get('random', {}).get('r2_overall', 'N/A')}\n")

    print(f"Summary saved -> {summary_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 10 ── MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Experiment 1: Linear Decodability Probe")
    print("pi0 VLM Hidden States -> Robot Actions")
    print(f"Device: {PRIMARY_DEVICE}  |  dtype: {MODEL_DTYPE}  |  frames: {NUM_FRAMES}")
    print("=" * 60 + "\n")

    policy, preprocessor, postprocessor, task_map, frames = load_model_and_dataset()

    print("\nRegistering layer hooks...")
    hidden_states_cache, hooks = register_layer_hooks(policy)

    # FIX: extract_features now returns 5 values (proprio/episode/task arrays
    # were added alongside layer_arrays/actions_array), so unpack all 5 —
    # the previous 2-value unpack is what raised "too many values to unpack".
    layer_arrays, actions_array, proprio_array, episode_array, task_array = extract_features(
        policy, preprocessor, frames, task_map, hidden_states_cache
    )

    for hook in hooks:
        hook.remove()
    print("  Hooks removed.")

    del policy
    aggressive_cleanup()
    print_gpu_mem("after freeing model")

    features_path = SAVE_DIR / "features.npz"
    save_dict = {f"layer_{k}": v for k, v in layer_arrays.items()}
    save_dict["actions"] = actions_array
    save_dict["proprio"]       = proprio_array
    save_dict["episode_index"] = episode_array
    save_dict["task_index"]    = task_array
    np.savez(features_path, **save_dict)
    print(f"\nFeatures saved -> {features_path}")

    results = train_probes(layer_arrays, actions_array)
    plot_results(results)
    save_summary(results)

    valid_keys = [k for k in PROBE_LAYERS if k in results]
    best_r2    = max(results[k]["r2_overall"] for k in valid_keys)

    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    if best_r2 > 0.7:
        print(f"PASS  (best R2 = {best_r2:.4f})")
        print("   Linear decoder assumption is valid.")
        print("   Proceed to Experiment 2 (embedding geometry check).")
    elif best_r2 > 0.4:
        print(f"PARTIAL  (best R2 = {best_r2:.4f})")
        print("   Partial linearity. Replace linear decoder with small MLP.")
        print("   Core idea survives but architecture needs adjustment.")
    else:
        print(f"FAIL  (best R2 = {best_r2:.4f})")
        print("   VLM hidden states do not linearly encode action-relevant structure.")
        print("   Central geometric assumption needs rethinking.")
    print("=" * 60)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 11 ── RE-RUN PROBES FROM SAVED FEATURES (no GPU needed)
# ══════════════════════════════════════════════════════════════════════════════

def run_from_saved(features_path="/kaggle/working/probe_results/features.npz"):
    print(f"Loading saved features from {features_path}...")
    data = np.load(features_path)

    actions_array = data["actions"]
    layer_arrays  = {}
    for key in data.files:
        if key.startswith("layer_"):
            suffix = key[len("layer_"):]
            try:
                layer_arrays[int(suffix)] = data[key]
            except ValueError:
                layer_arrays[suffix] = data[key]

    print(f"  {len(actions_array)} samples, {len(layer_arrays)} layer arrays\n")

    results = train_probes(layer_arrays, actions_array)
    plot_results(results)
    save_summary(results)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 12 ── DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def inspect_model_tree(policy, max_depth=5):
    """Print the named module tree. Run if register_layer_hooks() fails."""
    print(f"pi0 module tree (depth <= {max_depth}):\n")
    for name, module in policy.named_modules():
        depth = name.count(".")
        if depth > max_depth:
            continue
        indent = "  " * depth
        print(f"{indent}{name or '[root]'}  ({type(module).__name__})")


def inspect_key_mismatch(policy, shard_path):
    from safetensors import safe_open

    with safe_open(shard_path, framework="pt", device="cpu") as s:
        ckpt_keys = list(s.keys())

    model_keys = list(dict(policy.named_parameters()).keys())

    print("=== CHECKPOINT keys (first 10) ===")
    for k in ckpt_keys[:10]:
        print(f"  {k}")

    print("\n=== MODEL keys (first 10) ===")
    for k in model_keys[:10]:
        print(f"  {k}")

    ckpt_set  = set(ckpt_keys)
    model_set = set(model_keys)
    print(f"\nDirect intersection: {len(ckpt_set & model_set)} keys")

    stripped = {k[len("model."):] if k.startswith("model.") else k for k in ckpt_set}
    print(f"After stripping 'model.' from ckpt: {len(stripped & model_set)} keys match")

    prefixed = {"model." + k for k in ckpt_set}
    print(f"After adding 'model.' to ckpt:      {len(prefixed & model_set)} keys match")


def inspect_frames(frames, n=3):
    """Print sample info to debug key names and action format."""
    print(f"Inspecting {n} frames from {len(frames)} total:\n")
    for i in range(min(n, len(frames))):
        s = frames[i]
        print(f"Frame {i}:")
        for k, v in s.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {type(v).__name__} shape={v.shape} dtype={v.dtype}")
            elif hasattr(v, '__len__'):
                arr = np.array(v)
                print(f"  {k}: {type(v).__name__} len={len(v)} -> array shape={arr.shape}")
            else:
                print(f"  {k}: {type(v).__name__} = {v}")
        print()


def inspect_task_map(task_map, n=10):
    """Print the first n task_index -> instruction mappings."""
    print(f"Task map has {len(task_map)} entries:")
    for i, (k, v) in enumerate(sorted(task_map.items())):
        if i >= n:
            break
        print(f"  {k}: {v}")


def inspect_buffer_devices(policy):
    """
    FIX 16 diagnostic helper: prints every named buffer and its device, so you
    can quickly confirm nothing is stranded on CPU after model construction.
    Run this right after load_model_and_dataset() if you ever see a device-
    mismatch error again, to see exactly which buffer(s) are misplaced before
    reaching for _force_all_buffers_to_device().
    """
    print("Named buffers and their devices:\n")
    off_device = []
    target = next(policy.parameters()).device
    for name, buf in policy.named_buffers():
        marker = "" if buf.device == target else "  <-- MISMATCH"
        if marker:
            off_device.append(name)
        print(f"  {name}: device={buf.device}, dtype={buf.dtype}, shape={tuple(buf.shape)}{marker}")
    if off_device:
        print(f"\n{len(off_device)} buffer(s) NOT on {target}: {off_device}")
    else:
        print(f"\nAll buffers on {target}.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()