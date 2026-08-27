# NOTE: not standalone. Extracts the vocab embedding table directly from
# the pi0/PaliGemma checkpoint shards. Requires `shards` (a list of
# safetensors shard paths) and `safe_open` (from `safetensors`) already
# available in the session -- see main_encoder_geometry_check.py for the
# checkpoint-loading context this assumes.
#
# This is one of two ways to get the vocab embedding table; the other is
# alt_load_vocab_embeddings_cached.py, which loads a pre-verified copy
# instead of re-extracting from shards.

import numpy as np

for shard_path in shards:
    with safe_open(shard_path, framework="pt", device="cpu") as shard:
        for k in shard.keys():
            if k.endswith("paligemma.lm_head.weight") and "gemma_expert" not in k:
                tensor = shard.get_tensor(k)
                assert tensor.shape[1] == 2048, f"Wrong table! Got dim {tensor.shape[1]}"
                np.save("/kaggle/working/probe_results/vocab_embeddings_verified.npy", tensor.float().numpy())
                print("Saved verified table, shape:", tuple(tensor.shape))