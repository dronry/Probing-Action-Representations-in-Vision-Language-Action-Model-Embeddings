# NOTE: not fully standalone -- loads a pre-computed .npz path specific
# to the original Kaggle dataset. Update the path before reuse. This is
# the cached alternative to alt_extract_vocab_embeddings_from_shards.py
# (bypasses re-extracting from checkpoint shards).

import numpy as np

def extract_vocab_embeddings():
    print("Using pre-verified vocab embedding table (bypassing broken in-script extraction)")
    arr = np.load("/kaggle/input/datasets/tahak9/boyfla/features.npz")
    print(f"  Loaded shape: {arr.shape}")
    assert arr.shape[1] == 2048, f"Unexpected dim {arr.shape[1]}"
    return arr