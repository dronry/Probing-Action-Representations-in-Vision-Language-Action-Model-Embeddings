# Exp 2 — Encoder Geometry Check

Tests whether newly-trained action-token embeddings (from an encoder trained
to map actions into the VLM's embedding space) naturally organize near real
language-token embeddings, or collapse into a degenerate region.

## Files

- `main_encoder_geometry_check.py` — the experiment itself.
- `alt_extract_vocab_embeddings_from_shards.py` — one way to obtain the
  vocab embedding table used as a geometric reference: extracts it directly
  from checkpoint safetensors shards. Not standalone (needs `shards` and
  `safe_open` from the checkpoint-loading context in the main script).
- `alt_load_vocab_embeddings_cached.py` — the other way: loads a
  pre-verified copy of the same table from a cached `.npz`. Use this if
  you've already run the extraction once and saved the result, or don't
  have the raw checkpoint shards available.

Pick whichever of the two `alt_*` files fits your setup; you don't need
both.
