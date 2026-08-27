# Exp 3 — Flow Trajectory Quality

Tests whether running a rectified flow through the VLM embedding space
produces geometrically simpler (straighter) trajectories than running the
same flow in raw action space — independent of the embedding-collapse
question in Exp 2.

## Run order

- `00_flow_trajectory_quality_main.py` — self-contained: the actual
  trajectory-geometry experiment (synthetic 2D task, embedding-space vs.
  raw-action-space conditions). Run this on its own; it doesn't depend on
  the numbered files below.

The numbered files below are a **separate, later debugging thread** that
trains a conditional rectified flow on real Exp1a features and diagnoses
why its R2 was initially poor. They must be run **in order, in the same
session** — each reuses variables/models defined by the previous one:

1. `01_load_features_and_setup.py` — loads Exp1a features, trains an
   action encoder/decoder, trains a first (unstandardized) conditional
   flow, reports R2.
2. `02_isolate_flow_vs_decode_failure.py` — diagnostic: run right after
   01, before 03. Checks whether 01's poor R2 comes from the flow or the
   decode step.
3. `03_standardize_embeddings.py` — the fix: standardizes target
   embeddings to the noise-prior scale and retrains the flow with
   mini-batching.
4. `04_context_to_embedding_mlp.py` — adds a plain regression baseline
   (context → embedding, no flow) to compare against.
5. `05_multilayer_feature_variant.py` — standalone (own data, own
   imports). A follow-up variant repeating the self- vs. real-conditioned
   flow comparison across multiple VLM feature layers instead of one.
