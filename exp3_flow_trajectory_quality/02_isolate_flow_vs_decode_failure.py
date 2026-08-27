# NOTE: not standalone. Run in the same session right after
# 01_load_features_and_setup.py, BEFORE 03_standardize_embeddings.py
# (which re-inits and overwrites `flow`). Reuses `flow`,
# `generate_action_token`, `ctx_val`, `emb_val_target`, `EMB_DIM` from
# 01's globals.
#
# Diagnostic: isolates whether Exp3's low R2 comes from the flow itself
# failing to match the target embedding distribution, or from the
# decode step -- by checking embedding-space error before decoding.

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- STEP 1: Isolate the failure -- check embedding-space error BEFORE decoding ----
# This tells us if the flow itself failed, or if decode is the problem.
with torch.no_grad():
    gen_emb_val = generate_action_token(flow, ctx_val.to(device), steps=32)
    true_emb_val = emb_val_target  # from Stage 1 encoder, already computed

emb_mse = F.mse_loss(gen_emb_val, true_emb_val.to(device)).item()
emb_r2 = r2_score(true_emb_val.cpu().numpy(), gen_emb_val.cpu().numpy())
print(f"Embedding-space MSE (gen vs true target emb): {emb_mse:.4f}")
print(f"Embedding-space R2  (gen vs true target emb): {emb_r2:.4f}")

# Trivial baselines to know what "no signal learned" looks like
trivial_zero = torch.zeros_like(true_emb_val)
trivial_mean = true_emb_val.mean(dim=0, keepdim=True).expand_as(true_emb_val)
print(f"Baseline (predict 0) embedding R2: "
    f"{r2_score(true_emb_val.detach().cpu().numpy(), trivial_zero.detach().cpu().numpy()):.4f}"
)

print(f"Baseline (predict mean) embedding R2: "
    f"{r2_score(true_emb_val.detach().cpu().numpy(), trivial_mean.detach().cpu().numpy()):.4f}"
)

# Scale check -- are noise and target embeddings even comparable magnitude?
print(f"\nTarget embedding per-dim std (mean over dims): {true_emb_val.std(dim=0).mean().item():.4f}")
print(f"Target embedding norm (mean over samples): {true_emb_val.norm(dim=1).mean().item():.4f}")
print(f"Standard normal noise expected norm (sqrt(2048)): {np.sqrt(EMB_DIM):.4f}")