# NOTE: not standalone. Run in the same session right after
# 01_load_features_and_setup.py -- reuses `emb_train_target`,
# `emb_val_target`, `device`, `EMB_DIM`, `ConditionalVelocityField`,
# `ctx_train`, `ctx_val`, `decoder`, `act_val`, `np` from that script.
#
# Iteration on 01: standardizes target embeddings to the noise-prior
# scale and retrains the flow with mini-batching (the first attempt in
# 01 suffered gradient starvation from full-batch training on
# unstandardized targets).

# ---- Standardize target embeddings to match noise-prior scale (per-dim, using train stats only) ----
emb_mean = emb_train_target.mean(dim=0, keepdim=True).to(device)
emb_std  = emb_train_target.std(dim=0, keepdim=True).to(device) + 1e-6

def standardize(e):   return (e - emb_mean) / emb_std
def unstandardize(z): return z * emb_std + emb_mean

emb_train_std = standardize(emb_train_target.to(device))
emb_val_std   = standardize(emb_val_target.to(device))

print(f"Post-standardization target per-dim std (mean): {emb_train_std.std(dim=0).mean().item():.4f}")
print(f"Post-standardization target norm (mean): {emb_train_std.norm(dim=1).mean().item():.4f}  (should be ~sqrt(2048)={np.sqrt(EMB_DIM):.1f})")

# ---- Re-init and retrain the flow in standardized space, with mini-batching (fixes gradient starvation too) ----
flow = ConditionalVelocityField(EMB_DIM, EMB_DIM).to(device)
opt_flow = torch.optim.Adam(flow.parameters(), lr=1e-3)

def rf_loss_batch(model, x1, ctx):
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], 1, device=x1.device)
    x_t = (1 - t) * x0 + t * x1
    return F.mse_loss(model(x_t, t, ctx), x1 - x0)

BATCH_SIZE = 128
n_train = emb_train_std.shape[0]
best_val, patience, bad = float('inf'), 20, 0

for epoch in range(500):
    flow.train()
    perm = torch.randperm(n_train, device=device)
    for i in range(0, n_train, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        opt_flow.zero_grad()
        loss = rf_loss_batch(flow, emb_train_std[idx], ctx_train.to(device)[idx])
        loss.backward()
        opt_flow.step()

    flow.eval()
    with torch.no_grad():
        val_loss = rf_loss_batch(flow, emb_val_std, ctx_val.to(device)).item()
    if val_loss < best_val:
        best_val, bad, best_flow_state = val_loss, 0, flow.state_dict()
    else:
        bad += 1
        if bad >= patience:
            print(f"Early stopping at epoch {epoch} (best val loss: {best_val:.6f})")
            break

flow.load_state_dict(best_flow_state); flow.eval()

# ---- Generate in standardized space, then invert before decoding ----
@torch.no_grad()
def generate_action_token_std(flow, ctx, steps=32):
    x = torch.randn(ctx.shape[0], EMB_DIM, device=ctx.device)  # matches standardized scale now
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.tensor([[i * dt]], device=ctx.device).float()
        x = x + flow(x, t, ctx) * dt
    return x

gen_emb_std_val = generate_action_token_std(flow, ctx_val.to(device), steps=32)

# Diagnostic: check embedding-space R2 BEFORE decoding, same as last time
emb_r2 = r2_score(emb_val_std.cpu().numpy(), gen_emb_std_val.cpu().numpy())
print(f"\nEmbedding-space R2 (standardized space): {emb_r2:.4f}")
print("Should now be well above the -0.20 / -354 baselines from before.")

# Only decode if the diagnostic looks sane
gen_emb_val = unstandardize(gen_emb_std_val)
gen_action = decoder(gen_emb_val).cpu().numpy()
act_val_np = act_val.numpy()

dims = ['x','y','z','roll','pitch','yaw','gripper']
print("\n" + "="*60)
print("GENERATED ACTION TOKEN: per-dim R2 (flow -> decode -> action)")
print("="*60)
for i, d in enumerate(dims):
    print(f"  {d:10s}: R2 = {r2_score(act_val_np[:,i], gen_action[:,i]):+.4f}")
print(f"\nOverall R2: {r2_score(act_val_np, gen_action):.4f}")