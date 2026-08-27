# NOTE: not standalone. Run after 01_load_features_and_setup.py and
# 03_standardize_embeddings.py -- reuses `EMB_DIM`, `device`,
# `ctx_train`, `ctx_val`, `emb_train_std`, `emb_val_std`, `n_train`,
# `BATCH_SIZE`, `decoder`, `unstandardize`, `act_val_np`, `dims` from
# those scripts.
#
# Adds a plain regression baseline (context -> embedding directly, no
# flow/noise) to compare the rectified-flow model against.

class ContextToEmbeddingMLP(nn.Module):
    def __init__(self, ctx_dim, emb_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ctx_dim,hidden), nn.GELU(),
                                  nn.Linear(hidden,hidden), nn.GELU(),
                                  nn.Linear(hidden,emb_dim))
    def forward(self, ctx): return self.net(ctx)

reg = ContextToEmbeddingMLP(EMB_DIM, EMB_DIM).to(device)
opt_reg = torch.optim.Adam(reg.parameters(), lr=1e-3)

best_val, bad = float('inf'), 0
for epoch in range(500):
    reg.train()
    perm = torch.randperm(n_train, device=device)
    for i in range(0, n_train, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        opt_reg.zero_grad()
        loss = F.mse_loss(reg(ctx_train.to(device)[idx]), emb_train_std[idx])
        loss.backward(); opt_reg.step()
    reg.eval()
    with torch.no_grad():
        val_loss = F.mse_loss(reg(ctx_val.to(device)), emb_val_std).item()
    if val_loss < best_val:
        best_val, bad, best_reg_state = val_loss, 0, reg.state_dict()
    else:
        bad += 1
        if bad >= 20:
            print(f"Early stopping at epoch {epoch} (best val loss: {best_val:.6f})")
            break
reg.load_state_dict(best_reg_state); reg.eval()

with torch.no_grad():
    reg_pred_std = reg(ctx_val.to(device))
print(f"Regression baseline embedding-space R2: {r2_score(emb_val_std.cpu().numpy(), reg_pred_std.cpu().numpy()):.4f}")

reg_pred_action = decoder(unstandardize(reg_pred_std)).cpu().numpy()
print("\nRegression baseline, decoded per-dim R2:")
for i, d in enumerate(dims):
    print(f"  {d:10s}: R2 = {r2_score(act_val_np[:,i], reg_pred_action[:,i]):+.4f}")