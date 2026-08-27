# Standalone. Loads Exp1a features, trains an action encoder/decoder,
# then trains a conditional rectified-flow model on top of it and
# reports per-dim R2 on the decoded, flow-generated actions.
#
# This is the base script for the rest of this folder: 02-05 continue
# debugging/extending the flow trained here and expect to run in the
# SAME session right after this file (they reuse `device`, `flow`,
# `decoder`, `emb_train_target`, `ctx_val`, `act_val`,
# `generate_action_token`, etc. from this script's globals).

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ---- Load Experiment 1a features: context (VLM hidden state) + actions (targets) ----
data = np.load('/kaggle/input/datasets/tahak9/boyfla/features.npz')  # adjust path
context = data['layer_-2']          # (1500, 2048) -- conditioning signal
actions = data['actions']           # (1500, 7)    -- ground-truth robot actions

context = torch.tensor(context, dtype=torch.float32)
actions = torch.tensor(actions, dtype=torch.float32)
EMB_DIM, ACT_DIM = context.shape[1], actions.shape[1]

ctx_train, ctx_val, act_train, act_val = train_test_split(
    context, actions, test_size=0.2, random_state=42
)

# ---- Stage 1: PLAIN action encoder/decoder (mirrors Exp2 condition (a) -- reg collapsed, plain didn't) ----
class ActionEncoder(nn.Module):
    def __init__(self, act_dim, emb_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(act_dim,256), nn.GELU(),
                                  nn.Linear(256,512), nn.GELU(),
                                  nn.Linear(512,emb_dim))
    def forward(self,a): return self.net(a)

class ActionDecoder(nn.Module):
    def __init__(self, emb_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim,512), nn.GELU(),
                                  nn.Linear(512,256), nn.GELU(),
                                  nn.Linear(256,act_dim))
    def forward(self,e): return self.net(e)

encoder, decoder = ActionEncoder(ACT_DIM,EMB_DIM).to(device), ActionDecoder(EMB_DIM,ACT_DIM).to(device)
opt = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=1e-3)

best_val, patience, bad = float('inf'), 20, 0
for epoch in range(300):
    encoder.train(); decoder.train()
    opt.zero_grad()
    emb = encoder(act_train.to(device))
    loss = F.mse_loss(decoder(emb), act_train.to(device))
    loss.backward(); opt.step()
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        val_loss = F.mse_loss(decoder(encoder(act_val.to(device))), act_val.to(device)).item()
    if val_loss < best_val:
        best_val, bad, best_state = val_loss, 0, (encoder.state_dict(), decoder.state_dict())
    else:
        bad += 1
        if bad >= patience:
            print(f"Early stopping at epoch {epoch} (best val loss: {best_val:.6f})")
            break
encoder.load_state_dict(best_state[0]); decoder.load_state_dict(best_state[1])
encoder.eval(); decoder.eval()
for p in list(encoder.parameters())+list(decoder.parameters()): p.requires_grad_(False)

with torch.no_grad():
    emb_train_target = encoder(act_train.to(device))
    emb_val_target = encoder(act_val.to(device))

# ---- Stage 2: Conditional rectified flow -- noise -> action-token embedding, conditioned on real VLM context ----
class ConditionalVelocityField(nn.Module):
    def __init__(self, emb_dim, ctx_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim+ctx_dim+1,hidden), nn.GELU(),
                                  nn.Linear(hidden,hidden), nn.GELU(),
                                  nn.Linear(hidden,emb_dim))
    def forward(self,x_t,t,ctx):
        return self.net(torch.cat([x_t, ctx, t.expand(x_t.shape[0],1)], dim=-1))

flow = ConditionalVelocityField(EMB_DIM, EMB_DIM).to(device)
opt_flow = torch.optim.Adam(flow.parameters(), lr=1e-3)

def rf_loss(model, x1, ctx):
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0],1, device=x1.device)
    x_t = (1-t)*x0 + t*x1
    return F.mse_loss(model(x_t,t,ctx), x1-x0)

best_val, bad = float('inf'), 0
for epoch in range(400):
    flow.train(); opt_flow.zero_grad()
    loss = rf_loss(flow, emb_train_target, ctx_train.to(device))
    loss.backward(); opt_flow.step()
    flow.eval()
    with torch.no_grad():
        val_loss = rf_loss(flow, emb_val_target, ctx_val.to(device)).item()
    if val_loss < best_val:
        best_val, bad, best_flow_state = val_loss, 0, flow.state_dict()
    else:
        bad += 1
        if bad >= patience:
            print(f"Early stopping at epoch {epoch} (best val loss: {best_val:.6f})")
            break
flow.load_state_dict(best_flow_state); flow.eval()

# ---- Inference: integrate ODE from noise, conditioned on real context ----
@torch.no_grad()
def generate_action_token(flow, ctx, steps=32):
    x = torch.randn(ctx.shape[0], EMB_DIM, device=ctx.device)
    dt = 1.0/steps
    for i in range(steps):
        t = torch.tensor([[i*dt]], device=ctx.device).float()
        x = x + flow(x, t, ctx)*dt
    return x

gen_emb = generate_action_token(flow, ctx_val.to(device), steps=32)
gen_action = decoder(gen_emb).cpu().numpy()
act_val_np = act_val.numpy()

dims = ['x','y','z','roll','pitch','yaw','gripper']
print("\n" + "="*60)
print("GENERATED ACTION TOKEN: per-dim R2 (flow -> decode -> action)")
print("="*60)
for i,d in enumerate(dims):
    print(f"  {d:10s}: R2 = {r2_score(act_val_np[:,i], gen_action[:,i]):+.4f}")
overall = r2_score(act_val_np, gen_action)
print(f"\nOverall R2: {overall:.4f}")
print("Compare vs Exp1b probe ceiling (~0.50-0.54 overall, ~0.30-0.38 orientation).")
print("Approaching ceiling = flow faithfully reproduces decodable signal.")
print("Well below = flow underfitting. Above = leakage, check the split.")