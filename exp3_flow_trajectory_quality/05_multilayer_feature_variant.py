# Standalone (own imports, own data loading, own models). A follow-up
# variant of 00-04: repeats the self- vs proprio/task-conditioned flow
# comparison across multiple VLM feature layers (-1, -2, -3, -4, avg)
# instead of a single fixed layer, using a fresh features file.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

loaded = np.load('/kaggle/input/datasets/tahak9/bosski/features (3).npz', allow_pickle=False)
LAYERS = [-1, -2, -3, -4, 'avg']
actions = torch.tensor(loaded['actions'], dtype=torch.float32)
proprio = torch.tensor(loaded['proprio'], dtype=torch.float32)
task_idx = loaded['task_index']
n_tasks = int(task_idx.max()) + 1
task_onehot = torch.tensor(np.eye(n_tasks)[task_idx], dtype=torch.float32)
context_real = torch.cat([proprio, task_onehot], dim=1)  # (N, 8+10=18)
CTX_DIM = context_real.shape[1]
ACT_DIM = actions.shape[1]
dims = ['x','y','z','roll','pitch','yaw','gripper']

idx_train, idx_val = train_test_split(np.arange(len(actions)), test_size=0.2, random_state=42)

def get_layer_key(layer):
    return 'layer_avg' if layer == 'avg' else f'layer_{layer}'

class Decoder(nn.Module):
    def __init__(self, emb_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim,256), nn.GELU(),
                                  nn.Linear(256,128), nn.GELU(),
                                  nn.Linear(128,act_dim))
    def forward(self,e): return self.net(e)

class VelocityField(nn.Module):
    def __init__(self, emb_dim, ctx_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim+ctx_dim+1,hidden), nn.GELU(),
                                  nn.Linear(hidden,hidden), nn.GELU(),
                                  nn.Linear(hidden,emb_dim))
    def forward(self,x_t,t,ctx):
        return self.net(torch.cat([x_t, ctx, t.expand(x_t.shape[0],1)], dim=-1))

def train_decoder(H_train, A_train, H_val, A_val, patience=20, max_epochs=300):
    dec = Decoder(H_train.shape[1], ACT_DIM).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
    best_val, bad, best_state = float('inf'), 0, None
    for epoch in range(max_epochs):
        dec.train(); opt.zero_grad()
        loss = F.mse_loss(dec(H_train), A_train)
        loss.backward(); opt.step()
        dec.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(dec(H_val), A_val).item()
        if val_loss < best_val:
            best_val, bad, best_state = val_loss, 0, dec.state_dict()
        else:
            bad += 1
            if bad >= patience: break
    dec.load_state_dict(best_state); dec.eval()
    return dec

def train_flow(H_train_std, ctx_train, H_val_std, ctx_val, batch_size=128, patience=20, max_epochs=400):
    emb_dim, ctx_dim = H_train_std.shape[1], ctx_train.shape[1]
    flow = VelocityField(emb_dim, ctx_dim).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    n = H_train_std.shape[0]
    best_val, bad, best_state = float('inf'), 0, None
    def rf_loss(x1, ctx):
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0],1, device=x1.device)
        x_t = (1-t)*x0 + t*x1
        return F.mse_loss(flow(x_t,t,ctx), x1-x0)
    for epoch in range(max_epochs):
        flow.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            b = perm[i:i+batch_size]
            if b.shape[0] < 8: continue
            opt.zero_grad()
            loss = rf_loss(H_train_std[b], ctx_train[b])
            loss.backward(); opt.step()
        flow.eval()
        with torch.no_grad():
            val_loss = rf_loss(H_val_std, ctx_val).item()
        if val_loss < best_val:
            best_val, bad, best_state = val_loss, 0, flow.state_dict()
        else:
            bad += 1
            if bad >= patience:
                print(f"    Early stopping at epoch {epoch} (best val loss: {best_val:.6f})")
                break
    flow.load_state_dict(best_state); flow.eval()
    return flow

@torch.no_grad()
def generate(flow, ctx, emb_dim, steps=32):
    x = torch.randn(ctx.shape[0], emb_dim, device=ctx.device)
    dt = 1.0/steps
    for i in range(steps):
        t = torch.tensor([[i*dt]], device=ctx.device).float()
        x = x + flow(x, t, ctx)*dt
    return x

results = []

for layer in LAYERS:
    print(f"\n{'='*60}\nLAYER {layer}\n{'='*60}")
    key = get_layer_key(layer)
    H = torch.tensor(loaded[key], dtype=torch.float32)
    H_train, H_val = H[idx_train].to(device), H[idx_val].to(device)
    A_train, A_val = actions[idx_train].to(device), actions[idx_val].to(device)
    ctx_real_train, ctx_real_val = context_real[idx_train].to(device), context_real[idx_val].to(device)

    # Standardize target (fixes the scale-mismatch bug from before)
    h_mean, h_std = H_train.mean(0, keepdim=True), H_train.std(0, keepdim=True) + 1e-6
    H_train_std = (H_train - h_mean) / h_std
    H_val_std = (H_val - h_mean) / h_std
    emb_dim = H.shape[1]

    # Probe ceiling for this layer (Exp1b-style)
    decoder = train_decoder(H_train, A_train, H_val, A_val)
    with torch.no_grad():
        probe_r2 = r2_score(A_val.cpu().numpy(), decoder(H_val).cpu().numpy())
    print(f"  Probe decoder ceiling R2: {probe_r2:.4f}")

    # Condition A: self-conditioned sanity ceiling (context = target itself)
    print("  Training self-conditioned flow (sanity ceiling)...")
    flow_self = train_flow(H_train_std, H_train_std, H_val_std, H_val_std)
    gen_self_std = generate(flow_self, H_val_std, emb_dim)
    gen_self = gen_self_std * h_std + h_mean
    emb_r2_self = r2_score(H_val_std.cpu().numpy(), gen_self_std.cpu().numpy())
    with torch.no_grad():
        action_r2_self = r2_score(A_val.cpu().numpy(), decoder(gen_self).cpu().numpy())
    print(f"    Embedding-space R2: {emb_r2_self:.4f}  |  Decoded action R2: {action_r2_self:.4f}")

    # Condition B: proprio+task-conditioned (the real generative test)
    print("  Training proprio+task-conditioned flow...")
    flow_real = train_flow(H_train_std, ctx_real_train, H_val_std, ctx_real_val)
    gen_real_std = generate(flow_real, ctx_real_val, emb_dim)
    gen_real = gen_real_std * h_std + h_mean
    emb_r2_real = r2_score(H_val_std.cpu().numpy(), gen_real_std.cpu().numpy())
    with torch.no_grad():
        action_r2_real = r2_score(A_val.cpu().numpy(), decoder(gen_real).cpu().numpy())
        per_dim_real = [r2_score(A_val[:,i].cpu().numpy(), decoder(gen_real)[:,i].cpu().numpy()) for i in range(ACT_DIM)]
    print(f"    Embedding-space R2: {emb_r2_real:.4f}  |  Decoded action R2: {action_r2_real:.4f}")

    results.append({
        'layer': layer, 'probe_ceiling_r2': probe_r2,
        'self_cond_emb_r2': emb_r2_self, 'self_cond_action_r2': action_r2_self,
        'real_cond_emb_r2': emb_r2_real, 'real_cond_action_r2': action_r2_real,
        'real_cond_per_dim': per_dim_real,
    })

print(f"\n{'='*70}\nSUMMARY ACROSS LAYERS\n{'='*70}")
print(f"{'Layer':<8}{'Probe ceil':<12}{'Self-cond':<12}{'Real-cond':<12}")
for r in results:
    print(f"{str(r['layer']):<8}{r['probe_ceiling_r2']:<12.4f}{r['self_cond_action_r2']:<12.4f}{r['real_cond_action_r2']:<12.4f}")

print(f"\nPer-dim R2 (proprio+task-conditioned) by layer:")
print(f"{'Layer':<8}" + "".join(f"{d:<9}" for d in dims))
for r in results:
    print(f"{str(r['layer']):<8}" + "".join(f"{v:+.3f}   " for v in r['real_cond_per_dim']))