import torch
import torch.nn.functional as F
import numpy as np
import math

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def to_x(u, E):
    return (E @ u.T).T

def make_embedding(d, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    E = torch.randn(d, 2, generator=g)
    Q, _ = torch.linalg.qr(E)
    return Q[:, :2].contiguous().to(device)

def sample_u(n):
    return 2 * torch.rand(n, 2) - 1.0

def y_circle(u, r0=0.6):
    return (u.pow(2).sum(dim=1) < r0 ** 2).float()

def y_bubble_flip(u, uc, rb=0.08, r0=0.6):
    y0 = y_circle(u, r0)
    flip = ((u - uc.to(u.device)).pow(2).sum(dim=1) < rb ** 2).float()
    return y0 + flip - 2 * y0 * flip

def sample_u_boosted(n, uc, rb, bubble_frac=0.5):
    n_bubble = int(n * bubble_frac)
    n_global = n - n_bubble
    u_global = sample_u(n_global)
    collected = []
    while len(collected) < n_bubble:
        delta = (2 * torch.rand(n_bubble * 4, 2, device=uc.device) - 1) * rb
        mask  = delta.pow(2).sum(dim=1) < rb ** 2
        pts   = uc.unsqueeze(0) + delta[mask]
        pts   = pts.clamp(-1, 1)
        collected.append(pts)
    u_bubble = torch.cat(collected)[:n_bubble]
    return torch.cat([u_global, u_bubble], dim=0)

@torch.no_grad()
def find_bubble_center(model, E, r0, radius, device, trials=10000, interior_samples=64):
    model.eval()
    for _ in range(trials):
        u_try = (2 * torch.rand(2, device=device) - 1)
        if u_try.pow(2).sum() >= (r0 - radius) ** 2:
            continue
        # Project center
        x_center = (E @ u_try.unsqueeze(0).T).T
        base_pat = model.gate_pattern(x_center)

        pts = []
        attempts = 0
        while len(pts) < interior_samples and attempts < interior_samples * 20:
            delta = (2 * torch.rand(2, device=device) - 1) * radius
            if delta.pow(2).sum() < radius**2:
                pts.append(u_try + delta)
            attempts += 1
        if len(pts) < interior_samples // 2: continue
        
        batch_u = torch.stack(pts)
        batch_x = (E @ batch_u.T).T
        pats = model.gate_pattern(batch_x)
        if (pats == base_pat).all():
            return u_try
    raise RuntimeError("Could not find bubble center")

@torch.no_grad()
def eval_acc(model, E, y_fn, device, n=5000):
    model.eval()
    u = sample_u(n).to(device)
    x = (E @ u.T).T
    preds = (torch.sigmoid(model(x)) > 0.5).float()
    labels = y_fn(u.cpu()).to(device)
    return preds.eq(labels).float().mean().item()

@torch.no_grad()
def eval_bubble_acc(model, E, uc, rb, y_fn, device, n=4000):
    model.eval()
    inside = []
    while len(inside) < n:
        u = sample_u(2000).to(device)
        mask = (u - uc.to(device)).pow(2).sum(dim=1) < rb ** 2
        if mask.any():
            inside.append(u[mask])
    u_b = torch.cat(inside)[:n]
    x = (E @ u_b.T).T
    preds = (torch.sigmoid(model(x)) > 0.5).float()
    labels = y_fn(u_b.cpu()).to(device)
    return preds.eq(labels).float().mean().item()

def train_base(model, E, y_fn, device, steps=2500, lr=3e-3, batch=512, weight_decay=1e-4):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        u = sample_u(batch).to(device)
        x = (E @ u.T).T
        loss = F.binary_cross_entropy_with_logits(model(x), y_fn(u).to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def train_until_loss(model, E, y_fn, uc, rb, device, target_loss=0.15, max_steps=8000, lr=3e-3, batch=512, weight_decay=1e-4, bubble_frac=0.5):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params: return model
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    for t in range(max_steps):
        u = sample_u_boosted(batch, uc, rb, bubble_frac).to(device)
        x = (E @ u.T).T
        loss = F.binary_cross_entropy_with_logits(model(x), y_fn(u).to(device))
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() <= target_loss:
            print(f"    -> Target loss {target_loss} reached at step {t}.")
            return model
    print(f"    -> WARNING: target loss not reached. Final: {loss.item():.4f}")
    return model
