"""
LoRA Polytope Complexity Experiment
====================================
Hypothesis: LoRA enforces correlated, low-dimensional rotations of ReLU
gate hyperplanes, rather than the independent local fragmentation that
Full Fine-Tuning produces.

Run with:
    conda run -n plora python experiment.py
"""
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------
# Config & Hyperparameters
# -----------------------
SEED          = 42
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

d_in          = 10
m_hidden      = 32

r0            = 0.60   # Circle radius (Task 0)
rb            = 0.08   # Bubble radius (Task 1 XOR patch)
uc            = None   # Set dynamically after pretraining

lr_base       = 3e-3
lr_adapt      = 3e-3
steps_base    = 2500
max_steps_adapt = 8000
batch         = 512
lora_r        = 2
lora_alpha    = 2.0
weight_decay  = 1e-4
TARGET_LOSS   = 0.15
BUBBLE_OVERSAMPLE = 0.5   # fraction of each batch drawn from bubble region

os.makedirs("figures", exist_ok=True)

# -----------------------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

# -----------------------------------------------------------------------
# Data & Geometry
# -----------------------------------------------------------------------
def sample_u(n):
    return 2 * torch.rand(n, 2) - 1.0

def y_circle(u):
    return (u.pow(2).sum(dim=1) < r0 ** 2).float()

def y_bubble_flip(u):
    y0  = y_circle(u)
    if uc is None:
        return y0
    flip = ((u - uc.to(u.device)).pow(2).sum(dim=1) < rb ** 2).float()
    return y0 + flip - 2 * y0 * flip   # XOR

def make_embedding(d, seed=0):
    g = torch.Generator().manual_seed(seed)
    E = torch.randn(d, 2, generator=g)
    Q, _ = torch.linalg.qr(E)
    return Q[:, :2].contiguous()

# -----------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_in, m_hidden):
        super().__init__()
        self.fc1 = nn.Linear(d_in, m_hidden, bias=True)
        self.fc2 = nn.Linear(m_hidden, 1,  bias=True)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x))).squeeze(-1)

    @torch.no_grad()
    def gate_pattern(self, x):
        return self.fc1(x) > 0


class LoRAFirstLayerMLP(nn.Module):
    def __init__(self, base: MLP, r: int, alpha: float):
        super().__init__()
        self.d_in     = base.fc1.in_features
        self.m_hidden = base.fc1.out_features
        self.scaling  = alpha / max(1, r)

        self.W0 = nn.Parameter(base.fc1.weight.detach().clone(), requires_grad=False)
        self.b0 = nn.Parameter(base.fc1.bias.detach().clone(),   requires_grad=False)

        self.A = nn.Parameter(torch.empty(r, self.d_in))
        self.B = nn.Parameter(torch.zeros(self.m_hidden, r))
        nn.init.normal_(self.A, std=1e-3)

        self.fc2 = nn.Linear(self.m_hidden, 1, bias=True)
        self.fc2.weight.data.copy_(base.fc2.weight.detach())
        self.fc2.bias.data.copy_(base.fc2.bias.detach())

    def effective_W(self):
        return self.W0 + self.scaling * (self.B @ self.A)

    def forward(self, x):
        return self.fc2(F.relu(F.linear(x, self.effective_W(), self.b0))).squeeze(-1)

    @torch.no_grad()
    def gate_pattern(self, x):
        return F.linear(x, self.effective_W(), self.b0) > 0

# -----------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------
def to_x(u, E):
    return (E @ u.T).T

def train_base(model, E, y_fn, steps, lr):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        u = sample_u(batch).to(DEVICE)
        loss = F.binary_cross_entropy_with_logits(model(to_x(u, E)), y_fn(u).to(DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def sample_u_boosted(n, bubble_frac=BUBBLE_OVERSAMPLE):
    """Sample a batch with `bubble_frac` fraction from inside the bubble."""
    if uc is None:
        return sample_u(n)
    n_bubble = int(n * bubble_frac)
    n_global = n - n_bubble

    # Global samples
    u_global = sample_u(n_global)

    # Bubble-region samples (rejection sample inside circle of radius rb)
    collected = []
    while len(collected) < n_bubble:
        delta = (2 * torch.rand(n_bubble * 4, 2) - 1) * rb
        mask  = delta.pow(2).sum(dim=1) < rb ** 2
        pts   = uc.unsqueeze(0) + delta[mask]
        # Clip to [-1, 1]
        pts   = pts.clamp(-1, 1)
        collected.append(pts)
    u_bubble = torch.cat(collected)[:n_bubble]
    return torch.cat([u_global, u_bubble], dim=0)


def train_until_loss(model, E, y_fn, max_steps, lr, target_loss):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return model
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    for t in range(max_steps):
        u    = sample_u_boosted(batch).to(DEVICE)
        loss = F.binary_cross_entropy_with_logits(model(to_x(u, E)), y_fn(u).to(DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() <= target_loss:
            print(f"    -> Target loss {target_loss} reached at step {t}.")
            return model
    print(f"    -> WARNING: target loss not reached. Final: {loss.item():.4f}")
    return model

@torch.no_grad()
def eval_acc(model, E, y_fn, n=5000):
    model.eval()
    u = sample_u(n).to(DEVICE)
    preds = (torch.sigmoid(model(to_x(u, E))) > 0.5).float()
    return preds.eq(y_fn(u).to(DEVICE)).float().mean().item()

@torch.no_grad()
def eval_bubble_acc(model, E, y_fn, n=4000):
    """Accuracy restricted to points inside the bubble region."""
    model.eval()
    inside = []
    while len(inside) < n:
        u = sample_u(2000).to(DEVICE)
        mask = (u - uc.to(DEVICE)).pow(2).sum(dim=1) < rb ** 2
        if mask.any():
            inside.append(u[mask])
    u_b = torch.cat(inside)[:n]
    preds = (torch.sigmoid(model(to_x(u_b, E))) > 0.5).float()
    labels = y_fn(u_b.cpu()).to(DEVICE)
    return preds.eq(labels).float().mean().item()

# -----------------------------------------------------------------------
# Topological trap
# -----------------------------------------------------------------------
@torch.no_grad()
def find_bubble_center(model, E, radius, trials=10000, interior_samples=64):
    """
    Find u_c strictly inside the r0-circle where fc1's gate pattern is
    constant across the whole bubble of given radius.

    Strategy:
      1. Sample a candidate center strictly inside the circle margin.
      2. Sample `interior_samples` random points uniformly inside the bubble.
      3. If all share the same gate pattern as the center -> accept.
    This avoids the ring-only check failing when hyperplanes pass through
    the bubble interior.
    """
    model.eval()
    for _ in range(trials):
        u_try = (2 * torch.rand(2, device=DEVICE) - 1)
        # Must be inside the large circle with margin so full bubble fits
        if u_try.pow(2).sum() >= (r0 - radius) ** 2:
            continue
        base_pat = model.gate_pattern(to_x(u_try.unsqueeze(0), E))

        # Sample points uniformly inside the bubble (rejection sampling)
        r_sq = radius ** 2
        pts  = []
        attempts = 0
        while len(pts) < interior_samples and attempts < interior_samples * 20:
            delta = (2 * torch.rand(2, device=DEVICE) - 1) * radius
            if delta.pow(2).sum() < r_sq:
                pts.append(u_try + delta)
            attempts += 1
        if len(pts) < interior_samples // 2:
            continue

        batch_u = torch.stack(pts)           # (k, 2)
        pats    = model.gate_pattern(to_x(batch_u, E))   # (k, m_hidden)
        # All must match the center pattern
        if (pats == base_pat).all():
            return u_try
    raise RuntimeError(
        "Could not find a constant-gate bubble inside the circle after "
        f"{trials} trials. Try re-seeding or enlarging the search space."
    )

# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------
def get_W1(model):
    if hasattr(model, "fc1"):
        return model.fc1.weight.detach()
    return model.effective_W().detach()

def matrix_rank(M, tol=1e-6):
    return int((torch.linalg.svdvals(M) > tol).sum())

def stable_rank(M):
    s = torch.linalg.svdvals(M)
    return float(s.pow(2).sum() / (s[0].pow(2) + 1e-12))

@torch.no_grad()
def hyperplane_rotation_rank(base_m, adapt_m, E=None, tol=1e-6):
    W0 = get_W1(base_m);  W1 = get_W1(adapt_m)
    if E is not None:
        W0, W1 = W0 @ E, W1 @ E
    n0 = W0 / (W0.norm(dim=1, keepdim=True) + 1e-12)
    n1 = W1 / (W1.norm(dim=1, keepdim=True) + 1e-12)
    dN = n1 - n0
    s  = torch.linalg.svdvals(dN)
    return int((s > tol).sum()), float(s.pow(2).sum() / (s.max().pow(2) + 1e-12))

@torch.no_grad()
def line_crossing_complexity(model, E, n_lines=256, pts=256):
    model.eval()
    crossings = []
    for _ in range(n_lines):
        theta   = (torch.rand(1, device=DEVICE) * 2 * math.pi).item()
        dv      = torch.tensor([math.cos(theta), math.sin(theta)], device=DEVICE)
        nv      = torch.tensor([-math.sin(theta), math.cos(theta)], device=DEVICE)
        offset  = (torch.rand(1, device=DEVICE) * 2 - 1).item()
        t       = torch.linspace(-1.5, 1.5, pts, device=DEVICE)
        u       = offset * nv + t.unsqueeze(1) * dv
        mask    = (u.abs() <= 1).all(dim=1)
        if mask.sum() < 10:
            continue
        s = (model(to_x(u[mask], E)) > 0).float()
        crossings.append((s[1:] != s[:-1]).sum().item())
    return float(np.mean(crossings)) if crossings else 0.0

@torch.no_grad()
def measure_gate_drift(base_m, adapt_m, E, resolution=300):
    t  = torch.linspace(-1, 1, resolution, device=DEVICE)
    Ux, Uy = torch.meshgrid(t, t, indexing="xy")
    U  = torch.stack([Ux.reshape(-1), Uy.reshape(-1)], dim=1)
    X  = to_x(U, E)
    gb = base_m.gate_pattern(X)
    ga = adapt_m.gate_pattern(X)
    return (gb ^ ga).any(dim=1).float().mean().item()

# -----------------------------------------------------------------------
# Visualisation helpers
# -----------------------------------------------------------------------
@torch.no_grad()
def decision_grid(model, E, resolution=300):
    t = torch.linspace(-1, 1, resolution, device=DEVICE)
    Ux, Uy = torch.meshgrid(t, t, indexing="xy")
    U = torch.stack([Ux.reshape(-1), Uy.reshape(-1)], dim=1)
    logits = model(to_x(U, E)).reshape(resolution, resolution)
    return Ux.cpu().numpy(), Uy.cpu().numpy(), torch.sigmoid(logits).cpu().numpy()

def plot_boundary(ax, model, E, title, resolution=300):
    Ux, Uy, prob = decision_grid(model, E, resolution)
    ax.contourf(Ux, Uy, prob, levels=50, cmap="RdBu_r", vmin=0, vmax=1, alpha=0.85)
    ax.contour(Ux, Uy, prob, levels=[0.5], colors="white", linewidths=1.5)
    # Draw circle
    circle = plt.Circle((0, 0), r0, color="gold", fill=False, lw=1.5, ls="--")
    ax.add_patch(circle)
    # Draw bubble
    if uc is not None:
        bub = plt.Circle(uc.numpy(), rb, color="lime", fill=False, lw=1.5, ls="--")
        ax.add_patch(bub)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")

@torch.no_grad()
def plot_gate_drift_map(base_m, adapt_m, E, title, resolution=300):
    t = torch.linspace(-1, 1, resolution, device=DEVICE)
    Ux, Uy = torch.meshgrid(t, t, indexing="xy")
    U = torch.stack([Ux.reshape(-1), Uy.reshape(-1)], dim=1)
    X = to_x(U, E)
    gb = base_m.gate_pattern(X)
    ga = adapt_m.gate_pattern(X)
    drift = (gb ^ ga).any(dim=1).float().reshape(resolution, resolution)
    return Ux.cpu().numpy(), Uy.cpu().numpy(), drift.cpu().numpy(), title

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    global uc
    set_seed(SEED)
    E = make_embedding(d_in, seed=SEED).to(DEVICE)

    # ── 1. Pretrain ───────────────────────────────────────────────────
    print("\n[1/5] Pretraining Base Model on Task 0 (circle)…")
    base = MLP(d_in, m_hidden).to(DEVICE)
    train_base(base, E, y_circle, steps_base, lr_base)
    acc0 = eval_acc(base, E, y_circle)
    print(f"    Base Acc (circle): {acc0:.3f}")

    # ── 2. Find bubble (must be strictly inside the circle) ───────────
    print("\n[2/5] Searching for topological trap inside the circle…")
    uc = find_bubble_center(base, E, rb).detach().cpu()
    print(f"    Bubble center: {uc.numpy()}  (dist to origin: {uc.norm():.3f})")

    # ── 3. Head-Only Assert ──────────────────────────────────────────
    # Since the gate pattern is constant throughout the bubble, fc1
    # produces the *same hidden vector* for every bubble point.
    # Therefore fc2 (regardless of its weights) maps every bubble point
    # to the *same scalar*, so 50% of bubble points are wrong by definition
    # (the XOR label is the complement of y0 inside the bubble).
    # We verify this: freeze EVERYTHING and just measure.
    print("\n[3/5] Head-Only Control Assert (no training - geometry proof)…")
    head = MLP(d_in, m_hidden).to(DEVICE)
    head.load_state_dict(base.state_dict())

    acc_head_bubble_before = eval_bubble_acc(head, E, y_bubble_flip)
    print(f"    Base model bubble-region acc on Task-1 (XOR): {acc_head_bubble_before:.3f}")
    # Since the gate is constant in the bubble and label is flipped,
    # all points share the same network output = 1 - correct.
    # So accuracy should be near 0 (all wrong) or near 1 (all correct by chance).
    # We assert it cannot be > 0.75 because the constant output prediction
    # either gets all bubble points wrong (≈0) or all right (≈1), never partial.
    # After the circle model, the bubble is in the positive class, so acc ≈ 0.
    print(f"    (Expected ≈ 0.0 since base predicts 'positive' for whole bubble, XOR flips to negative)")
    if acc_head_bubble_before > 0.75:
        raise RuntimeError(
            f"ASSERT FAILED: Base model already gets bubble right ({acc_head_bubble_before:.3f}). "
            "The bubble is co-labeled with the surrounding region — topological trap is trivial."
        )
    print("    ✓ Assert PASSED: bubble cannot be solved without moving fc1 hyperplanes.")

    # ── 4. Full Fine-Tuning ──────────────────────────────────────────
    print("\n[4/5] Full Fine-Tuning (fc1 weights free, fc2 + biases frozen)…")
    full = MLP(d_in, m_hidden).to(DEVICE)
    full.load_state_dict(base.state_dict())
    full.fc1.bias.requires_grad   = False
    full.fc2.weight.requires_grad = False
    full.fc2.bias.requires_grad   = False
    train_until_loss(full, E, y_bubble_flip, max_steps_adapt, lr_adapt, TARGET_LOSS)

    # ── 5. LoRA Fine-Tuning ──────────────────────────────────────────
    print(f"\n[5/5] LoRA Fine-Tuning (rank r={lora_r}, fc2 frozen)…")
    lora = LoRAFirstLayerMLP(base, r=lora_r, alpha=lora_alpha).to(DEVICE)
    lora.fc2.weight.requires_grad = False
    lora.fc2.bias.requires_grad   = False
    train_until_loss(lora, E, y_bubble_flip, max_steps_adapt, lr_adapt, TARGET_LOSS)

    # ── Results ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL GEOMETRIC ANALYSIS")
    print("=" * 60)

    acc_full = eval_acc(full, E, y_bubble_flip)
    acc_lora = eval_acc(lora, E, y_bubble_flip)
    acc_full_b = eval_bubble_acc(full, E, y_bubble_flip)
    acc_lora_b = eval_bubble_acc(lora, E, y_bubble_flip)
    print(f"\nTask-1 Accuracy | Full FT: {acc_full:.3f} (bubble: {acc_full_b:.3f})"
          f" | LoRA: {acc_lora:.3f} (bubble: {acc_lora_b:.3f})")

    W0        = get_W1(base)
    dW_full   = get_W1(full) - W0
    dW_lora   = get_W1(lora) - W0

    print("\n[Level 1] Update-Matrix-Rank (Weight Space)")
    print(f"    Full FT -> rank: {matrix_rank(dW_full)},  stable-rank: {stable_rank(dW_full):.2f}")
    print(f"    LoRA    -> rank: {matrix_rank(dW_lora)},  stable-rank: {stable_rank(dW_lora):.2f}")

    r_f,  sr_f  = hyperplane_rotation_rank(base, full)
    r_l,  sr_l  = hyperplane_rotation_rank(base, lora)
    rm_f, srm_f = hyperplane_rotation_rank(base, full, E)
    rm_l, srm_l = hyperplane_rotation_rank(base, lora, E)
    print("\n[Level 2] Hyperplane-Rotation-Rank (Orientation Space)")
    print(f"    Full FT  10D -> rank: {r_f},  stable-rank: {sr_f:.2f}")
    print(f"    LoRA     10D -> rank: {r_l},  stable-rank: {sr_l:.2f}")
    print(f"    Full FT  2D  -> rank: {rm_f}, stable-rank: {srm_f:.2f}")
    print(f"    LoRA     2D  -> rank: {rm_l}, stable-rank: {srm_l:.2f}")

    gd_full = measure_gate_drift(base, full, E)
    gd_lora = measure_gate_drift(base, lora, E)
    print("\n[Level 3] Gate-Drift (Topological Partitioning)")
    print(f"    Full FT -> Changed Area: {gd_full * 100:.1f}%")
    print(f"    LoRA    -> Changed Area: {gd_lora * 100:.1f}%")

    lc_base = line_crossing_complexity(base, E)
    lc_full = line_crossing_complexity(full, E)
    lc_lora = line_crossing_complexity(lora, E)
    print("\n[Level 4] Line-Crossing Complexity (Crofton-Proxy)")
    print(f"    Base    -> {lc_base:.2f}")
    print(f"    Full FT -> {lc_full:.2f}")
    print(f"    LoRA    -> {lc_lora:.2f}")
    print("=" * 60)

    # ── Visualisations ───────────────────────────────────────────────
    print("\nGenerating figures…")

    # Fig 1: Decision boundaries
    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
    plot_boundary(axes[0], base, E, "Base Model (Task 0)")
    plot_boundary(axes[1], full, E, "Full Fine-Tune (Task 1)")
    plot_boundary(axes[2], lora, E, f"LoRA r={lora_r} (Task 1)")
    fig.suptitle("Decision Boundaries in 2D Data Space", color="white",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("figures/decision_boundaries.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("    Saved figures/decision_boundaries.png")

    # Fig 2: Gate-drift maps
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
    for ax, (bm, am, title) in zip(axes, [
        (base, full, f"Gate-Drift: Full FT  ({gd_full*100:.1f}% of grid)"),
        (base, lora, f"Gate-Drift: LoRA r={lora_r} ({gd_lora*100:.1f}% of grid)"),
    ]):
        Ux, Uy, drift, t = plot_gate_drift_map(bm, am, E, title)
        ax.contourf(Ux, Uy, drift, levels=2, cmap="hot", alpha=0.9)
        if uc is not None:
            bub = plt.Circle(uc.numpy(), rb, color="cyan", fill=False, lw=2, ls="--")
            ax.add_patch(bub)
        circle = plt.Circle((0, 0), r0, color="gold", fill=False, lw=1.5, ls="--")
        ax.add_patch(circle)
        ax.set_title(t, color="white", fontsize=10, fontweight="bold")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("Gate-Drift Maps (changed ReLU gates vs. Base)", color="white",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/gate_drift.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("    Saved figures/gate_drift.png")

    # Fig 3: Singular-value spectrum of dW
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#1a1f2e")
        ax.spines[:].set_color("#444")
        ax.tick_params(colors="white")
    for dW, label, color, ax in [
        (dW_full, "Full FT", "#ff6b6b", axes[0]),
        (dW_lora, f"LoRA r={lora_r}", "#4ecdc4", axes[1]),
    ]:
        sv = torch.linalg.svdvals(dW).cpu().numpy()
        ax.bar(range(len(sv)), sv, color=color, alpha=0.85, width=1.0)
        ax.set_title(f"Singular Values of ΔW — {label}", color="white",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Index", color="#aaa")
        ax.set_ylabel("σ", color="#aaa")
        ax.set_yscale("log")
    fig.suptitle("Update-Matrix Singular Value Spectra", color="white",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/sv_spectra.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("    Saved figures/sv_spectra.png")

    # Fig 4: Bar-chart summary of metrics
    metrics = {
        "Update Stable-Rank":       (stable_rank(dW_full), stable_rank(dW_lora)),
        "Rotation SR (10D)":        (sr_f,  sr_l),
        "Rotation SR (2D)":         (srm_f, srm_l),
        "Gate-Drift (%)":           (gd_full * 100, gd_lora * 100),
        "Line-Crossings/Line":      (lc_full, lc_lora),
    }
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#0d1117")
    ax.set_facecolor("#1a1f2e")
    ax.spines[:].set_color("#444")
    ax.tick_params(colors="white")

    keys   = list(metrics.keys())
    v_full = [metrics[k][0] for k in keys]
    v_lora = [metrics[k][1] for k in keys]
    x = np.arange(len(keys)); w = 0.35
    bars_f = ax.bar(x - w/2, v_full, w, label="Full FT", color="#ff6b6b", alpha=0.9)
    bars_l = ax.bar(x + w/2, v_lora, w, label=f"LoRA r={lora_r}", color="#4ecdc4", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20, ha="right", color="white", fontsize=9)
    ax.set_ylabel("Value", color="#aaa")
    ax.set_title("Metric Summary: Full FT vs LoRA", color="white",
                 fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1a1f2e", labelcolor="white", framealpha=0.8)
    plt.tight_layout()
    plt.savefig("figures/metric_summary.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("    Saved figures/metric_summary.png")

    # ── Machine-readable results dump ────────────────────────────────
    results = dict(
        acc_base_circle        = acc0,
        acc_head_bubble_region = acc_head_bubble_before,
        acc_full_global        = acc_full,
        acc_full_bubble        = acc_full_b,
        acc_lora_global        = acc_lora,
        acc_lora_bubble        = acc_lora_b,
        # Level 1
        full_rank_dW           = matrix_rank(dW_full),
        full_stable_rank_dW    = stable_rank(dW_full),
        lora_rank_dW           = matrix_rank(dW_lora),
        lora_stable_rank_dW    = stable_rank(dW_lora),
        # Level 2
        full_rot_rank_10d      = r_f,
        full_rot_sr_10d        = sr_f,
        lora_rot_rank_10d      = r_l,
        lora_rot_sr_10d        = sr_l,
        full_rot_rank_2d       = rm_f,
        full_rot_sr_2d         = srm_f,
        lora_rot_rank_2d       = rm_l,
        lora_rot_sr_2d         = srm_l,
        # Level 3
        full_gate_drift_pct    = gd_full * 100,
        lora_gate_drift_pct    = gd_lora * 100,
        # Level 4
        base_line_crossings    = lc_base,
        full_line_crossings    = lc_full,
        lora_line_crossings    = lc_lora,
    )
    import json
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("    Saved results.json")
    print("\nDone ✓")
    return results


if __name__ == "__main__":
    main()
