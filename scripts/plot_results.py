import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from models.utils import eval_acc, to_x
from metrics.curvature import boundary_discrete_curvature

def plot_boundary(ax, model, E, uc, rb, r0, device, title, resolution=300):
    t = torch.linspace(-1, 1, resolution, device=device)
    Ux, Uy = torch.meshgrid(t, t, indexing="xy")
    U = torch.stack([Ux.reshape(-1), Uy.reshape(-1)], dim=1)
    
    with torch.no_grad():
        X = (E @ U.T).T
        logits = model(X).reshape(resolution, resolution)
        prob = torch.sigmoid(logits).cpu().numpy()
        
    Ux_np, Uy_np = Ux.cpu().numpy(), Uy.cpu().numpy()
    ax.contourf(Ux_np, Uy_np, prob, levels=50, cmap="RdBu_r", vmin=0, vmax=1, alpha=0.85)
    ax.contour(Ux_np, Uy_np, prob, levels=[0.5], colors="white", linewidths=1.5)
    
    # Draw circle
    circle = plt.Circle((0, 0), r0, color="gold", fill=False, lw=1.5, ls="--")
    ax.add_patch(circle)
    # Draw bubble
    if uc is not None:
        bub = plt.Circle(uc.cpu().numpy(), rb, color="lime", fill=False, lw=1.5, ls="--")
        ax.add_patch(bub)
        
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")

def plot_drift_heatmap(ax, base_m, adapt_m, E, uc, rb, r0, device, title, resolution=300):
    t = torch.linspace(-1, 1, resolution, device=device)
    Ux, Uy = torch.meshgrid(t, t, indexing="xy")
    U = torch.stack([Ux.reshape(-1), Uy.reshape(-1)], dim=1)
    
    with torch.no_grad():
        X = (E @ U.T).T
        gb = base_m.gate_pattern(X)
        ga = adapt_m.gate_pattern(X)
        drift = (gb ^ ga).any(dim=1).float().reshape(resolution, resolution).cpu().numpy()
        
    Ux_np, Uy_np = Ux.cpu().numpy(), Uy.cpu().numpy()
    
    # Plot drift as hot map
    ax.contourf(Ux_np, Uy_np, drift, levels=[-0.5, 0.5, 1.5], colors=["#0d1117", "crimson"], alpha=0.9)
    
    # Overlay context
    circle = plt.Circle((0, 0), r0, color="gold", fill=False, lw=1.5, ls="--", alpha=0.5)
    ax.add_patch(circle)
    if uc is not None:
        bub = plt.Circle(uc.cpu().numpy(), rb, color="cyan", fill=False, lw=1.5, ls="--", alpha=0.8)
        ax.add_patch(bub)
        
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=11, fontweight="bold", color="white")
    ax.axis("off")

def plot_curvature_histogram(ax, model, E, device, title):
    # Get raw curvatures. Use a deep copy so the original model is never moved.
    curvatures = []
    model = copy.deepcopy(model).to("cpu")
    E_cpu = E.to("cpu")
    dtype = E_cpu.dtype
    for _ in range(400):  # More lines for a smooth histogram
        theta = torch.rand((), device="cpu", dtype=dtype) * 2 * np.pi
        direction = torch.tensor([torch.cos(theta), torch.sin(theta)], device="cpu", dtype=dtype)
        normal = torch.tensor([-direction[1], direction[0]], device="cpu", dtype=dtype)
        offset = (torch.rand((), device="cpu", dtype=dtype) * 2 - 1)
        t = torch.linspace(-1.5, 1.5, 400, device="cpu", dtype=dtype)
        u = offset * normal + t.unsqueeze(1) * direction
        mask = (torch.abs(u) <= 1).all(dim=1)
        if mask.sum() < 20: continue
        u_valid = u[mask]
        with torch.no_grad():
            x_eval = (E_cpu @ u_valid.T).T
            logits = model(x_eval)
            s = (logits > 0).float()
            crossings = torch.where(s[1:] != s[:-1])[0]
        if len(crossings) < 2: continue
        normals = []
        for idx in crossings:
            alpha = torch.abs(logits[idx]) / (torch.abs(logits[idx]) + torch.abs(logits[idx+1]) + 1e-12)
            x_cross = (1 - alpha) * x_eval[idx] + alpha * x_eval[idx+1]
            xi = x_cross.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                f = model(xi.unsqueeze(0))
                g = torch.autograd.grad(f, xi)[0]
            n = g / (g.norm() + 1e-12)
            normals.append(n.detach())
        normals = torch.stack(normals)
        for i in range(len(normals) - 1):
            cos_val = torch.clamp(torch.dot(normals[i], normals[i+1]), -1.0, 1.0)
            angle = torch.acos(cos_val)
            curvatures.append(angle.item())
            
    if len(curvatures) > 0:
        ax.hist(curvatures, bins=40, color="teal", alpha=0.7, log=True)
        mean_c = np.mean(curvatures)
        median_c = np.median(curvatures)
        p90_c = np.percentile(curvatures, 90)
        ax.axvline(mean_c, color="gold", linestyle="solid", linewidth=1.5, label=f"Mean: {mean_c:.2f}")
        ax.axvline(median_c, color="orange", linestyle="dashed", linewidth=1.5, label=f"Median: {median_c:.2f}")
        ax.axvline(p90_c, color="crimson", linestyle="dotted", linewidth=1.5, label=f"p90: {p90_c:.2f}")
        ax.legend(facecolor="#0d1117", edgecolor="none", labelcolor="white", fontsize=9)
    
    ax.set_facecolor("#0d1117")
    ax.set_title(title, fontsize=11, fontweight="bold", color="white")
    ax.tick_params(colors="white")
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

def save_baseline_plots(base, full, lora, E, uc, rb, r0, device):
    os.makedirs("results/figures", exist_ok=True)
    plt.style.use('dark_background')
    
    # 1. Boundaries
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0d1117")
    plot_boundary(axes[0], base, E, uc, rb, r0, device, "Base Model")
    plot_boundary(axes[1], full, E, uc, rb, r0, device, "Full FT")
    plot_boundary(axes[2], lora, E, uc, rb, r0, device, "LoRA")
    plt.tight_layout()
    plt.savefig("results/figures/baseline_boundaries.png", dpi=150, facecolor="#0d1117")
    plt.close()
    
    # 2. Gate Drift Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor="#0d1117")
    plot_drift_heatmap(axes[0], base, full, E, uc, rb, r0, device, "Gate Drift (Full FT)")
    plot_drift_heatmap(axes[1], base, lora, E, uc, rb, r0, device, "Gate Drift (LoRA)")
    plt.tight_layout()
    plt.savefig("results/figures/gate_drift_heatmaps.png", dpi=150, facecolor="#0d1117")
    plt.close()

    # 3. Curvature Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0d1117")
    plot_curvature_histogram(axes[0], full, E, device, "Boundary Curvature (Full FT)")
    plot_curvature_histogram(axes[1], lora, E, device, "Boundary Curvature (LoRA)")
    plt.tight_layout()
    plt.savefig("results/figures/curvature_histograms.png", dpi=150, facecolor="#0d1117")
    plt.close()
    
    print("Saved results/figures/baseline_boundaries.png")
    print("Saved results/figures/gate_drift_heatmaps.png")
    print("Saved results/figures/curvature_histograms.png")


def save_rank_sweep_plots(results):
    """Plot rank vs line_crossing, curvature, region_creation."""
    os.makedirs("results/figures", exist_ok=True)
    ranks = sorted(int(k.split("_")[1]) for k in results)
    line_crossings = [results[f"rank_{r}"]["line_crossing"] for r in ranks]
    curvatures = [results[f"rank_{r}"]["curvature"]["mean"] for r in ranks]
    region_creation = [results[f"rank_{r}"]["region_count"]["new"] for r in ranks]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#0d1117")

    specs = [
        (line_crossings, "Line Crossings / Line", "Rank vs Line Crossing", "teal"),
        (curvatures, "Mean Curvature (rad)", "Rank vs Curvature", "gold"),
        (region_creation, "New Regions Created", "Rank vs Region Creation", "crimson"),
    ]
    for ax, (values, ylabel, title, color) in zip(axes, specs):
        ax.plot(ranks, values, "o-", color=color, linewidth=2, markersize=8)
        ax.set_xlabel("LoRA Rank", color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.set_title(title, fontsize=11, fontweight="bold", color="white")
        ax.set_xticks(ranks)
        ax.tick_params(colors="white")
        ax.set_facecolor("#0d1117")
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/figures/rank_sweep_complexity.png", dpi=150, facecolor="#0d1117")
    plt.close()
    print("Saved results/figures/rank_sweep_complexity.png")


def save_depth_geometry_plots(results):
    """Plot depth vs curvature, line_crossing, region_creation for Full FT and LoRA."""
    os.makedirs("results/figures", exist_ok=True)
    depths = sorted(int(k.split("_")[1]) for k in results)

    def _extract(method):
        return {
            "curvature": [results[f"depth_{d}"][method]["curvature"]["mean"] for d in depths],
            "line_crossing": [results[f"depth_{d}"][method]["line_crossing"] for d in depths],
            "region_new": [results[f"depth_{d}"][method]["region_count"]["new"] for d in depths],
        }

    full_vals = _extract("full_ft")
    lora_vals = _extract("lora")

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#0d1117")

    specs = [
        ("curvature", "Mean Curvature (rad)", "Depth vs Curvature"),
        ("line_crossing", "Line Crossings / Line", "Depth vs Line Crossing"),
        ("region_new", "New Regions Created", "Depth vs Region Creation"),
    ]
    for ax, (key, ylabel, title) in zip(axes, specs):
        ax.plot(depths, full_vals[key], "o-", color="#e05c5c", linewidth=2, markersize=8, label="Full FT")
        ax.plot(depths, lora_vals[key], "s-", color="teal", linewidth=2, markersize=8, label="LoRA")
        ax.set_xlabel("Network Depth", color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.set_title(title, fontsize=11, fontweight="bold", color="white")
        ax.set_xticks(depths)
        ax.tick_params(colors="white")
        ax.set_facecolor("#0d1117")
        ax.legend(facecolor="#0d1117", edgecolor="none", labelcolor="white", fontsize=9)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/figures/depth_geometry.png", dpi=150, facecolor="#0d1117")
    plt.close()
    print("Saved results/figures/depth_geometry.png")


def save_random_lowrank_plots(results):
    """Bar chart comparing Full FT, LoRA, and Random Low-Rank across geometric metrics."""
    os.makedirs("results/figures", exist_ok=True)
    labels = ["Full FT", "LoRA", "Random\nLow-Rank"]
    keys = ["full_ft", "lora", "random_low_rank"]
    colors = ["#e05c5c", "teal", "#9370db"]

    line_crossings = [results[k]["line_crossing"] for k in keys]
    gate_drifts = [results[k]["gate_drift"] for k in keys]
    region_new = [results[k]["region_count"]["new"] for k in keys]
    curv_mean = [results[k]["curvature"]["mean"] for k in keys]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), facecolor="#0d1117")

    specs = [
        (line_crossings, "Line Crossings / Line", "Line Crossing"),
        (gate_drifts, "Gate Drift (fraction)", "Gate Drift"),
        (region_new, "New Regions Created", "Region Creation"),
        (curv_mean, "Mean Curvature (rad)", "Mean Curvature"),
    ]
    for ax, (vals, ylabel, title) in zip(axes, specs):
        ax.bar(labels, vals, color=colors, alpha=0.85)
        ax.set_ylabel(ylabel, color="white")
        ax.set_title(title, fontsize=10, fontweight="bold", color="white")
        ax.tick_params(colors="white")
        ax.set_facecolor("#0d1117")
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/figures/random_lowrank_comparison.png", dpi=150, facecolor="#0d1117")
    plt.close()
    print("Saved results/figures/random_lowrank_comparison.png")
