import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models.utils import eval_acc, to_x

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

def save_baseline_plots(base, full, lora, E, uc, rb, r0, device):
    os.makedirs("results/figures", exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#0d1117")
    
    plot_boundary(axes[0], base, E, uc, rb, r0, device, "Base Model")
    plot_boundary(axes[1], full, E, uc, rb, r0, device, "Full FT")
    plot_boundary(axes[2], lora, E, uc, rb, r0, device, "LoRA")
    
    # Placeholder for a metric plot or residual
    axes[3].axis("off") # Or add more comparison logic
    
    plt.tight_layout()
    plt.savefig("results/figures/baseline_boundaries.png", dpi=150, facecolor="#0d1117")
    plt.close()
    print("Saved results/figures/baseline_boundaries.png")
