"""
Random Low-Rank Control Experiment
===================================
Compare three conditions on the same bubble-flip task:
  1. Full Fine-Tuning
  2. LoRA (rank-2, trained)
  3. Random Low-Rank update (rank-2, random B@A, matched update norm, NOT trained)

The random low-rank control tests whether the geometric effects observed for
LoRA are due to the rank constraint alone or require the learned structure.

Results are saved to results/logs/random_lowrank_results.json.
Plots are saved to results/figures/random_lowrank_comparison.png.
"""

import torch
import os
import json
from models.mlp import MLP
from models.lora_layer import LoRAFirstLayerMLP
from models.utils import (set_seed, make_embedding, find_bubble_center,
                          train_base, train_until_loss, y_circle, y_bubble_flip)
from metrics.gate_drift import measure_gate_drift
from metrics.line_crossing import line_crossing_complexity
from metrics.rotation_rank import matrix_rank, stable_rank, get_W1
from metrics.curvature import boundary_discrete_curvature
from metrics.adjacency import polytope_adjacency_graph_drift
from metrics.region_count import count_regions_and_overlap
from metrics.random_low_rank import apply_random_low_rank_update, RandomLowRankMLP
from scripts.plot_results import save_random_lowrank_plots


def run_random_lowrank_control(seed=42, d_in=10, m_hidden=32, r0=0.6, rb=0.08,
                                lora_r=2, target_loss=0.15):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    E = make_embedding(d_in, device, seed=seed)

    # Pretrain base
    print("[1/4] Pretraining base model...")
    base = MLP(d_in, m_hidden).to(device)
    train_base(base, E, lambda u: y_circle(u, r0), device)
    uc = find_bubble_center(base, E, r0, rb, device)
    y_fn = lambda u: y_bubble_flip(u, uc, rb, r0)

    # Full FT
    print("[2/4] Full Fine-Tuning...")
    full = MLP(d_in, m_hidden).to(device)
    full.load_state_dict(base.state_dict())
    full.fc1.bias.requires_grad = False
    full.fc2.weight.requires_grad = False
    full.fc2.bias.requires_grad = False
    train_until_loss(full, E, y_fn, uc, rb, device, target_loss=target_loss)

    # LoRA
    print("[3/4] LoRA Fine-Tuning...")
    lora = LoRAFirstLayerMLP(base, r=lora_r, alpha=2.0).to(device)
    lora.fc2.weight.requires_grad = False
    lora.fc2.bias.requires_grad = False
    train_until_loss(lora, E, y_fn, uc, rb, device, target_loss=target_loss)

    # Random low-rank: matched scale to LoRA update norm
    print("[4/4] Building random low-rank control...")
    W0 = get_W1(base)
    dW_lora = get_W1(lora) - W0
    lora_norm = float(torch.norm(dW_lora).item())

    # Compute unit random low-rank delta to determine scale
    _, dW_unit = apply_random_low_rank_update(W0, lora_r, 1.0, seed=seed + 99)
    rand_scale = lora_norm / (float(torch.norm(dW_unit).item()) + 1e-12)
    rand_model = RandomLowRankMLP(base, rank=lora_r, scale=rand_scale,
                                  seed=seed + 99).to(device)

    def compute_metrics(model):
        dW = get_W1(model) - W0
        regions = count_regions_and_overlap(base, model, E, device, resolution=1000)
        curv = boundary_discrete_curvature(model, E, device)
        return {
            "rank_dW": matrix_rank(dW),
            "stable_rank_dW": stable_rank(dW),
            "gate_drift": measure_gate_drift(base, model, E, device),
            "adjacency_drift": polytope_adjacency_graph_drift(base, model, E, device),
            "line_crossing": line_crossing_complexity(model, E, device),
            "curvature": curv,
            "region_count": regions,
        }

    results = {
        "full_ft": compute_metrics(full),
        "lora": compute_metrics(lora),
        "random_low_rank": compute_metrics(rand_model),
    }

    os.makedirs("results/logs", exist_ok=True)
    with open("results/logs/random_lowrank_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Random low-rank control results saved to results/logs/random_lowrank_results.json")

    save_random_lowrank_plots(results)
    return results


if __name__ == "__main__":
    run_random_lowrank_control()
