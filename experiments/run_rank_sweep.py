"""
Rank Sweep Experiment
=====================
Study how LoRA rank affects geometric boundary complexity.

Tests ranks r = [1, 2, 4, 8] and measures:
  - stable_rank(ΔW)
  - gate_drift
  - adjacency_drift
  - line_crossing
  - curvature (mean, median, p90)
  - region_count

Results are saved to results/logs/rank_sweep_results.json.
Plots are saved to results/figures/rank_sweep_complexity.png.
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
from scripts.plot_results import save_rank_sweep_plots


def run_rank_sweep(seed=42, d_in=10, m_hidden=32, r0=0.6, rb=0.08, target_loss=0.15):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ranks = [1, 2, 4, 8]
    all_results = {}

    set_seed(seed)
    E = make_embedding(d_in, device, seed=seed)

    # Pretrain base model once; reuse across all rank conditions.
    print("[1/2] Pretraining base model...")
    base = MLP(d_in, m_hidden).to(device)
    train_base(base, E, lambda u: y_circle(u, r0), device)
    uc = find_bubble_center(base, E, r0, rb, device)
    y_fn = lambda u: y_bubble_flip(u, uc, rb, r0)

    print("[2/2] Sweeping LoRA ranks...")
    for r in ranks:
        print(f"\n  --- Rank r={r} ---")
        set_seed(seed + r)  # Fresh seed per rank for fair comparison.
        lora = LoRAFirstLayerMLP(base, r=r, alpha=float(r)).to(device)
        lora.fc2.weight.requires_grad = False
        lora.fc2.bias.requires_grad = False
        train_until_loss(lora, E, y_fn, uc, rb, device, target_loss=target_loss)

        W0 = get_W1(base)
        dW = get_W1(lora) - W0
        regions = count_regions_and_overlap(base, lora, E, device, resolution=1000)
        curv = boundary_discrete_curvature(lora, E, device)

        all_results[f"rank_{r}"] = {
            "rank_r": r,
            "rank_dW": matrix_rank(dW),
            "stable_rank_dW": stable_rank(dW),
            "gate_drift": measure_gate_drift(base, lora, E, device),
            "adjacency_drift": polytope_adjacency_graph_drift(base, lora, E, device),
            "line_crossing": line_crossing_complexity(lora, E, device),
            "curvature": curv,
            "region_count": regions,
        }

        entry = all_results[f"rank_{r}"]
        print(f"    stable_rank={entry['stable_rank_dW']:.3f}, "
              f"line_crossing={entry['line_crossing']:.3f}, "
              f"curvature_mean={curv['mean']:.3f}, "
              f"regions_new={regions['new']}")

    os.makedirs("results/logs", exist_ok=True)
    with open("results/logs/rank_sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("\nRank sweep results saved to results/logs/rank_sweep_results.json")

    save_rank_sweep_plots(all_results)
    return all_results


if __name__ == "__main__":
    run_rank_sweep()
