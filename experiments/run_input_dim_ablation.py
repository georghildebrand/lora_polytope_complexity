"""
Ambient Input Dimension Ablation
=================================
Tests whether the ambient dimension of the input space (d_in) changes the
observed geometric effects of LoRA vs Full FT.

Conditions:
  d_in = 2  — data lives directly in 2D; embedding E is a 2×2 orthonormal matrix
  d_in = 10 — data embedded into 10D via E ∈ R^(10×2) (standard setup)

The data manifold (2D circle + bubble task) is identical in both cases;
only the ambient parameter space changes.

Metrics compared per condition:
  - rotation_rank / rotation_stable_rank (ΔN in full space)
  - line_crossing
  - curvature
  - region_count (new regions created)

Results saved to results/logs/input_dim_ablation.json.
"""

import torch
import os
import json
from models.mlp import MLP
from models.lora_layer import LoRAFirstLayerMLP
from models.utils import (set_seed, make_embedding, find_bubble_center,
                          train_base, train_until_loss, y_circle, y_bubble_flip)
from metrics.line_crossing import line_crossing_complexity
from metrics.rotation_rank import matrix_rank, stable_rank, hyperplane_rotation_rank, get_W1
from metrics.curvature import boundary_discrete_curvature
from metrics.region_count import count_regions_and_overlap


def run_input_dim_ablation(seed=42, r0=0.6, rb=0.08, m_hidden=32,
                            lora_r=2, target_loss=0.15):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dims = [2, 10]
    all_results = {}

    for d_in in input_dims:
        print(f"\n=== d_in={d_in} ===")
        set_seed(seed)
        E = make_embedding(d_in, device, seed=seed)

        base = MLP(d_in, m_hidden).to(device)
        train_base(base, E, lambda u: y_circle(u, r0), device)
        uc = find_bubble_center(base, E, r0, rb, device)
        y_fn = lambda u: y_bubble_flip(u, uc, rb, r0)

        # Full FT
        full = MLP(d_in, m_hidden).to(device)
        full.load_state_dict(base.state_dict())
        full.fc1.bias.requires_grad = False
        full.fc2.weight.requires_grad = False
        full.fc2.bias.requires_grad = False
        train_until_loss(full, E, y_fn, uc, rb, device, target_loss=target_loss)

        # LoRA
        lora = LoRAFirstLayerMLP(base, r=lora_r, alpha=2.0).to(device)
        lora.fc2.weight.requires_grad = False
        lora.fc2.bias.requires_grad = False
        train_until_loss(lora, E, y_fn, uc, rb, device, target_loss=target_loss)

        def metrics_for(model):
            W0 = get_W1(base)
            dW = get_W1(model) - W0
            rot_rank, rot_sr = hyperplane_rotation_rank(base, model)
            regions = count_regions_and_overlap(base, model, E, device, resolution=1000)
            curv = boundary_discrete_curvature(model, E, device)
            return {
                "rank_dW": matrix_rank(dW),
                "stable_rank_dW": stable_rank(dW),
                "rotation_rank": rot_rank,
                "rotation_stable_rank": rot_sr,
                "line_crossing": line_crossing_complexity(model, E, device),
                "curvature": curv,
                "region_count": regions,
            }

        all_results[f"d_in_{d_in}"] = {
            "full_ft": metrics_for(full),
            "lora": metrics_for(lora),
        }

        print(f"  Full FT: rot_rank={all_results[f'd_in_{d_in}']['full_ft']['rotation_rank']}, "
              f"regions_new={all_results[f'd_in_{d_in}']['full_ft']['region_count']['new']}")
        print(f"  LoRA:    rot_rank={all_results[f'd_in_{d_in}']['lora']['rotation_rank']}, "
              f"regions_new={all_results[f'd_in_{d_in}']['lora']['region_count']['new']}")

    os.makedirs("results/logs", exist_ok=True)
    with open("results/logs/input_dim_ablation.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("\nInput dim ablation results saved to results/logs/input_dim_ablation.json")
    return all_results


if __name__ == "__main__":
    run_input_dim_ablation()
