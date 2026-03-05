import torch
import os
import json
import matplotlib.pyplot as plt
from models.mlp import MLP
from models.lora_layer import LoRAFirstLayerMLP
from models.utils import (set_seed, make_embedding, find_bubble_center, 
                          eval_acc, eval_bubble_acc, train_base, train_until_loss,
                          y_circle, y_bubble_flip)
from metrics.gate_drift import measure_gate_drift
from metrics.line_crossing import line_crossing_complexity
from metrics.rotation_rank import matrix_rank, stable_rank, hyperplane_rotation_rank, get_W1
from metrics.curvature import boundary_discrete_curvature
from metrics.adjacency import polytope_adjacency_graph_drift
from metrics.region_count import count_regions_and_overlap
from scripts.plot_results import save_baseline_plots

def run_experiment(seed=42, d_in=10, m_hidden=32, r0=0.6, rb=0.08, lora_r=2, target_loss=0.15):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    E = make_embedding(d_in, device, seed=seed)
    
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # 1. Pretrain
    print("\n[1/5] Pretraining Base Model...")
    base = MLP(d_in, m_hidden).to(device)
    train_base(base, E, lambda u: y_circle(u, r0), device)
    
    # 2. Find bubble
    print("\n[2/5] Searching for constant-gate bubble...")
    uc = find_bubble_center(base, E, r0, rb, device)
    
    # 3. Assert head-only
    print("\n[3/5] Head-only check...")
    # Base accuracy on task 1
    acc_base_bubble = eval_bubble_acc(base, E, uc, rb, lambda u: y_bubble_flip(u, uc, rb, r0), device)
    if acc_base_bubble > 0.75:
        raise RuntimeError(f"Base accuracy too high: {acc_base_bubble}")
    print("    ✓ Assert Passed")

    # 4. Full FT
    print("\n[4/5] Full Fine-Tuning...")
    full = MLP(d_in, m_hidden).to(device)
    full.load_state_dict(base.state_dict())
    full.fc1.bias.requires_grad = False
    full.fc2.weight.requires_grad = False
    full.fc2.bias.requires_grad = False
    train_until_loss(full, E, lambda u: y_bubble_flip(u, uc, rb, r0), uc, rb, device, target_loss=target_loss)

    # 5. LoRA
    print("\n[5/5] LoRA Fine-Tuning...")
    lora = LoRAFirstLayerMLP(base, r=lora_r, alpha=2.0).to(device)
    lora.fc2.weight.requires_grad = False
    lora.fc2.bias.requires_grad = False
    train_until_loss(lora, E, lambda u: y_bubble_flip(u, uc, rb, r0), uc, rb, device, target_loss=target_loss)

    # Metrics
    W0 = get_W1(base)
    dW_full = get_W1(full) - W0
    dW_lora = get_W1(lora) - W0
    
    results = {
        "full_ft": {
            "rank": matrix_rank(dW_full),
            "stable_rank": stable_rank(dW_full),
            "gate_drift": measure_gate_drift(base, full, E, device),
            "adjacency_drift": polytope_adjacency_graph_drift(base, full, E, device),
            "line_crossing": line_crossing_complexity(full, E, device),
            "curvature": boundary_discrete_curvature(full, E, device),
            "acc_bubble": eval_bubble_acc(full, E, uc, rb, lambda u: y_bubble_flip(u, uc, rb, r0), device)
        },
        "lora": {
            "rank": matrix_rank(dW_lora),
            "stable_rank": stable_rank(dW_lora),
            "gate_drift": measure_gate_drift(base, lora, E, device),
            "adjacency_drift": polytope_adjacency_graph_drift(base, lora, E, device),
            "line_crossing": line_crossing_complexity(lora, E, device),
            "curvature": boundary_discrete_curvature(lora, E, device),
            "acc_bubble": eval_bubble_acc(lora, E, uc, rb, lambda u: y_bubble_flip(u, uc, rb, r0), device)
        }
    }
    
    # Region Creation vs Region Movement
    print("\n   => Region Counting (Creation vs Movement)...")
    results["full_ft"]["regions"] = count_regions_and_overlap(base, full, E, device, resolution=1000)
    results["lora"]["regions"] = count_regions_and_overlap(base, lora, E, device, resolution=1000)
    
    with open("results/logs/baseline_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    save_baseline_plots(base, full, lora, E, uc, rb, r0, device)
        
    print("\nResults saved to results/logs/baseline_results.json")
    return results

if __name__ == "__main__":
    run_experiment()
