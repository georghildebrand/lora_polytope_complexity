import torch
import os
import json
from models.deep_mlp import DeepMLP, LoRAFirstLayerDeepMLP
from models.utils import (set_seed, make_embedding, find_bubble_center, 
                          train_base, train_until_loss, y_circle, y_bubble_flip,
                          eval_bubble_acc)
from metrics.rotation_rank import matrix_rank, stable_rank, get_W1

def run_depth_study(seed=42):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_in = 10
    width = 32
    r0 = 0.6
    rb = 0.08
    lora_r = 2
    target_loss = 0.15
    depths = [1, 4, 8]
    
    all_results = {}

    for d in depths:
        print(f"\n=== Running Depth: {d} ===")
        set_seed(seed)
        E = make_embedding(d_in, device, seed=seed)
        
        # Pretrain
        base = DeepMLP(d_in, width, d).to(device)
        train_base(base, E, lambda u: y_circle(u, r0), device)
        uc = find_bubble_center(base, E, r0, rb, device)
        
        y_fn = lambda u: y_bubble_flip(u, uc, rb, r0)

        # Full FT (only fc1 of DeepMLP)
        full = DeepMLP(d_in, width, d).to(device)
        full.load_state_dict(base.state_dict())
        # Freeze everything except first layer
        for p in full.parameters():
            p.requires_grad = False
        full.model[0].weight.requires_grad = True # Only fc1 weight
        
        train_until_loss(full, E, y_fn, uc, rb, device, target_loss=target_loss)

        # LoRA (only on first layer)
        lora = LoRAFirstLayerDeepMLP(base, r=lora_r, alpha=2.0).to(device)
        train_until_loss(lora, E, y_fn, uc, rb, device, target_loss=target_loss)

        W0 = get_W1(base)
        dW_full = get_W1(full) - W0
        dW_lora = get_W1(lora) - W0
        
        all_results[f"depth_{d}"] = {
            "full_ft": {
                "rank": matrix_rank(dW_full),
                "stable_rank": stable_rank(dW_full),
                "acc": eval_bubble_acc(full, E, uc, rb, y_fn, device)
            },
            "lora": {
                "rank": matrix_rank(dW_lora),
                "stable_rank": stable_rank(dW_lora),
                "acc": eval_bubble_acc(lora, E, uc, rb, y_fn, device)
            }
        }

    os.makedirs("results/logs", exist_ok=True)
    with open("results/logs/depth_study_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("\nDepth study results saved.")

if __name__ == "__main__":
    run_depth_study()
