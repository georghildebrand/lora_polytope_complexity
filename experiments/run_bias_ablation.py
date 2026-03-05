import torch
import os
import json
from models.mlp import MLP
from models.lora_layer import LoRAFirstLayerMLP
from models.utils import (set_seed, make_embedding, find_bubble_center, 
                          train_base, train_until_loss, y_circle, y_bubble_flip)
from metrics.rotation_rank import matrix_rank, stable_rank, get_W1

def run_bias_ablation(seed=42):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_in = 10
    m_hidden = 32
    r0 = 0.6
    rb = 0.08
    lora_r = 2
    target_loss = 0.15
    
    modes = ["frozen", "trainable"]
    all_results = {}

    for mode in modes:
        print(f"\n=== Running Bias Mode: {mode} ===")
        set_seed(seed)
        E = make_embedding(d_in, device, seed=seed)
        
        # Pretrain
        base = MLP(d_in, m_hidden).to(device)
        train_base(base, E, lambda u: y_circle(u, r0), device)
        uc = find_bubble_center(base, E, r0, rb, device)
        
        # Adaptation tasks
        y_fn = lambda u: y_bubble_flip(u, uc, rb, r0)

        # Full FT
        full = MLP(d_in, m_hidden).to(device)
        full.load_state_dict(base.state_dict())
        full.fc1.bias.requires_grad = (mode == "trainable")
        full.fc2.weight.requires_grad = False
        full.fc2.bias.requires_grad = False
        train_until_loss(full, E, y_fn, uc, rb, device, target_loss=target_loss)

        # LoRA
        # Note: LoRAFirstLayerMLP as implemented uses the fixed base.fc1.bias (b0)
        # We need to decide if the bias ablation applies to the LoRA wrapper too.
        # The prompt says: "Comparison of how rank(dW) and geometric metrics change"
        # Let's adjust LoRAFirstLayerMLP or handle bias separately.
        # Original LoRA layer has b0 as a Parameter(requires_grad=False).
        lora = LoRAFirstLayerMLP(base, r=lora_r, alpha=2.0).to(device)
        lora.b0.requires_grad = (mode == "trainable")
        lora.fc2.weight.requires_grad = False
        lora.fc2.bias.requires_grad = False
        train_until_loss(lora, E, y_fn, uc, rb, device, target_loss=target_loss)

        W0 = get_W1(base)
        dW_full = get_W1(full) - W0
        dW_lora = get_W1(lora) - W0
        
        all_results[mode] = {
            "full_ft": {
                "rank": matrix_rank(dW_full),
                "stable_rank": stable_rank(dW_full)
            },
            "lora": {
                "rank": matrix_rank(dW_lora),
                "stable_rank": stable_rank(dW_lora)
            }
        }

    os.makedirs("results/logs", exist_ok=True)
    with open("results/logs/bias_ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("\nAblation results saved.")

if __name__ == "__main__":
    run_bias_ablation()
