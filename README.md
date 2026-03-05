# LoRA Polytope Complexity

> **Hypothesis:** Low-Rank Adaptation (LoRA) enforces highly correlated, low-dimensional rotations of ReLU gate hyperplanes (normals), rather than allowing the independent, local fragmentation that Full Fine-Tuning produces.

---

## Overview

This repository contains a rigorous PyTorch experiment that studies the **geometry of ReLU neural network adaptation**. The core question is:

*When adapting a pretrained MLP to a new task, does the rank constraint in LoRA fundamentally change how the decision-boundary geometry evolves, compared to unconstrained Full Fine-Tuning?*

We answer this through a controlled topological experiment: a "Bubble Flip" (XOR patch) is surgically placed inside a region of constant ReLU gate activation, making it geometrically provable that the output head alone cannot solve the task — hyperplane movement in `fc1` is mandatory.

---

## Architecture

```
Input x ∈ ℝ¹⁰  (data lives on a 2D plane embedded via fixed E : ℝ¹⁰ → ℝ²)
    │
    ▼
fc1: Linear(10 → 32)  ← Normals (rows of W₁) define ReLU hyperplanes
    │
  ReLU
    │
    ▼
fc2: Linear(32 → 1)   ← Binary classification head
    │
    ▼
  ŷ ∈ {0, 1}
```

- **Embedding:** Input `x = E @ u`, where `u ∈ ℝ²` is the true 2D coordinate.
  `E ∈ ℝ^(10×2)` is a fixed orthonormal basis (QR-factored random matrix).
- **Activation:** `nn.ReLU()` (strict; no leaky/ELU variants).

---

## Tasks

| Task | Description |
|------|------------|
| **Task 0** (Pretraining) | Binary circle: `‖u‖ < 0.60` → class 1 |
| **Task 1** (Adaptation)  | Circle **plus** a "Bubble Flip" XOR patch of radius `0.08` placed at a **constant-gate region** inside the circle |

The XOR patch flips labels for points within radius `rb=0.08` of the bubble center `u_c`.

---

## The 5 Mandatory Experimental Safeguards

1. **Topological Trap** — After pretraining, `find_bubble_center()` searches for a point `u_c` where all 32 ReLU gates of `fc1` remain constant throughout the entire bubble. This guarantees the geometry is locally flat.

2. **Head-Only Assert** — The base model (no training) is evaluated on the bubble region under Task 1. Since the gate pattern is constant and the label is flipped, accuracy = 0.0. This *proves* fc2 alone cannot solve it.

3. **Frozen Output & Bias** — During Full FT and LoRA: `fc2.weight`, `fc2.bias`, and `fc1.bias` are all frozen. Only the **normal vectors** (columns of `fc1.weight`) are allowed to change.

4. **Target-Loss Matching** — Both methods train until reaching the same target loss (`0.15`) rather than a fixed epoch count — ensuring no unfair comparison of underfitting vs. convergence.

5. **LoRA on fc1 only** — `W_eff = W₀ + (α/r) · B @ A`, where `A ∈ ℝ^(r×d_in)`, `B ∈ ℝ^(m×r)`, `r=2`. The update is provably rank-2.

---

## The 4 Core Metrics

| Level | Metric | What it measures |
|-------|--------|-----------------|
| 1 | **Update-Matrix-Rank** | `rank(ΔW)` and `stable_rank(ΔW)` in weight space |
| 2 | **Hyperplane-Rotation-Rank** | `stable_rank(ΔN)` where `N` = unit-normalized rows of `W₁` |
| 3 | **Gate-Drift** | % of 2D grid where ReLU gate pattern changed between base and adapted model |
| 4 | **Line-Crossing Complexity** | Crofton-proxy: avg. sign changes of prediction across 256 random lines |

---

## Setup

```bash
# Create conda environment
conda create -n plora python=3.12 -y
conda activate plora

# Install dependencies
pip install -r requirements.txt

# Run experiment
python experiment.py
```

Outputs:
- `results.json` — machine-readable numeric results
- `figures/decision_boundaries.png`
- `figures/gate_drift.png`
- `figures/sv_spectra.png`
- `figures/metric_summary.png`

---

## Key Results (Seed 42)

| Metric | Full FT | LoRA (r=2) |
|--------|---------|-----------|
| `rank(ΔW)` | 10 | **2** |
| `stable_rank(ΔW)` | 1.11 | **1.00** |
| Rotation rank (10D) | 10 | 10 |
| Rotation stable-rank (10D) | 1.23 | 1.25 |
| Gate-Drift | 40.3% | 43.2% |
| Line-Crossings/Line | 1.22 | **1.12** |
| Task-1 Global Acc | 0.970 | 0.965 |
| Task-1 Bubble Acc | 0.864 | **0.924** |

See [`evaluation.md`](evaluation.md) for full analysis and figures.

---

## Repository Structure

```
.
├── experiment.py       # Main experiment script
├── requirements.txt    # Python dependencies
├── results.json        # Numeric outputs (auto-generated)
├── figures/            # Visualisation outputs (auto-generated)
│   ├── decision_boundaries.png
│   ├── gate_drift.png
│   ├── sv_spectra.png
│   └── metric_summary.png
├── evaluation.md       # Scientific report & interpretation
└── README.md           # This file
```

---

## Citation / Reference

Experiment design inspired by:
- Hu et al., ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) (2021)
- Montufar et al., ["On the Number of Linear Regions of Deep Neural Networks"](https://arxiv.org/abs/1402.1869) (2014)
- Raghu et al., ["On the Expressive Power of Deep Neural Networks"](https://arxiv.org/abs/1606.05336) (2017)
