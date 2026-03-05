# LoRA Polytope Complexity

> **Hypothesis:** Low-Rank Adaptation (LoRA) enforces highly correlated, low-dimensional rotations of ReLU gate hyperplanes (normals), rather than allowing the independent, local fragmentation that Full Fine-Tuning produces.

---

## Overview

This repository contains a rigorous PyTorch experiment that studies the **geometry of ReLU neural network adaptation**. The core question is:

*When adapting a pretrained MLP to a new task, does the rank constraint in LoRA fundamentally change how the decision-boundary geometry evolves, compared to unconstrained Full Fine-Tuning?*

We answer this through a controlled topological experiment: a "Bubble Flip" (XOR patch) is surgically placed inside a region of constant ReLU gate activation, making it geometrically provable that the output head alone cannot solve the task — hyperplane movement in `fc1` is mandatory.

## Scientific Statement

This repository provides geometric evidence supporting the claim: **Low-rank adaptation restricts the dimension of the hyperplane deformation field.** 

This is stronger and more mathematically precise than the typical statement that "LoRA regularizes fine-tuning" (which is vague). Low-dimensional deformation is measurable, as shown in this repo.

---

## Interpretation of the Metrics

⚠️ **Important Clarification:** This repository **does NOT measure the theoretical expressivity** of neural networks. 

Instead, it evaluates **how the decision partition deforms during adaptation.** Low-rank updates restrict the directions in which hyperplane normals can move. This induces a correlated deformation of the partition, rather than the chaotic, independent motion of individual hyperplanes seen in Full Fine-Tuning.

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
| 3 | **Gate-Drift / Polytope Adjacency** | % of area changed; Jaccard distance of the Polytope Adjacency Graph before/after adaptation |
| 4 | **Discrete Boundary Curvature** | Mean, median, and 90th percentile (p90) turning angles of the boundary's geometric normal vector. |
| 5 | **Line-Crossing Complexity** | Crofton-proxy: avg. sign changes of prediction across 256 random lines |

### Curvature Distribution
Decision boundaries often contain rare but large kinks. Therefore, we report not only the mean curvature, but also the median and the 90th percentile of turning angles across generated boundary points.

### Theoretical Connection: Tropical Geometry
The rotation-rank metric is essentially measuring the **dimension of the normal fan deformation of the ReLU partition**. In terms of tropical and polyhedral geometry: 
`rank(ΔW) → dimension of Newton polytope deformation`.
LoRA's low-rank structure geometrically bounds the complexity of the topological changes applied to the data manifold.

---

## Setup

```bash
# Create conda environment
conda create -n plora python=3.12 -y
conda activate plora

# Install dependencies
pip install -r requirements.txt

# Run baseline experiment
python -m experiments.run_baseline

# Run bias ablation
python -m experiments.run_bias_ablation

# Run depth study
python -m experiments.run_depth
```

Outputs:
- `results/logs/` — machine-readable numeric results (JSON)
- `results/figures/` — generated plots

---

## Key Results (Seed 42)

| Metric | Full FT | LoRA (r=2) |
|--------|---------|-----------|
| `rank(ΔW)` | 10 | **2** |
| `stable_rank(ΔW)` | 1.11 | **1.00** |
| Rotation rank (10D) | 10 | 10 |
| Rotation stable-rank (10D) | 1.23 | 1.25 |
| Gate-Drift | 62.3% | **56.0%** |
| Adjacency Graph Drift | 63.9% | **50.8%** |
| Boundary Curvature (mean) | 2.03 rad | **1.98 rad** |
| Line-Crossings/Line | 1.18 | **1.13** |
| Task-1 Global Acc | 0.970 | 0.965 |
| Task-1 Bubble Acc | 0.937 | 0.932 |

See [`evaluation.md`](evaluation.md) for full analysis and figures.

---

## Experimental Extensions (Depth Ablation)

One conceptual limitation of the 1-layer architecture is that it isolates geometry nicely, but leaves open the question: *Does the effect survive deep composition?* Deep ReLU networks can create exponentially many regions.

To address this, the codebase includes `experiments/run_depth.py`, where LoRA is applied **only to the first layer** across `depth = 1`, `4`, and `8` networks to answer: *Does low-rank deformation propagate through depth?*

---

## Repository Structure

```
.
├── models/
│   ├── mlp.py              # Base MLP model
│   ├── lora_layer.py       # LoRA implementation for first layer
│   ├── deep_mlp.py         # Deep MLP model and LoRA variant
│   └── utils.py            # Data sampling and training utilities
├── metrics/
│   ├── gate_drift.py       # ReLU gate pattern drift metric
│   ├── line_crossing.py    # Crofton complexity proxy
│   ├── rotation_rank.py    # Update rank and rotation metrics
│   └── curvature.py        # Discrete boundary curvature
├── experiments/
│   ├── run_baseline.py      # Original experiment replicating baseline
│   ├── run_bias_ablation.py # Bias frozen vs trainable ablation
│   └── run_depth.py         # Study on effects of network depth
├── scripts/
│   └── plot_results.py      # Plotting and visualization suite
├── results/
│   ├── logs/               # JSON results and logs
│   └── figures/            # Generated plots
├── requirements.txt        # Python dependencies
├── evaluation.md           # Scientific report & interpretation
└── README.md               # This file
```

---

## Citation / Reference

Experiment design inspired by:
- Hu et al., ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) (2021)
- Montufar et al., ["On the Number of Linear Regions of Deep Neural Networks"](https://arxiv.org/abs/1402.1869) (2014)
- Raghu et al., ["On the Expressive Power of Deep Neural Networks"](https://arxiv.org/abs/1606.05336) (2017)
