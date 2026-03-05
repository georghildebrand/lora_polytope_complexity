import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_random_low_rank_update(W0, rank, scale, seed=None):
    """
    Generate a random low-rank weight update ΔW = scale * (B @ A) and apply it to W0.

    A ∈ R^(rank × d_in)
    B ∈ R^(m_hidden × rank)

    Returns:
        W_updated: W0 + ΔW
        dW:        the update ΔW
    """
    m_hidden, d_in = W0.shape

    # Save and restore global RNG state so this function is side-effect-free.
    rng_state = None
    if seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)

    A = torch.randn(rank, d_in, dtype=W0.dtype, device=W0.device)
    B = torch.randn(m_hidden, rank, dtype=W0.dtype, device=W0.device)

    if rng_state is not None:
        torch.set_rng_state(rng_state)

    dW = scale * (B @ A)
    return W0 + dW, dW


class RandomLowRankMLP(nn.Module):
    """
    MLP whose fc1 weight is replaced by W0 + scale * (B @ A) with random B, A.
    No training is performed — this is a static random perturbation used as a
    geometric control baseline.
    """

    def __init__(self, base, rank, scale, seed=None):
        super().__init__()
        self.m_hidden = base.fc1.out_features

        W0 = base.fc1.weight.detach()
        W_eff, _ = apply_random_low_rank_update(W0, rank, scale, seed=seed)

        self.W_eff = nn.Parameter(W_eff, requires_grad=False)
        self.b0 = nn.Parameter(base.fc1.bias.detach().clone(), requires_grad=False)

        self.fc2 = nn.Linear(self.m_hidden, 1, bias=True)
        self.fc2.weight.data.copy_(base.fc2.weight.detach())
        self.fc2.bias.data.copy_(base.fc2.bias.detach())

    def forward(self, x):
        return self.fc2(F.relu(F.linear(x, self.W_eff, self.b0))).squeeze(-1)

    def effective_W(self):
        return self.W_eff

    @torch.no_grad()
    def gate_pattern(self, x):
        return F.linear(x, self.W_eff, self.b0) > 0
