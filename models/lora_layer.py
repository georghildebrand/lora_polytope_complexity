import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLP

class LoRAFirstLayerMLP(nn.Module):
    def __init__(self, base: MLP, r: int, alpha: float):
        super().__init__()
        self.d_in     = base.fc1.in_features
        self.m_hidden = base.fc1.out_features
        self.scaling  = alpha / max(1, r)

        self.W0 = nn.Parameter(base.fc1.weight.detach().clone(), requires_grad=False)
        self.b0 = nn.Parameter(base.fc1.bias.detach().clone(),   requires_grad=False)

        self.A = nn.Parameter(torch.empty(r, self.d_in))
        self.B = nn.Parameter(torch.zeros(self.m_hidden, r))
        nn.init.normal_(self.A, std=1e-3)

        self.fc2 = nn.Linear(self.m_hidden, 1, bias=True)
        self.fc2.weight.data.copy_(base.fc2.weight.detach())
        self.fc2.bias.data.copy_(base.fc2.bias.detach())

    def effective_W(self):
        return self.W0 + self.scaling * (self.B @ self.A)

    def forward(self, x):
        return self.fc2(F.relu(F.linear(x, self.effective_W(), self.b0))).squeeze(-1)

    @torch.no_grad()
    def gate_pattern(self, x):
        return F.linear(x, self.effective_W(), self.b0) > 0
