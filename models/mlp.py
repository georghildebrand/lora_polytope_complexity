import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_in, m_hidden, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(d_in, m_hidden, bias=bias)
        self.fc2 = nn.Linear(m_hidden, 1, bias=bias)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x))).squeeze(-1)

    @torch.no_grad()
    def gate_pattern(self, x):
        return self.fc1(x) > 0
