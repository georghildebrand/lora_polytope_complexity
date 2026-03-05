import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepMLP(nn.Module):
    def __init__(self, d_in, width, depth, bias=True):
        super().__init__()
        layers = []
        layers.append(nn.Linear(d_in, width, bias=bias))
        layers.append(nn.ReLU())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width, bias=bias))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, 1, bias=bias))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)

    @torch.no_grad()
    def gate_pattern(self, x):
        # We define gate pattern as the activation of the FIRST hidden layer for consistency
        return self.model[0](x) > 0

class LoRAFirstLayerDeepMLP(nn.Module):
    def __init__(self, base: DeepMLP, r: int, alpha: float):
        super().__init__()
        first_linear = base.model[0]
        self.d_in     = first_linear.in_features
        self.width    = first_linear.out_features
        self.scaling  = alpha / max(1, r)

        self.W0 = nn.Parameter(first_linear.weight.detach().clone(), requires_grad=False)
        self.b0 = nn.Parameter(first_linear.bias.detach().clone(),   requires_grad=False)

        self.A = nn.Parameter(torch.empty(r, self.d_in))
        self.B = nn.Parameter(torch.zeros(self.width, r))
        nn.init.normal_(self.A, std=1e-3)

        # Clone and freeze the rest of the model
        import copy
        self.rest = copy.deepcopy(base.model[1:])
        for p in self.rest.parameters():
            p.requires_grad = False

    def effective_W(self):
        return self.W0 + self.scaling * (self.B @ self.A)

    def forward(self, x):
        h = F.relu(F.linear(x, self.effective_W(), self.b0))
        return self.rest(h).squeeze(-1)

    @torch.no_grad()
    def gate_pattern(self, x):
        return F.linear(x, self.effective_W(), self.b0) > 0
