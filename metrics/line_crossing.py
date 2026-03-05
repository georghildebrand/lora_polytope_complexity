import torch
import numpy as np

@torch.no_grad()
def line_crossing_complexity(model, E, device, n_lines=256, pts=256):
    model.eval()
    crossings = []
    dtype   = E.dtype
    for _ in range(n_lines):
        theta   = (torch.rand(1, device=device, dtype=dtype) * 2 * np.pi).item()
        dv      = torch.tensor([np.cos(theta), np.sin(theta)], device=device, dtype=dtype)
        nv      = torch.tensor([-np.sin(theta), np.cos(theta)], device=device, dtype=dtype)
        offset  = (torch.rand(1, device=device, dtype=dtype) * 2 - 1).item()
        t       = torch.linspace(-1.5, 1.5, pts, device=device, dtype=dtype)
        u       = offset * nv + t.unsqueeze(1) * dv
        mask    = (u.abs() <= 1).all(dim=1)
        if mask.sum() < 10:
            continue
        # Project u to x
        X = (E @ u[mask].T).T
        s = (model(X) > 0).float()
        crossings.append((s[1:] != s[:-1]).sum().item())
    return float(np.mean(crossings)) if crossings else 0.0
