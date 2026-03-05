import copy
import torch
import numpy as np

def boundary_discrete_curvature(model, E, device, n_lines=200, points_per_line=400):
    curvatures = []

    # Avoid PyTorch MPS segfaults with autograd by forcing CPU.
    # Use a deep copy so the original model is never moved.
    model_cpu = copy.deepcopy(model).to("cpu")
    E = E.to("cpu")
    device = "cpu"
    dtype = E.dtype
    
    for _ in range(n_lines):
        theta = torch.rand((), device=device, dtype=dtype) * 2 * np.pi
        direction = torch.tensor([torch.cos(theta), torch.sin(theta)], device=device, dtype=dtype)
        normal = torch.tensor([-direction[1], direction[0]], device=device, dtype=dtype)
        offset = (torch.rand((), device=device, dtype=dtype) * 2 - 1)

        t = torch.linspace(-1.5, 1.5, points_per_line, device=device, dtype=dtype)
        u = offset * normal + t.unsqueeze(1) * direction
        mask = (torch.abs(u) <= 1).all(dim=1)
        if mask.sum() < 20: continue
        u_valid = u[mask]

        # Fast pass
        with torch.no_grad():
            x_eval = (E @ u_valid.T).T
            logits = model_cpu(x_eval)
            s = (logits > 0).float()
            crossings = torch.where(s[1:] != s[:-1])[0]

        if len(crossings) < 2: continue

        normals = []
        for idx in crossings:
            # Interpolate boundary location exactly
            alpha = torch.abs(logits[idx]) / (torch.abs(logits[idx]) + torch.abs(logits[idx+1]) + 1e-12)
            x_cross = (1 - alpha) * x_eval[idx] + alpha * x_eval[idx+1]
            xi = x_cross.detach().clone().requires_grad_(True)

            with torch.enable_grad():
                f = model_cpu(xi.unsqueeze(0))
                g = torch.autograd.grad(f, xi)[0]
            n = g / (g.norm() + 1e-12)
            normals.append(n.detach().cpu())

        normals = torch.stack(normals)
        for i in range(len(normals) - 1):
            cos_val = torch.clamp(torch.dot(normals[i], normals[i+1]), -1.0, 1.0)
            angle = torch.acos(cos_val)
            curvatures.append(angle.item())

    if len(curvatures) == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}

    return {
        "mean": float(np.mean(curvatures)),
        "median": float(np.median(curvatures)),
        "p90": float(np.percentile(curvatures, 90))
    }
