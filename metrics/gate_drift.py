import torch

@torch.no_grad()
def measure_gate_drift(base_m, adapt_m, E, device, resolution=300):
    t  = torch.linspace(-1, 1, resolution, device=device)
    Ux, Uy = torch.meshgrid(t, t, indexing="xy")
    U  = torch.stack([Ux.reshape(-1), Uy.reshape(-1)], dim=1)
    X  = (E @ U.T).T
    gb = base_m.gate_pattern(X)
    ga = adapt_m.gate_pattern(X)
    return (gb ^ ga).any(dim=1).float().mean().item()
