import torch

def count_regions_and_overlap(model_base, model_adapted, E, device, resolution=1000):
    t = torch.linspace(-1.5, 1.5, resolution, device=device)
    Ux, Uy = torch.meshgrid(t, t, indexing="xy")
    U = torch.stack([Ux.reshape(-1), Uy.reshape(-1)], dim=1)
    
    with torch.no_grad():
        X = (E @ U.T).T
        # Get binary gate patterns for all points
        gb = model_base.gate_pattern(X)
        ga = model_adapted.gate_pattern(X)
        
    # Convert binary gate patterns to tuples for hashing — works for any hidden size
    regions_base = set(tuple(row.tolist()) for row in gb.cpu())
    regions_adapted = set(tuple(row.tolist()) for row in ga.cpu())
    
    shared_regions = regions_base.intersection(regions_adapted)
    new_regions = regions_adapted - regions_base
    lost_regions = regions_base - regions_adapted
    
    return {
        "base_total": len(regions_base),
        "adapted_total": len(regions_adapted),
        "shared": len(shared_regions),
        "new": len(new_regions),
        "lost": len(lost_regions)
    }
