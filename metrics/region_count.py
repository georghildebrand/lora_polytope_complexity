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
        
    # Convert binary gate patterns to unique integer hashes explicitly
    # Assuming m_hidden <= 64, we can pack them into a 64-bit int
    powers = 2 ** torch.arange(gb.shape[1], device=device)
    
    hash_b = (gb.long() * powers).sum(dim=1)
    hash_a = (ga.long() * powers).sum(dim=1)
    
    # Use pure python sets for unique counting and set operations
    regions_base = set(hash_b.tolist())
    regions_adapted = set(hash_a.tolist())
    
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
