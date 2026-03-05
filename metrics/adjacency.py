import torch
import numpy as np

@torch.no_grad()
def polytope_adjacency_graph_drift(base_m, adapt_m, E, device, resolution=150):
    """
    Measures how the Polytope Adjacency Graph changes between the base and adapted model.
    In a ReLU network, each constant-gate region is a polytope. Two polytopes are
    adjacent if their gate patterns differ by exactly 1.
    We scan a 2D grid, identify adjacent regions, and calculate the Jaccard distance 
    between the sets of adjacency edges in the base vs adapted models.
    """
    t = torch.linspace(-1.5, 1.5, resolution, device=device)
    Ux, Uy = torch.meshgrid(t, t, indexing="xy")
    U = torch.stack([Ux.reshape(-1), Uy.reshape(-1)], dim=1)
    
    # Project 2D coordinates to embedding space
    X = (E @ U.T).T
    
    gb = base_m.gate_pattern(X)
    ga = adapt_m.gate_pattern(X)
    
    def get_edges(g):
        # g is (resolution^2, m_hidden) bool tensor
        g_grid = g.reshape(resolution, resolution, -1).int()
        
        # Horizontal diffs
        diff_h = torch.abs(g_grid[:, :-1] - g_grid[:, 1:]).sum(dim=-1)
        # Vertical diffs
        diff_v = torch.abs(g_grid[:-1, :] - g_grid[1:, :]).sum(dim=-1)
        
        edges = set()
        
        # Horizontal edges where hamming distance == 1
        h_idx = torch.where(diff_h == 1)
        for r, c in zip(h_idx[0].tolist(), h_idx[1].tolist()):
            p1 = tuple(g_grid[r, c].tolist())
            p2 = tuple(g_grid[r, c+1].tolist())
            edges.add(tuple(sorted((p1, p2))))
            
        # Vertical edges where hamming distance == 1
        v_idx = torch.where(diff_v == 1)
        for r, c in zip(v_idx[0].tolist(), v_idx[1].tolist()):
            p1 = tuple(g_grid[r, c].tolist())
            p2 = tuple(g_grid[r+1, c].tolist())
            edges.add(tuple(sorted((p1, p2))))
            
        return edges

    eb = get_edges(gb)
    ea = get_edges(ga)
    
    if len(eb) == 0 and len(ea) == 0:
        return 0.0
        
    intersection = len(eb.intersection(ea))
    union = len(eb.union(ea))
    
    jaccard_distance = 1.0 - (intersection / union)
    return float(jaccard_distance)
