import torch
from metrics.rotation_rank import get_W1


def normal_motion_rank(base_model, adapted_model, tol=1e-6):
    """
    Measure the rank and stable rank of the normal-vector movement field.

    Steps:
    1. Extract W1 from base and adapted model.
    2. Normalize rows to unit vectors (hyperplane normals).
    3. Compute ΔN = N1 - N0.
    4. Compute rank(ΔN) and stable_rank(ΔN).

    Returns a dict with keys 'rank' and 'stable_rank'.
    """
    W0 = get_W1(base_model).to("cpu")
    W1 = get_W1(adapted_model).to("cpu")

    N0 = W0 / (W0.norm(dim=1, keepdim=True) + 1e-12)
    N1 = W1 / (W1.norm(dim=1, keepdim=True) + 1e-12)

    dN = N1 - N0
    s = torch.linalg.svdvals(dN)

    if s.numel() == 0:
        return {"rank": 0, "stable_rank": 0.0}

    rank = int((s > tol).sum())
    stable_rank = float(s.pow(2).sum() / (s[0].pow(2) + 1e-12))
    return {"rank": rank, "stable_rank": stable_rank}
