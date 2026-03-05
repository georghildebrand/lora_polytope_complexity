import torch

def get_W1(model):
    if hasattr(model, "fc1"):
        return model.fc1.weight.detach()
    if hasattr(model, "model"): # DeepMLP
        return model.model[0].weight.detach()
    return model.effective_W().detach()

def matrix_rank(M, tol=1e-6):
    M = M.to("cpu")
    if M.numel() == 0: return 0
    return int((torch.linalg.svdvals(M) > tol).sum())

def stable_rank(M):
    M = M.to("cpu")
    if M.numel() == 0: return 0
    s = torch.linalg.svdvals(M)
    return float(s.pow(2).sum() / (s[0].pow(2) + 1e-12))

@torch.no_grad()
def hyperplane_rotation_rank(base_m, adapt_m, E=None, tol=1e-6):
    W0 = get_W1(base_m);  W1 = get_W1(adapt_m)
    if E is not None:
        W0, W1 = W0 @ E, W1 @ E
    n0 = W0 / (W0.norm(dim=1, keepdim=True) + 1e-12)
    n1 = W1 / (W1.norm(dim=1, keepdim=True) + 1e-12)
    dN = (n1 - n0).to("cpu")
    s  = torch.linalg.svdvals(dN)
    if s.numel() == 0: return 0, 0.0
    return int((s > tol).sum()), float(s.pow(2).sum() / (s.max().pow(2) + 1e-12))
