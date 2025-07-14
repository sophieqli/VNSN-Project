import torch
from torch.nn.functional import cosine_similarity

def compute_redundancy(h_t, cache, tau):
    if not cache:
        return 0.0
    sims = [cosine_similarity(h_t.unsqueeze(0), h_i.unsqueeze(0), dim=1).item()
            for h_i in cache]
    return max(sims)
