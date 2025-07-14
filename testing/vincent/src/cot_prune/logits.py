import torch

def modify_logits_with_alpha(logits, alpha, reflection_ids=None, forward_ids=None):
    # example: down-weight reflection tokens
    if reflection_ids is not None:
        logits[reflection_ids] -= alpha
    if forward_ids is not None:
        logits[forward_ids] += alpha
    return logits