# cot_prune/drift_scorer.py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

def pcs_score(hidden_states, k=5, pca_dim=128, use_pca=True):
    h_np = hidden_states.cpu().numpy()
    n, d = h_np.shape
    if use_pca:
        pca_dim = min(pca_dim, n, d)
        pca = PCA(n_components=pca_dim)
        h_proj = pca.fit_transform(h_np)
    else:
        h_proj = h_np
    #     h_np = PCA(n_components=pca_dim).fit_transform(h_np)
    # h_t   = torch.tensor(h_np[-1])
    # sims  = F.cosine_similarity(h_t.unsqueeze(0), torch.tensor(h_np[:-1]), dim=1)
    # topk, _ = torch.topk(sims, k)
    # weights = torch.softmax(topk/0.07, dim=0)
    # return (weights * topk).sum().item()
    h_t = torch.tensor(h_proj[-1])
    prev = torch.tensor(h_proj[:-1])
    
    if prev.size(0) == 0:
        return 0.0
    
    sims = F.cosine_similarity(h_t.unsqueeze(0), prev, dim=1)
    
    k_eff = min(k, sims.size(0))
    topk, _ = torch.topk(sims, k_eff)
    
    weights = torch.softmax(topk / 0.07, dim=0)
    score = torch.sum(weights * topk)
    
    return score.item()
    
    

class DriftScorer:
    def __init__(self, tau_red=0.9, tau_stall=0.9,
                 momentum_beta=0.9, lambda_red=1.0, lambda_stall=1.0):
        self.cache        = []
        self.mu           = None
        self.cov_inv      = None
        self.momentum     = None
        self.tau_stall    = tau_stall
        self.beta         = momentum_beta
        self.lambda_red   = lambda_red
        self.lambda_stall = lambda_stall

    def update_cache(self, h_t):
        self.cache.append(h_t.detach())
        if len(self.cache)>2:
            H    = torch.stack(self.cache)
            cov  = torch.from_numpy(np.cov(H.cpu().numpy(), rowvar=False)
                                     +1e-5*np.eye(H.size(1)))
            self.cov_inv = torch.linalg.inv(cov).to(H.device)
            self.mu      = H.mean(dim=0)

    def max_similarity_to_cache(self, h_t):
        if not self.cache: return 0.0
        sims = [F.cosine_similarity(h_t.unsqueeze(0), h_i.unsqueeze(0),dim=1)
                for h_i in self.cache]
        return torch.stack(sims).max().item()

    def stall_score(self):
        if len(self.cache)<3: return 0.0
        sim1 = F.cosine_similarity(
                   self.cache[-1].unsqueeze(0),
                   self.cache[-2].unsqueeze(0),dim=1).item()
        sim2 = F.cosine_similarity(
                   self.cache[-2].unsqueeze(0),
                   self.cache[-3].unsqueeze(0),dim=1).item()
        if sim1>self.tau_stall and abs(sim1-sim2)<0.01:
            return sim1
        if sim1>0.7*self.tau_stall:
            return sim1*max(0,1-abs(sim1-sim2)*50)
        return 0.0

    def expected_direction(self, h_t):
        delta = h_t - self.cache[-1]
        self.momentum = delta if self.momentum is None \
                        else self.beta*self.momentum + (1-self.beta)*delta
        cos_sim = F.cosine_similarity(
                      delta.unsqueeze(0), self.momentum.unsqueeze(0),dim=1
                  ).item()
        return self.momentum, cos_sim

    def compute_intervention_weight(self, h_t):
        red   = self.max_similarity_to_cache(h_t)
        stall = self.stall_score()
        α     = self.lambda_red*red + self.lambda_stall*stall
        return α, red, stall

    def redirect_logits(self, logits, tokenizer, alpha):
        """
        softly reduce logits of stall tokens (transitions, fillers)
        boost logits of forward tokens (conclusion, digits, answer markers).
        """
        #(batch_size, vocab_size)

        #examples, expand later
        reflection_tokens = ["alternatively", "so maybe", "however", "well", "perhaps", "on the other hand"]
        conclusion_tokens = ["thus", "therefore", "hence", "answer", "final", "result", "=", ".", ","]

        reflection_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in reflection_tokens if len(tokenizer.encode(t, add_special_tokens=False))==1]
        conclusion_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in conclusion_tokens if len(tokenizer.encode(t, add_special_tokens=False))==1]

        # Also boost digits and option letters (e.g., 'A', 'B', 'C', 'D')
        digit_ids = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
        option_ids = [tokenizer.encode(ch, add_special_tokens=False)[0] for ch in ["A", "B", "C", "D", "E"]]
        print("digit ids ", digit_ids)
        # Combine conclusion + digits + options
        forward_ids = conclusion_ids + digit_ids + option_ids
        print("forward ids:: ", forward_ids)
        # Scale factors
        reflection_scale = 1.0 - 0.5 * alpha  # reduce logits softly for reflection tokens
        forward_scale = 1.0 + 0.5 * alpha     # boost logits softly for forward tokens

        mod_logits = logits.clone()
        # Apply reductions
        for rid in reflection_ids:
            mod_logits[:, rid] *= reflection_scale
        # Apply boosts
        for fid in forward_ids:
            mod_logits[:, fid] *= forward_scale

        return mod_logits

    def sealing_condition(self, logits, h_t, answer_token_ids, threshold=0.8):
        probs, idxs = F.softmax(logits,dim=-1).max(dim=-1)
        return idxs.item() in answer_token_ids and probs.item()>threshold

    def seal_logits(self, logits, tokenizer):
        #triggered only when sealing condition met, i.e., model confidently favors answer tokens
        #A hard intervention, prevents wasting tokens
        seal_tokens = ["thus", "the answer is", "final answer", "=", ".", ","]

        seal_ids = []
        for t in seal_tokens:
            enc = tokenizer.encode(t, add_special_tokens=False)
            if len(enc) == 1:
                seal_ids.append(enc[0])
        seal_boost = 3.0  # strong boost

        mod_logits = logits.clone()
        for sid in seal_ids:
            mod_logits[:, sid] *= seal_boost

        return mod_logits

    def intervene(self, logits, h_t, tokenizer, answer_token_ids):
        α, red, stall = self.compute_intervention_weight(h_t)
        self.update_cache(h_t)
        mod_logits = self.redirect_logits(logits.unsqueeze(0), tokenizer, α)
        sealing = False
        if self.sealing_condition(mod_logits, h_t, answer_token_ids):
            mod_logits = self.seal_logits(mod_logits, tokenizer)
            sealing = True
        return mod_logits.squeeze(0), {
            "penalty": α, "redundancy": red,
            "stall": stall, "sealing_triggered": sealing
        }
