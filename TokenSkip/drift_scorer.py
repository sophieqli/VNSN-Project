
import torch
import torch.nn.functional as F
import numpy as np

class DriftScorer:
    def __init__(self, tau_red=0.9, tau_stall=0.9, momentum_beta=0.9, lambda_red=1.0, lambda_stall=1.0):
        self.cache = []  # list of prior hidden states [h_1, ..., h_{t-1}], each shape: [hidden_dim]
        self.mu = None   # mean vector of cache
        self.cov_inv = None  # inv cov matrix of cache
        self.momentum = None  # exp. drift dir (momentum-style)
        self.tau_red = tau_red
        self.tau_stall = tau_stall
        self.beta = momentum_beta
        self.lambda_red = lambda_red
        self.lambda_stall = lambda_stall

    def update_cache(self, h_t):
        self.cache.append(h_t.detach())


    #we have projected cosine sim above, but can try max for comparison
    def cosine_similarity(self, a, b):
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def max_similarity_to_cache(self, h_t):
        # max cosine similarity to any cached hidden state
        if len(self.cache) == 0: return 0.0
        sims = torch.stack([F.cosine_similarity(h_t.unsqueeze(0), h_i.unsqueeze(0)) for h_i in self.cache])
        return sims.max().item()

    def stall_score(self):
        # Compute if drift has plateaued:
        # Sim(h_{t-1}, h_t) approx Sim(h_{t-2}, h_{t-1}) > tau_stall
        if len(self.cache) < 2:
            return 0.0
        sim1 = self.cosine_similarity(self.cache[-1], self.cache[-2])
        # compare with previous step if available
        if len(self.cache) < 3:
            return 0.0
        sim2 = self.cosine_similarity(self.cache[-2], self.cache[-3])
        delta = abs(sim1 - sim2)
        # Stall if similarity high and difference between similarities is small (stable plateau)
        if sim1 > self.tau_stall and abs(sim1 - sim2) < 0.01:
            return sim1  # high stall score
        if sim1 > self.tau_stall * 0.7:  # some lower bound
          return sim1 * max(0, 1 - delta * 50)  # scale down as delta increases

        return 0.0
    def expected_direction(self, h_t):
        # momentum-style expected direction vector
        if len(self.cache) < 1:
            return None, 0.0
        delta = h_t - self.cache[-1]  # current change vector
        if self.momentum is None:
            self.momentum = delta
        else:
            self.momentum = self.beta * self.momentum + (1 - self.beta) * delta

        # cosine similarity between delta and expected drift
        cos_sim = F.cosine_similarity(delta.unsqueeze(0), self.momentum.unsqueeze(0)).item()
        return self.momentum, cos_sim

    def drift_penalty(self, h_t, forward_token_ids=None, lambda1=1.0, lambda2=1.0):
        redundancy_score = self.max_similarity_to_cache(h_t)
        stall_score = self.stall_score()
        #2 OPTIONS -> TRY DIFF SIMILARITY METRICS
        combined_redundancy = max(redundancy_score, stall_score)
        #self.update_cache(h_t)
        #combined_redundancy = pcs_score(self.cache)
        #self.cache.pop()

        # espected drift direction similarity
        _, drift_cos_sim = self.expected_direction(h_t)
        #tune lambdas accordingly
        penalty = lambda1 * combined_redundancy - lambda2 * drift_cos_sim
        # can also include mahal_dist if want to discourage outliers:
        #mahal_dist = self.mahalanobis_distance(h_t)
        #penalty += 0.1 * mahal_dist  # scale as needed

        return penalty, {
            "redundancy": combined_redundancy,
            "stall": stall_score,
            "mahalanobis": 0.0, #zeroed out
            "drift_cos_sim": drift_cos_sim,
        }

    def compute_intervention_weight(self, h_t):
        redundancy = self.max_similarity_to_cache(h_t)
        stall = self.stall_score()

        # comb with tunable lambdas
        alpha = self.lambda_red * redundancy + self.lambda_stall * stall
        return alpha, redundancy, stall

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

        # also boost digits and option letters ('A', 'B', 'C', 'D')
        digit_ids = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
        option_ids = [tokenizer.encode(ch, add_special_tokens=False)[0] for ch in ["A", "B", "C", "D", "E"]]
        # Combine conclusion + digits + options
        forward_ids = conclusion_ids + digit_ids + option_ids
        # Scale factors
        reflection_scale = 1.0 - 0.5 * alpha  # reduce logits softly for reflection tokens
        forward_scale = 1.0 + 0.5 * alpha     # boost logits softly for forward tokens

        mod_logits = logits.clone()
        # reductions
        for rid in reflection_ids: mod_logits[:, rid] *= reflection_scale
        # boosts
        for fid in forward_ids: mod_logits[:, fid] *= forward_scale

        return mod_logits


    def sealing_condition(self, logits, h_t, answer_token_ids, threshold=0.8):
        """
        determine if answer sealing should trigger:
        - top logits strongly favor answer tokens (numeric, option letter)
        - cld also use latent similarity checks or classifier outputs
        """
        probs = F.softmax(logits, dim=-1)
        top_prob, top_idx = probs.max(dim=-1)

        # Check if top token is an answer token with high confidence
        if top_idx.item() in answer_token_ids and top_prob.item() > threshold:
            return True
        return False

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
        # Compute redundancy/stall BEFORE updating cache
        alpha, redundancy, stall = self.compute_intervention_weight(h_t)
        self.update_cache(h_t)

        # Redirect intervention
        mod_logits = self.redirect_logits(logits, tokenizer, alpha)

        # Answer sealing intervention
        if self.sealing_condition(mod_logits, h_t, answer_token_ids):
            mod_logits = self.seal_logits(mod_logits, tokenizer)
            sealing_triggered = True
        else:
            sealing_triggered = False

        penalty = alpha

        info = {
            "redundancy": redundancy,
            "stall": stall,
            "penalty": penalty,
            "sealing_triggered": sealing_triggered,
        }
        return mod_logits, info
    '''
      import torch.nn.functional as F

    def compute_perplexity(logits, target_ids):
        """
        logits: [T, V]
        target_ids: [T] (token ids)
        Returns: average perplexity over tokens
        """
        log_probs = F.log_softmax(logits, dim=-1)  # [T, V]
        target_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)  # [T]
        avg_neg_log_prob = -target_log_probs.mean()
        perplexity = avg_neg_log_prob.exp()
        return perplexity.item()
    '''
