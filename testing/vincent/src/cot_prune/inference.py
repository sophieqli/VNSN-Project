import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import cosine_similarity
from cot_prune.redundancy import compute_redundancy
from cot_prune.logits import modify_logits_with_alpha
from cot_prune.drift_scorer import DriftScorer


def generate_with_pruning(model_name, steer_path, prompt, max_steps=50,
                          tau_red=0.9, lambda1=1.0, lambda2=1.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    candidates = [str(i) for i in range(10)] + ["A","B","C","D","E","answer","final"]
    answer_token_ids = []
    for tok in candidates:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) == 1:
            answer_token_ids.append(ids[0])
    
    model     = AutoModelForCausalLM.from_pretrained(
        model_name, output_hidden_states=True, return_dict=True
    ).to(device)
    steer_vec = torch.load(steer_path).to(device)

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    # cache, h_prev = [], None
    drift_scorer = DriftScorer(tau_red=tau_red, tau_stall=tau_red,lambda_red=lambda1, lambda_stall=lambda2)

    for _ in range(max_steps):
        outputs = model(input_ids)
        h_t = outputs.hidden_states[-1][0, -1, :]
        # cache.append(h_t)

        # R_t = compute_redundancy(h_t, cache[:-1], tau_red)
        # if h_prev is not None:
        #     delta = (h_t - h_prev).unsqueeze(0)
        #     D_t = cosine_similarity(delta, steer_vec.unsqueeze(0), dim=1).item()
        # else:
        #     D_t = 1.0

        # alpha_t = lambda1 * R_t + lambda2 * (1 - D_t)
        # logits = outputs.logits[0, -1, :]
        # logits = modify_logits_with_alpha(logits, alpha_t)
        logits = outputs.logits[0,-1,:]
        
        mod_logits, info = drift_scorer.intervene(logits,h_t,tokenizer,answer_token_ids)
        logits = mod_logits

        # next_id = logits.argmax().unsqueeze(0).unsqueeze(0)
        next_id = logits.argmax().view(1,1)
        input_ids = torch.cat([input_ids, next_id.to(device)], dim=1)
        # h_prev = h_t

    tokens = input_ids[0].tolist()
    print(tokenizer.decode(tokens))