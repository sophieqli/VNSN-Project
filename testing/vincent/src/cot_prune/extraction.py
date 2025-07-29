import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from cot_prune.drift_scorer import pcs_score

def extract_and_save_hidden(model_name, raw_path, out_dir, split="correct"):
    data = [json.loads(l) for l in open(raw_path)]
    correct, incorrect = [], []
    for d in data:
        for resp, ok in zip(d['model_generation'], d['all_eval']):
            target = correct if ok else incorrect
            target.append({'prompt': d['prompt'], 'response': resp})
    examples = correct if split=='correct' else incorrect

    save_dir = os.path.join(out_dir, f"hidden_{split}")
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_hidden_states=True, return_dict=True
    )
    model.eval()

    hidden_dict = [{} for _ in range(model.config.num_hidden_layers+1)]

    for idx, ex in enumerate(tqdm(examples)):
        text   = ex['prompt'] + ex['response']
        tokens = tokenizer(text, return_tensors='pt', padding=True)
        with torch.no_grad():
            out = model(**tokens)
        hiddens = [h.detach().cpu() for h in out.hidden_states]

        steps = text.split("\n")

        for layer, h in enumerate(hiddens):
            step_idxs    = list(range(len(steps)))
            layer_states = h[0, step_idxs, :]  

            score = pcs_score(
                hidden_states=layer_states,
                k=5,
                pca_dim=128,
                use_pca=True
            )

            hidden_dict[layer][idx] = {
                "states":    layer_states,
                "pcs_score": score
            }

    torch.save(hidden_dict, os.path.join(save_dir, 'hidden.pt'))
    print(f"Saved hidden states to {save_dir}/hidden.pt")
