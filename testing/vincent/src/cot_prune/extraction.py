import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def extract_and_save_hidden(model_name, raw_path, out_dir, split="correct"):
    # load raw JSONL
    data = [json.loads(l) for l in open(raw_path)]
    # split based on a field e.g. 'all_eval'
    correct, incorrect = [], []
    # assume each line has 'model_generation' and 'all_eval'
    # flatten into examples
    for d in data:
        for resp, ok in zip(d['model_generation'], d['all_eval']):
            target = correct if ok else incorrect
            target.append({
                'prompt': d['prompt'],
                'response': resp
            })
    examples = correct if split=='correct' else incorrect
    save_dir = os.path.join(out_dir, f"hidden_{split}")
    os.makedirs(save_dir, exist_ok=True)
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(
        model_name, output_hidden_states=True, return_dict=True
    )
    model.eval()
    hidden_dict = []
    for _ in range(model.config.num_hidden_layers+1):
        hidden_dict.append({})
    # generate and extract
    for idx, ex in enumerate(tqdm(examples)):
        text = ex['prompt'] + ex['response']
        tokens = tokenizer(text, return_tensors='pt', padding=True)
        input_ids = tokens.input_ids
        with torch.no_grad():
            out = model(**tokens)
        hiddens = [h.detach().cpu() for h in out.hidden_states]
        # assume simple step split at newlines
        steps = (text.split("\n"))
        # for each layer, collect first token of each step
        for layer, h in enumerate(hiddens):
            # map step positions to token indices (stub)
            step_idxs = list(range(len(steps)))
            layer_states = h[0, step_idxs, :]
            hidden_dict[layer][idx] = layer_states
    torch.save(hidden_dict, os.path.join(save_dir, 'hidden.pt'))
    print(f"Saved hidden states to {save_dir}/hidden.pt")