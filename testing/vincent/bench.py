import time
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from cot_prune.inference import generate_with_pruning

model_name = "gpt2-medium"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token     = tokenizer.eos_token
tokenizer.pad_token_id  = tokenizer.eos_token_id
tokenizer.padding_side  = "left"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE).eval()

# load a handful (or all) of GSM8K test questions
ds = load_dataset("gsm8k", "main", split="test[:100]")

def run_baseline(prompt, max_new_tokens=512):
    # 1) tokenize *with* padding so we get an attention_mask
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # sync if on GPU
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    # pass the mask and pad_token_id explicitly
    out_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    text   = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    length = out_ids.shape[1]
    return text, length, t1 - t0


def run_pruned(prompt, **kw):
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    text = generate_with_pruning(
        model_name=model_name,
        steer_vec_path=kw["steer_vec"],
        prompt=prompt,
        max_steps=kw["max_steps"],
        tau_red=kw["tau_red"],
        lambda1=kw["lambda1"],
        lambda2=kw["lambda2"],
    )
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t1 = time.time()
    length = tokenizer(text, return_tensors="pt").input_ids.shape[1]
    return text, length, t1-t0

# configure your prune params
prune_args = {
    "steer_vec": "data/vectors/layer_20_steer_vec.pt",
    "max_steps": 100,
    "tau_red":   0.9,
    "lambda1":   1.0,
    "lambda2":   1.0,
}

records = []
for ex in ds:
    prompt = ex["question"] + "\n\nLet's think step by step."
    gold   = ex["answer"].strip()

    # baseline
    base_txt, base_len, base_time = run_baseline(prompt)
    base_pred = base_txt.split("\n")[-1].strip()
    base_acc  = int(base_pred == gold)

    # pruned
    pr_txt, pr_len, pr_time = run_pruned(prompt, **prune_args)
    pr_pred = pr_txt.split("\n")[-1].strip()
    pr_acc  = int(pr_pred == gold)

    records.append({
        "baseline_len": base_len,
        "pruned_len":   pr_len,
        "baseline_time": base_time,
        "pruned_time":   pr_time,
        "baseline_acc":  base_acc,
        "pruned_acc":    pr_acc,
    })

df = pd.DataFrame(records)
print(df.mean().rename("avg"))
print("Token Savings:  {:.1%}".format(1 - df.pruned_len.mean()/df.baseline_len.mean()))
print("Speed‑Up:        {:.2f}×".format(df.baseline_time.mean()/df.pruned_time.mean()))
print("ΔAccuracy:      {:.1%}".format(df.pruned_acc.mean() - df.baseline_acc.mean()))
