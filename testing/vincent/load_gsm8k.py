from datasets import load_dataset
import json, os

os.makedirs("data", exist_ok=True)
ds = load_dataset("gsm8k", "main")["train"]

with open("data/raw_problems.jsonl", "w") as f:
    for ex in ds:
        f.write(json.dumps({
            "prompt": ex["question"],
            "problem": ex["question"],
            "model_generation": [ex["answer"]],
            "all_eval": [True], 
            "level": "unknown",
            "answer": ex["answer"]
        }) + "\n")
