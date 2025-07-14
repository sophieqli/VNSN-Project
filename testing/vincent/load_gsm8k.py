from datasets import load_dataset
import json, os

os.makedirs("data", exist_ok=True)
ds = load_dataset("gsm8k", "main")["train"]

with open("data/raw_problems.jsonl", "w") as f:
    for ex in ds:
        # ex is a dict with keys like "question", "answer", etc.
        # adapt to whatever your extract script expects 
        f.write(json.dumps({
            "prompt": ex["question"],
            "problem": ex["question"],
            "model_generation": [ex["answer"]],  # or however you wrap it
            "all_eval": [True],                  # dummy label
            "level": "unknown",
            "answer": ex["answer"]
        }) + "\n")
