# CoT Pruning Pipeline

A modular repo for inference‑time chain‑of‑thought pruning using hidden‑state signals and steering vectors (based on SEAL). This README guides you through setup, data preparation, and running the three main stages: hidden‑state extraction, steering‑vector generation, and pruned inference.

---

## 📋 Repository Structure

```
my-cot-prune/
├── .gitignore
├── requirements.txt        # Python dependencies
├── load_gsm8k.py      # Download & dump GSM8K to JSONL
├── scripts/
│   ├── extract_hidden.py  # Stage 1: extract hidden states
│   ├── build_steer_vec.py # Stage 2: compute steering vectors
│   └── run_inference.py   # Stage 3: pruned CoT generation
├── data/                   # Local data (ignored by Git)
│   ├── raw_problems.jsonl  # GSM8K dump (generated locally)
│   ├── hidden/             # Hidden‑state outputs
│   └── vectors/            # Steering vectors
└── src/
    └── cot_prune/          # Core Python modules
        ├── extraction.py
        ├── steering.py
        ├── redundancy.py
        ├── logits.py
        └── inference.py
```

---

## ⚙️ Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/my-cot-prune.git
   cd my-cot-prune
   ```
2. **Create and activate a Python 3 env**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**

   ```bash
   pip install --upgrade pip setuptools
   pip install -r requirements.txt
   ```

---

## 🗂 Data Preparation

By default, raw GSM8K data is **not** tracked in Git. You can generate it locally:

```bash
python3 load_gsm8k.py
```

This writes `data/raw_problems.jsonl` (the GSM8K training split) into `data/`.

> **Note**: Make sure `data/raw_problems.jsonl` is in `.gitignore`.

---

## 🚀 Pipeline Stages

### 1. Extract Hidden States

Dump per‑step hidden‑state tensors from your chosen model.

```bash
python3 scripts/extract_hidden.py \
  --model  EleutherAI/gpt-neo-2.7B \
  --input  data/raw_problems.jsonl \
  --output data/hidden \
  --split  correct
```

* **`--split`**: `correct` or `incorrect` examples.
* Outputs: `data/hidden/hidden_correct/hidden.pt` and `prompts.json`.

### 2. Build Steering Vectors

Compute latent steering directions per layer.

```bash
python3 scripts/build_steer_vec.py \
  --hidden_dir data/hidden/hidden_correct \
  --out_dir    data/vectors \
  --layers     20
```

* **`--layers`**: list of layer indices to process.
* Outputs: `data/vectors/layer_20_steer_vec.pt`.

### 3. Run Pruned Inference

Generate a pruned chain‑of‑thought with live redundancy & drift interventions.

```bash
python3 scripts/run_inference.py \
  --model      EleutherAI/gpt-neo-2.7B \
  --steer_vec  data/vectors/layer_20_steer_vec.pt \
  --prompt     "Q: ... A: Let's think step by step." \
  --max_steps  50 \
  --tau_red    0.9 \
  --lambda1    1.0 \
  --lambda2    1.0
```

* **`--tau_red`**: cosine‑similarity threshold for redundancy.
* **`--lambda1, --lambda2`**: weights for redundancy and drift signals.

The script prints the final pruned CoT trace to stdout.

---

## 🔧 Configuration

* **`.gitignore`** ensures that `data/*.jsonl` and large dumps aren’t checked in.
* **`requirements.txt`** pins PyTorch, Transformers, Datasets, and TQDM.
* Feel free to adjust thresholds, model names, or layers via script flags.

