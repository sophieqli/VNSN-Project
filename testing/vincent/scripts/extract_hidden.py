import argparse
from cot_prune.extraction import extract_and_save_hidden

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",    required=True)
    p.add_argument("--input",    required=True)  # data/raw_problems.jsonl
    p.add_argument("--output",   required=True)  # data/hidden
    p.add_argument("--split",    choices=["correct","incorrect"], default="correct")
    args = p.parse_args()
    extract_and_save_hidden(
        model_name=args.model,
        raw_path=args.input,
        out_dir=args.output,
        split=args.split
    )
