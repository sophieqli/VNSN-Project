import argparse
from cot_prune.extraction import extract_and_save_hidden
from cot_prune.steering import build_steering_vectors
from cot_prune.inference import generate_with_pruning

def main_extract():
    p = argparse.ArgumentParser("extract-hidden")
    p.add_argument("--model",  required=True)
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split",  choices=["correct","incorrect"], default="correct")
    args = p.parse_args()
    extract_and_save_hidden(
        model_name=args.model,
        raw_path=args.input,
        out_dir=args.output,
        split=args.split,
    )

def main_build():
    p = argparse.ArgumentParser("build-steer")
    p.add_argument("--hidden_dir", required=True)
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--layers",     type=int, nargs="+", required=True)
    args = p.parse_args()
    build_steering_vectors(
        data_dir=args.hidden_dir,
        layers=args.layers,
        save_dir=args.out_dir,
    )

def main_prune():
    p = argparse.ArgumentParser("prune-gen")
    p.add_argument("--model",     required=True)
    p.add_argument("--steer_vec", required=True)
    p.add_argument("--prompt",    required=True)
    p.add_argument("--max_steps", type=int, required=True)
    p.add_argument("--tau_red",   type=float, required=True)
    p.add_argument("--lambda1",   type=float, required=True)
    p.add_argument("--lambda2",   type=float, required=True)
    args = p.parse_args()
    generate_with_pruning(
        model_name=args.model,
        steer_vec_path=args.steer_vec,
        prompt=args.prompt,
        max_steps=args.max_steps,
        tau_red=args.tau_red,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
    )

if __name__=="__main__":
    import sys
    name = sys.argv[0].split("/")[-1]
    if   name=="extract-hidden": main_extract()
    elif name=="build-steer":   main_build()
    elif name=="prune-gen":     main_prune()
    else:
        print("Unknown entry:", name)
        sys.exit(1)
