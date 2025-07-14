import argparse
from cot_prune.inference import generate_with_pruning

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",        required=True)
    p.add_argument("--steer_vec",    required=True)  # data/vectors/layer_20_*.pt
    p.add_argument("--prompt",       required=True)
    p.add_argument("--max_steps",    type=int, default=50)
    p.add_argument("--tau_red",      type=float, default=0.92)
    p.add_argument("--lambda1",      type=float, default=1.0)
    p.add_argument("--lambda2",      type=float, default=1.0)
    args = p.parse_args()
    generate_with_pruning(
        model_name=args.model,
        steer_path=args.steer_vec,
        prompt=args.prompt,
        max_steps=args.max_steps,
        tau_red=args.tau_red,
        lambda1=args.lambda1,
        lambda2=args.lambda2
    )
