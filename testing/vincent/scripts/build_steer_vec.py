import argparse
from cot_prune.steering import build_steering_vectors

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_dir", required=True)    # data/hidden_correct_0_500
    p.add_argument("--out_dir",    required=True)    # data/vectors
    p.add_argument("--layers",     type=int, nargs="+", default=[20])
    args = p.parse_args()
    build_steering_vectors(
        data_dir=args.hidden_dir,
        layers=args.layers,
        save_dir=args.out_dir
    )
