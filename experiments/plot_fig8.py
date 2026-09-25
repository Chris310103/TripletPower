from pathlib import Path
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser=arg.ArgumentParser()
    parser.add_argument("--n_values", type=int, nargs="+", required=True)
    parser.add_argument("--cnn_rank_paths", type=str, nargs="+", required=True)
    parser.add_argument("--triplet_rank_paths", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, default="Output/fig8/fig8.png")
    return parser.parse_args()


def load_rank(path):
    data=np.load(path)
    return np.asarray(data["y"], dtype=np.float64)


def main():
    args=parse_args()

    if not (len(args.n_values)==len(args.cnn_rank_paths)==len(args.triplet_rank_paths)):
        raise ValueError("n_values, cnn_rank_paths and triplet_rank_paths must have the same length")

    num_panels=len(args.n_values)

    fig, axes=plt.subplots(1, num_panels, figsize=(5*num_panels, 4), squeeze=False)
    axes=axes[0]

    for i, n_traces in enumerate(args.n_values):
        cnn_rank=load_rank(args.cnn_rank_paths[i])
        triplet_rank=load_rank(args.triplet_rank_paths[i])

        max_len=min(len(cnn_rank), len(triplet_rank))
        cnn_rank=cnn_rank[:max_len]
        triplet_rank=triplet_rank[:max_len]

        x=np.arange(1, max_len+1)

        ax=axes[i]

        ax.plot(x, cnn_rank, color="black", linewidth=1.2, marker="s", markevery=max(1,max_len//5), markerfacecolor="none", label="CNN")
        ax.plot(x, triplet_rank, linewidth=1.2, marker="o", markevery=max(1,max_len//5), markerfacecolor="none", label="TripletPower (Ours)")

        ax.set_xlabel("No. of test traces")
        ax.set_ylabel("Mean rank")
        ax.set_ylim(0,256)
        ax.set_yticks([0,64,128,192,256])
        ax.set_title(f"N = {n_traces}")
        ax.legend()

    output_path=Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved Fig.8-style plot to: {output_path}")


if __name__=="__main__":
    main()