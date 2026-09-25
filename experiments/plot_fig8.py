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

def plot_panel(ax, cnn_rank, triplet_rank, n_traces):
    max_len=min(len(cnn_rank), len(triplet_rank))
    cnn_rank=cnn_rank[:max_len]
    triplet_rank=triplet_rank[:max_len]
    x=np.arange(1, max_len+1)

    marker_idx=np.linspace(0, max_len-1, 5, dtype=int)

    ax.plot(x, cnn_rank, color="black", linewidth=0.8, marker="s", markevery=marker_idx, markersize=5.5, markerfacecolor="none", markeredgewidth=0.8, label="CNN")
    ax.plot(x, triplet_rank, color="#b8ad00", linewidth=0.8, marker="o", markevery=marker_idx, markersize=5.5, markerfacecolor="none", markeredgewidth=0.8, label="TripletPower (Ours)")

    ax.set_xlabel("No. of test traces",fontsize=8)
    ax.set_ylabel("Mean rank",fontsize=8)



    ax.set_ylim(0,256)
    ax.set_yticks([0,64,128,192,256])

    ax.set_xlim(0,max_len)
    ax.tick_params(axis="both",labelsize=7,width=0.7,length=3)

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    ax.legend(fontsize=7,frameon=True,loc="best",handlelength=2.0,borderpad=0.35,labelspacing=0.3)
    ax.text(0.5,-0.24,f"No. of training traces $N={n_traces:,}$",transform=ax.transAxes,ha="center",va="top",fontsize=8)    


def main():
    args=parse_args()

    if not (len(args.n_values)==len(args.cnn_rank_paths)==len(args.triplet_rank_paths)):
        raise ValueError("n_values, cnn_rank_paths and triplet_rank_paths must have the same length")

    num_panels=len(args.n_values)
    ig,axes=plt.subplots(1,num_panels,figsize=(3.8*num_panels,2.7),squeeze=False)
    axes=axes[0]

    for i,n_traces in enumerate(args.n_values):
        cnn_rank=load_rank(args.cnn_rank_paths[i])
        triplet_rank=load_rank(args.triplet_rank_paths[i])
        plot_panel(axes[i],cnn_rank,triplet_rank,n_traces)

    output_path=Path(args.output)
    output_path.parent.mkdir(parents=True,exist_ok=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25,wspace=0.42)
    plt.savefig(output_path,dpi=300,bbox_inches="tight")
    plt.close()

    print(f"Saved Fig.8-style plot to: {output_path}")


if __name__=="__main__":
    main()