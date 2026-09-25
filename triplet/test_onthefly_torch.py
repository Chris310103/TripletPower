from pathlib import Path
import argparse as arg
import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from joblib import load
from sklearn.metrics import accuracy_score

from tools.ascad_loader import load_dataset, dissemble_data_dict
from tools.model_zoo_pytorch import build_cnn_best
from tools.loadData import get_labels

from triplet.triplet_pytorch import extract_embeddings

def parse_args():
    parser = arg.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)

    parser.add_argument("--model_root", type=str, default="Output/triplet_pytorch/on-the-fly")

    parser.add_argument("--figure_name", type=str, default=None, help="output figure filename, e.g. fig10_smoke.png")

    parser.add_argument("--target_byte", type=int, default=2)

    parser.add_argument("--leakage_model", type=str, default="HW")

    parser.add_argument("--test_trace_nums", type=int, nargs="+", default=[25, 50, 100, 200])

    parser.add_argument("--start_idx", type=int, default=0)

    parser.add_argument("--end_idx", type=int, default=1)

    return parser.parse_args()

def plot_paper_style_accuracy(guess_keys, acc_matrix, test_trace_nums, real_key, save_path):
    plt.figure(figsize=(6, 4))

    x_pos = np.arange(len(test_trace_nums))

    incorrect_label_added = False
    correct_key_plotted = False

    for i, guess_key in enumerate(guess_keys):

        if guess_key == real_key:
            plt.plot(x_pos, acc_matrix[i], color="red", marker="o", \
                    markerfacecolor="none", markersize=9, linewidth=1.0, zorder=3, label=f"Correct Key ({hex(real_key)})")
            correct_key_plotted = True

        else:
            plt.plot(x_pos, acc_matrix[i], color="gray", marker=".", markersize=2, \
                    linewidth=0.7, alpha=0.7, zorder=1, label="Incorrect Keys" if not incorrect_label_added else None)

            incorrect_label_added = True

    plt.xticks(x_pos, test_trace_nums)

    plt.xlabel("No. of test traces")
    plt.ylabel("Accuracy")

    plt.ylim(0.0, 1.0)

    if incorrect_label_added or correct_key_plotted:
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    opts=parse_args()

    attack_data=load_dataset(data_path=opts.data_path, which_one="test")
    attack_traces, _, attack_plaintext, attack_real_key = \
    dissemble_data_dict(attack_data, tracewindow=(0,700), which_one="test")

    max_test_traces = max(opts.test_trace_nums)

    if max_test_traces > len(attack_traces):
        raise ValueError(
            f"max test trace number {max_test_traces} exceeds "
            f"available attack traces {len(attack_traces)}"
        )

    attack_traces_eval = attack_traces[:max_test_traces]
    attack_plaintext_eval = attack_plaintext[:max_test_traces]

    model_root=Path(opts.model_root)
    acc_matrix = []
    guess_keys = []
    real_key = int(attack_real_key[opts.target_byte])
    print(f"Real key byte: {real_key}")

    # =========================================================================
    # Device
    # =========================================================================
    if torch.cuda.is_available():
        device=torch.device("cuda")

    elif torch.backends.mps.is_available():
        device=torch.device("mps")

    else:
        device = torch.device("cpu")

    print("Device:", device)

    for guess_key in tqdm( range(opts.start_idx, opts.end_idx), desc="Testing guessed keys", dynamic_ncols=True):
        guess_dir =  model_root / f"guess_key_{guess_key}"

        ckpt_path = guess_dir / "ckpt" / "triplet_best.pt"
        knn_path = guess_dir / "knn" / "knn_model.joblib"
        
        # =========================================================================
        # Load feat model
        # =========================================================================    
        model = build_cnn_best(input_shape=(700, 1), emb_size=256, classification=False)

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}"
            )

        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        model.to(device)
        model.eval()

        attack_embeddings = extract_embeddings(attack_traces_eval, model)

        # =========================================================================
        # Load knn
        # =========================================================================
        if not knn_path.exists():
            raise FileNotFoundError(
                f"kNN model not found: {knn_path}"
            )
        
        classifier = load(knn_path)

        pred_y = classifier.predict(attack_embeddings)
 
        # ============================================================
        # Labels under current guessed key
        # ============================================================

        expected_y = get_labels(attack_plaintext_eval, guess_key, opts.target_byte, opts.leakage_model)
        acc_per_trace_num = []
        
        for trace_num in opts.test_trace_nums:

            if trace_num > len(pred_y):
                raise ValueError(f"test_trace_num={trace_num} exceeds " f"available attack traces={len(pred_y)}")

            acc = accuracy_score( expected_y[:trace_num], pred_y[:trace_num] )
            acc_per_trace_num.append(acc)

        guess_keys.append(guess_key)
        acc_matrix.append(acc_per_trace_num)

        tqdm.write(f"guess_key={guess_key}, " f"accuracies={acc_per_trace_num}")

    acc_matrix = np.asarray(acc_matrix, dtype=np.float64)

    guess_keys = np.asarray(guess_keys, dtype=np.int64)

    test_trace_nums = np.asarray(opts.test_trace_nums, dtype=np.int64)

    result_dir = model_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = (result_dir / f"paper_accuracy_{opts.start_idx}_{opts.end_idx}.npz")
    np.savez(result_path, guess_keys=guess_keys, test_trace_nums=test_trace_nums, accuracies=acc_matrix, real_key=real_key)

    print(f"Saved accuracy results to: {result_path}")

    # =========================================================================
    # Draw Plot
    # =========================================================================
    if opts.figure_name is None:
        figure_name = f"paper_accuracy_{opts.start_idx}_{opts.end_idx}.png"
    else:
        figure_name = opts.figure_name

    figure_path = result_dir / figure_name

    plot_paper_style_accuracy(guess_keys=guess_keys, acc_matrix=acc_matrix, test_trace_nums=test_trace_nums, real_key=real_key, save_path=figure_path)

    print(f"Saved paper-style figure to: {figure_path}")  

if __name__ == "__main__":
    main()