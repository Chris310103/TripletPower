from pathlib import Path
import argparse as arg
import numpy as np
import torch
import random
from tqdm.auto import tqdm

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

    parser.add_argument("--target_byte", type=int, default=2)

    parser.add_argument("--leakage_model", type=str, default="HW")

    parser.add_argument("--start_idx", type=int, default=0)

    parser.add_argument("--end_idx", type=int, default=1)

    return parser.parse_args()

def main():
    opts=parse_args()
    seed=42
    guess_key = opts.start_idx

    attack_data=load_dataset(data_path=opts.data_path, which_one="test")
    attack_traces, _, attack_plaintext, attack_real_key = \
    dissemble_data_dict(attack_data, tracewindow=(0,700), which_one="test")

    model_root=Path(opts.model_root)
    acc_list = []

    for guess_key in tqdm( range(opts.start_idx, opts.end_idx), desc="Testing guessed keys", dynamic_ncols=True):
        guess_dir =  model_root / "guess_key_0"

        ckpt_path = guess_dir / "ckpt" / "triplet_best.pt"
        knn_path = guess_dir / "knn" / "knn_model.joblib"

        # =========================================================================
        # Reproducibility
        # =========================================================================

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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
        
        # =========================================================================
        # Load feat model
        # =========================================================================    
        model = build_cnn_best(input_shape=(700, 1), emb_size=256, classification=False)

        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        model.to(device)
        model.eval()

        attack_embeddings = extract_embeddings(attack_traces, model)

        # =========================================================================
        # Load knn
        # =========================================================================
        classifier = load(knn_path)

        pred_y = classifier.predict(attack_embeddings)

        expected_y = get_labels(attack_plaintext, guess_key, opts.target_byte, opts.leakage_model)

        acc = accuracy_score(expected_y, pred_y)

        print(f"guess_key={guess_key}, " f"accuracy={acc:.6f}")

        # ============================================================
        # Labels under current guessed key
        # ============================================================

        expected_y = get_labels(attack_plaintext, guess_key, opts.target_byte, opts.leakage_model)

        acc = accuracy_score(expected_y, pred_y)

        acc_list.append(acc)

        tqdm.write(f"guess_key={guess_key}, accuracy={acc:.6f}")

        acc_list = np.asarray(acc_list, dtype=np.float64)

        result_dir = model_root / "results"
        result_dir.mkdir(parents=True, exist_ok=True)

        result_path = (result_dir / f"accuracy_{opts.start_idx}_{opts.end_idx}.npy")

        np.save(result_path, acc_list)

        print(f"Saved accuracy results to: {result_path}")

if __name__ == "__main__":
    main()