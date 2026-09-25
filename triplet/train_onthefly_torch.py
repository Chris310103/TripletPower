from pathlib import Path
import random
import numpy as np
import torch
import argparse as arg

from tools.ascad_loader import load_dataset, dissemble_data_dict
from tools.loadData import get_labels
from tools.model_zoo_pytorch import build_cnn_best
from triplet.triplet_pytorch import (
    getCLSidDict,
    build_positive_pairs,
    train_tripletpower,
    train_knn,
    predict_knn_prob,
    limit_per_class
)

def parse_args():
    parser=arg.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True, help="ascad data path")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--target_byte', type=int, default=2)
    parser.add_argument("--sample_num_limit", type=int, default=300)
    parser.add_argument('--leakage_model', type=str, choices=['HW', 'ID'], default="HW")
    parser.add_argument('--n_traces', type=int, default=2000)
    parser.add_argument('--alpha_value', type=float, default=0.5)
    parser.add_argument('--n_neighbors', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument( "--start_idx", type=int, default=0, )
    parser.add_argument( "--end_idx", type=int, default=2, )

    return parser.parse_args()
    

def get_params(args):
    data_path=args.data_path
    epochs=args.epochs
    batch_size=args.batch_size
    sample_num_limit=args.sample_num_limit
    target_byte=args.target_byte
    leakage_model=args.leakage_model
    n_traces=args.n_traces
    alpha_value=args.alpha_value
    n_neighbors=args.n_neighbors
    seed=args.seed
    lr=args.learning_rate
    start_idx=args.start_idx
    end_idx=args.end_idx

    return data_path, epochs, batch_size, target_byte, sample_num_limit,  leakage_model, n_traces, alpha_value, n_neighbors, seed, lr, start_idx, end_idx

def main():
    args=parse_args()
    data_path, epochs, batch_size, target_byte, sample_num_limit,\
        leakage_model, n_traces, alpha_value, n_neighbors, seed, lr, start_idx, end_idx=\
        get_params(args)

    output_root=Path("Output/triplet_pytorch/on-the-fly")
    output_root.mkdir(parents=True, exist_ok=True)

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

    # ============================================================
    # Load profiling data ONCE
    # ============================================================

    data_dict = load_dataset( data_path=args.data_path, which_one="train" )

    x, _, plain_text, key = dissemble_data_dict( data_dict=data_dict, tracewindow=(0, 700), which_one="train" )

    print("Full profiling traces:", x.shape)
    print("Full plaintext:", plain_text.shape)

    if args.n_traces > len(x):
        raise ValueError(f"n_traces={args.n_traces} > dataset size={len(x)}")

    selected_n_indices = np.random.choice(len(x), size=args.n_traces, replace=False)

    x_n = x[selected_n_indices]
    plain_text_n = plain_text[selected_n_indices]

    print("Selected N traces:", x_n.shape)
    print("Selected N plaintext:", plain_text_n.shape)

    # =========================================================================
    # Guess-key Loop
    # =========================================================================
    for guess_key in range( args.start_idx, args.end_idx ):
        print()
        print("=" * 70)
        print(f"Guess key: {guess_key}")
        print("=" * 70)

        labels_n = get_labels(plain_text_n, guess_key, args.target_byte, args.leakage_model)
        x_limited, labels_limited, label_2_id, id_2_label, selected_u_indices = limit_per_class(x_n, labels_n, args.sample_num_limit)
        a_ids, p_ids = build_positive_pairs(sorted(label_2_id.keys()), label_2_id)

        guess_dir = output_root / f"guess_key_{guess_key}"

        ckpt_path = ( guess_dir / "ckpt" / "triplet_best.pt")

        ckpt_path.parent.mkdir( parents=True, exist_ok=True)

        unique, counts = np.unique(
            labels_n,
            return_counts=True
        )

        distribution_n = dict(
            zip(unique.tolist(), counts.tolist())
        )

        unique_u, counts_u = np.unique(
            labels_limited,
            return_counts=True
        )

        distribution_u = dict(
            zip(unique_u.tolist(), counts_u.tolist())
        )

        print("labels_n shape:", labels_n.shape)
        print("N class distribution:", distribution_n)

        print("x_limited shape:", x_limited.shape)
        print("labels_limited shape:", labels_limited.shape)
        print("U class distribution:", distribution_u)

        print(
            "label_2_id keys:",
            sorted(label_2_id.keys())
        )

        print(
            "id_2_label size:",
            len(id_2_label)
        )

        print(
            "positive pairs:",
            len(a_ids)
        )

        assert len(x_n) == len(labels_n)

        assert len(x_limited) == len(labels_limited)

        assert all(
            count <= args.sample_num_limit
            for count in distribution_u.values()
        )

        assert len(a_ids) == len(p_ids)

        print("\nOn-the-fly data preparation smoke test finished.")

        # =========================================================================
        # Train feat model
        # =========================================================================
        model = build_cnn_best(input_shape=(x_limited.shape[1], 1), emb_size=256, classification=False)

        model = model.to(device)
        model, loss_log = train_tripletpower(
            model=model,
            all_traces=x_limited,
            a_ids=a_ids,
            p_ids=p_ids,
            id_2_label=id_2_label,
            device=device,
            ckpt_path=ckpt_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            alpha_value=alpha_value,
        )

        print( f"Finished Triplet training for guess_key={guess_key}" )

if __name__ == "__main__":
    main()        