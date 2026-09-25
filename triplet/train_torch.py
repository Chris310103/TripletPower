from pathlib import Path
import random
import numpy as np
import torch
import argparse as arg

from tools.ascad_loader import load_dataset, dissemble_data_dict
from tools.model_zoo_pytorch import build_cnn_best
from triplet.triplet_pytorch import (
    getCLSidDict,
    build_positive_pairs,
    train_tripletpower,
    train_knn,
    predict_knn_prob
)
from tools.key_rank_new import ranking_curve

def parse_args():
    parser=arg.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True, help="ascad data path")
    parser.add_argument('--rank_name', type=str, required=True, help='output rank folder name')
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

    return parser.parse_args()
    

def get_params(args):
    data_path=args.data_path
    rank_name=args.rank_name
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

    return data_path, rank_name, epochs, batch_size, target_byte, sample_num_limit,  leakage_model, n_traces, alpha_value, n_neighbors, seed, lr

def main():
    args=parse_args()
    data_path, rank_name, epochs, batch_size, target_byte, sample_num_limit, leakage_model, n_traces, alpha_value, n_neighbors, seed, lr=\
        get_params(args)

    output_root=Path("Output/triplet_pytorch/profiling")
    ckpt_path=Path(output_root/rank_name/"ckpt"/"triplet_best.pt")
    rank_root=output_root/"rank"/ rank_name

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    rank_root.parent.mkdir(parents=True, exist_ok=True)

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
    # Prepare profiling data
    # ============================================================

    (x_n, labels_n, x_limited, labels_limited, label_2_id, id_2_label,) = \
        getCLSidDict(data_path=data_path, n_traces=n_traces, sample_num_limit=sample_num_limit, 
                    leakage_model=leakage_model,target_byte=target_byte,)

    print("All N profiling traces:", x_n.shape)
    print("Triplet subset:", x_limited.shape)

    # =========================================================================
    # Build positive pairs
    # =========================================================================
    a_ids, p_ids=build_positive_pairs(sorted(label_2_id.keys()), label_2_id)
    
    print("Number of positive pairs:", len(a_ids))

    # =========================================================================
    # Build Tripletpower embedding network
    # =========================================================================
    model=build_cnn_best(input_shape=(x_limited.shape[1], 1), emb_size=256, classification=False)

    model=model.to(device)

    # =========================================================================
    # Train Triplet network
    # =========================================================================
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

    # =========================================================================
    # Train k-nn on all n profiling traces
    # =========================================================================
    classifier = train_knn(
        model=model,
        traces=x_n,
        labels=labels_n,
        n_neighbors=n_neighbors,
    )

    print("KNN classes:", classifier.classes_)

    # =========================================================================
    # Load Attack data
    # =========================================================================
    data_dict=load_dataset(data_path=data_path, which_one="test")
    attack_traces, attack_label, attack_plaintext, attack_real_key=dissemble_data_dict(data_dict=data_dict, tracewindow=(0, 700), which_one="test")

    # =========================================================================
    # Attack
    # ========================================================================= 
    attack_probabilities = predict_knn_prob(
        model,
        classifier,
        attack_traces,
        leakage_model=leakage_model
    )

    # =========================================================================
    # Key-rank
    # =========================================================================
    ranking_curve(
        preds=attack_probabilities,
        key=attack_real_key,
        plaintext=attack_plaintext,
        target_byte=target_byte,
        rank_root=rank_root,
        leakage_model=leakage_model,
    )

    print("TripletPower pipeline finished.")

if __name__=="__main__":
    main()

