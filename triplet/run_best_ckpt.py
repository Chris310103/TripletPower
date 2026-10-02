import argparse
import random
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import accuracy_score

from tools.ascad_loader import load_dataset, dissemble_data_dict
from tools.loadData import get_labels
from tools.model_zoo_pytorch import build_cnn_best
from triplet.triplet_pytorch import getCLSidDict, train_knn, predict_knn_prob
from tools.key_rank_new import ranking_curve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/ASCAD.h5')
    parser.add_argument('--ckpt_path', type=str, default='Output/triplet_pytorch/profiling/true_semihard_2000_traces/ckpt/triplet_best.pt')
    parser.add_argument('--rank_name', type=str, default='true_semihard_2000_traces_BEST')
    parser.add_argument('--n_traces', type=int, default=2000)
    parser.add_argument('--target_byte', type=int, default=2)
    parser.add_argument('--leakage_model', type=str, default='HW')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # ================= 核心修复：对齐随机数种子 =================
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # ============================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading EXACT {args.n_traces} profiling traces (Seed {args.seed})...")
    x_n, labels_n, _, _, _, _ = getCLSidDict(
        data_path=args.data_path, 
        n_traces=args.n_traces,
        sample_num_limit=300, 
        leakage_model=args.leakage_model, 
        target_byte=args.target_byte
    )

    print(f"[INFO] Initializing model and loading the best checkpoint: {args.ckpt_path}")
    model = build_cnn_best(input_shape=(x_n.shape[1], 1), emb_size=256, classification=False)
    
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("[INFO] Training KNN classifier...")
    classifier = train_knn(model=model, traces=x_n, labels=labels_n, n_neighbors=10)

    print("[INFO] Loading attack test data...")
    data_dict = load_dataset(data_path=args.data_path, which_one="test")
    attack_traces, _, attack_plaintext, attack_real_key = dissemble_data_dict(
        data_dict=data_dict, tracewindow=(0, 700), which_one="test"
    )

    print("[INFO] Executing attack on full test set...")
    attack_probabilities = predict_knn_prob(model, classifier, attack_traces, leakage_model=args.leakage_model)
    attack_expected = get_labels(attack_plaintext, int(attack_real_key[args.target_byte]), args.target_byte, args.leakage_model)
    
    attack_pred = np.argmax(attack_probabilities, axis=1)
    attack_acc = accuracy_score(attack_expected, attack_pred)
    
    print("\n" + "="*50)
    print(f"[*] SEEDED Best Checkpoint Attack Accuracy: {attack_acc:.6f}")
    print("="*50 + "\n")

    rank_root = Path("Output/triplet_pytorch/profiling/rank") / args.rank_name
    rank_root.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Calculating and plotting Key Rank curve...")
    ranking_curve(
        preds=attack_probabilities, 
        key=attack_real_key, 
        plaintext=attack_plaintext, 
        target_byte=args.target_byte, 
        rank_root=rank_root, 
        leakage_model=args.leakage_model, 
        trace_num_max=5000, 
        num_averaged=5
    )

if __name__ == "__main__":
    main()