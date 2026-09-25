from pathlib import Path
import argparse as arg
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

from tools.ascad_loader import load_dataset, dissemble_data_dict
from tools.model_zoo_pytorch import build_cnn_best
from tools.key_rank_new import ranking_curve


def parse_args():
    parser=arg.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_root", type=str, default="Output/cnn_pytorch")
    parser.add_argument("--n_traces", type=int, required=True)
    parser.add_argument("--target_byte", type=int, default=2)
    parser.add_argument("--leakage_model", type=str, choices=["HW","ID"], default="HW")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--trace_num_max", type=int, default=500)
    parser.add_argument("--num_averaged", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def predict_probabilities(model, traces, device, batch_size=512):
    model.eval()

    traces=torch.as_tensor(traces, dtype=torch.float32)

    if traces.ndim==2:
        traces=traces.unsqueeze(-1)

    dataset=TensorDataset(traces)
    loader=DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    probabilities=[]

    with torch.no_grad():
        for (batch,) in tqdm(loader, desc="CNN inference", dynamic_ncols=True):
            batch=batch.to(device)
            logits=model(batch)
            probs=torch.softmax(logits, dim=1)
            probabilities.append(probs.cpu())

    return torch.cat(probabilities, dim=0).numpy()


def main():
    args=parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        device=torch.device("cuda")
    elif torch.backends.mps.is_available():
        device=torch.device("mps")
    else:
        device=torch.device("cpu")

    print("Device:", device)

    # =========================================================================
    # Paths
    # =========================================================================

    run_root=Path(args.model_root)/f"N_{args.n_traces}"
    ckpt_path=run_root/"ckpt"/"cnn_best.pt"
    rank_root=run_root/"rank"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {ckpt_path}")

    # =========================================================================
    # Load attack data
    # =========================================================================

    data_dict=load_dataset(data_path=args.data_path, which_one="test")
    attack_traces, _, attack_plaintext, attack_key=dissemble_data_dict(data_dict=data_dict, tracewindow=(0,700), which_one="test")

    print("Attack traces:", attack_traces.shape)
    print("Attack plaintext:", attack_plaintext.shape)
    print("Real key byte:", int(attack_key[args.target_byte]))

    if args.trace_num_max > len(attack_traces):
        raise ValueError(f"trace_num_max={args.trace_num_max} exceeds attack dataset size={len(attack_traces)}")

    # =========================================================================
    # Build and load CNN
    # =========================================================================

    num_classes=9 if args.leakage_model=="HW" else 256

    model=build_cnn_best(input_shape=(attack_traces.shape[1],1), emb_size=num_classes, classification=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model=model.to(device)

    # =========================================================================
    # CNN probabilities
    # =========================================================================

    probabilities=predict_probabilities(model, attack_traces, device, args.batch_size)

    print("CNN probability shape:", probabilities.shape)

    assert probabilities.ndim==2
    assert probabilities.shape[0]==len(attack_traces)
    assert probabilities.shape[1]==num_classes

    # =========================================================================
    # Key rank
    # =========================================================================

    mean_rank=ranking_curve(probabilities, attack_key, attack_plaintext, args.target_byte, rank_root, args.leakage_model, args.trace_num_max, args.num_averaged)

    print("Mean rank shape:", mean_rank.shape)
    print("CNN test finished.")


if __name__=="__main__":
    main()