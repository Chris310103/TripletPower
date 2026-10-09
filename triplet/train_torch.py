from pathlib import Path
import random
import numpy as np
import torch
import argparse as arg
from sklearn.metrics import accuracy_score

from tools.ascad_loader import load_dataset, dissemble_data_dict
from tools.loadData import get_labels   
from tools.model_zoo_pytorch import build_cnn_best
from triplet.triplet_pytorch import (
    getCLSidDict,
    build_positive_pairs,
    train_tripletpower,
    train_knn,
    predict_knn_prob,
    extract_embeddings
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
    parser.add_argument("--selected_indices_path", type=str, default=None)
    parser.add_argument("--trace_num_max", type=int, default=500)
    parser.add_argument("--tracewindow", type=int, nargs=2, default=(0,700))
    parser.add_argument("--num_averaged", type=int, default=100)
    parser.add_argument('--leakage_model', type=str, choices=['HW', 'ID'], default="HW")
    parser.add_argument('--n_traces', type=int, default=2000)
    parser.add_argument("--attack_size", type=int, default=10000)
    parser.add_argument('--alpha_value', type=float, default=0.5)
    parser.add_argument("--alpha_mine", type=float, default=None)
    parser.add_argument("--loss_reduction", type=str, choices=["mean", "mean_nonzero"], default="mean")
    parser.add_argument('--n_neighbors', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument("--negative_mode", type=str, 
                    choices=[ "current", "true_semihard", "random_valid", "tf_legacy", "mixed_valid"], default="current",)
    parser.add_argument("--mixed_violation_prob", type=float, default=0.10,)
    parser.add_argument("--pair_mode", choices=["all_pairs", "dynamic"], default="all_pairs")
    parser.add_argument("--val_size", type=int, default=0)
    parser.add_argument("--val_every_steps", type=int, default=100)
    parser.add_argument("--val_rank_traces", type=int, default=1000)
    parser.add_argument("--val_rank_runs", type=int, default=20)

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
    alpha_mine=alpha_value if args.alpha_mine is None else args.alpha_mine

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
    selected_indices=None

    if args.selected_indices_path is not None:
        selected_indices=np.load(args.selected_indices_path)
        print("Loaded fixed profiling indices:", args.selected_indices_path)
        print("Number of fixed indices:", len(selected_indices)) 

    selection_rng_state=np.random.get_state()
    x_n, labels_n, x_limited, labels_limited, label_2_id, id_2_label = \
        getCLSidDict(data_path=data_path, n_traces=n_traces, attack_size=args.attack_size,\
        sample_num_limit=sample_num_limit, leakage_model=leakage_model, target_byte=target_byte, selected_indices=selected_indices, tracewindow=args.tracewindow)

    print("All N profiling traces:", x_n.shape)
    print("Triplet subset:", x_limited.shape)

    # =========================================================================
    # Build positive pairs
    # =========================================================================
    if args.pair_mode=="dynamic":
        a_ids=p_ids=np.empty(0, dtype=np.int64)
        print("Positive pairs will be resampled every epoch")
    else:
        a_ids, p_ids=build_positive_pairs(sorted(label_2_id.keys()), label_2_id)
        print("Number of positive pairs:", len(a_ids))
    # =========================================================================
    # Build Tripletpower embedding network
    # =========================================================================
    model=build_cnn_best(input_shape=(x_limited.shape[1], 1), emb_size=256, classification=False)

    model=model.to(device)

    # =========================================================================
    # Sanity Check 
    # =========================================================================
    xt = torch.as_tensor(
        x_limited,
        dtype=torch.float32,
        device=device
    ).unsqueeze(-1)

    model.eval()

    with torch.no_grad():
        emb = model(xt)
        norms = torch.linalg.vector_norm(emb, dim=1)

    print(
        "[INIT INPUT]",
        "min=", xt.min().item(),
        "max=", xt.max().item(),
        "mean=", xt.mean().item(),
        "std=", xt.std().item()
    )

    print(
        "[INIT EMB]",
        "mean_norm=", norms.mean().item(),
        "min_norm=", norms.min().item(),
        "max_norm=", norms.max().item(),
        "zero_fraction=",
        (norms < 1e-8).float().mean().item()
    )
    # =========================================================================
    # Split Validation data and set its Function
    # =========================================================================
    validation_fn=None

    if args.val_size>0:
        pool=load_dataset(data_path=data_path, attack_size=args.attack_size, which_one="train")

        if selected_indices is None:
            saved_rng=np.random.RandomState()
            saved_rng.set_state(selection_rng_state)
            train_idx=saved_rng.choice(len(pool["X_train"]), size=n_traces, replace=False)
        else:
            train_idx=np.asarray(selected_indices, dtype=np.int64)

        start, end=args.tracewindow

        assert np.array_equal(x_n, pool["X_train"][train_idx, start:end]), "Profiling indices mismatch"

        available=np.setdiff1d(np.arange(len(pool["X_train"])), train_idx)

        if args.val_size>len(available):
            raise ValueError("Validation size larger than unused profiling pool")

        val_idx=np.random.default_rng(seed+2026).choice(available, size=args.val_size, replace=False)

        val_x=pool["X_train"][val_idx, start:end].copy()
        val_pt=pool["plaintext"][val_idx].copy()
        val_key=np.asarray(pool["key"]).copy()
        del pool

        assert val_key.ndim==1, "Validation currently requires a fixed key"

        np.save(ckpt_path.parent/"profiling_indices.npy", train_idx)
        np.save(ckpt_path.parent/"validation_indices.npy", val_idx)

        history_path=ckpt_path.parent/"validation_history.csv"
        history_path.write_text("epoch,step,hw_accuracy,mean_rank,zero_fraction\n")
        val_root=ckpt_path.parent/"validation_rank"

        def validation_fn(eval_model, epoch, step):
            classifier=train_knn(eval_model, x_n, labels_n, n_neighbors=n_neighbors, leakage_model=leakage_model)
            probs=predict_knn_prob(eval_model, classifier, val_x, leakage_model=leakage_model)

            expected=get_labels(val_pt, int(val_key[target_byte]), target_byte, leakage_model)
            acc=float(np.mean(np.argmax(probs, axis=1)==expected))

            val_emb=extract_embeddings(val_x, eval_model)
            zero=float(np.mean(np.linalg.norm(val_emb, axis=1)<1e-8))

            py_state, np_state=random.getstate(), np.random.get_state()

            try:
                random.seed(2026)
                np.random.seed(2026)

                ranking_curve(
                    preds=probs, key=val_key, plaintext=val_pt, target_byte=target_byte,
                    rank_root=val_root, leakage_model=leakage_model,
                    trace_num_max=min(args.val_rank_traces, len(val_x)),
                    num_averaged=args.val_rank_runs
                )
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)

            with np.load(val_root/"ranking_raw_data.npz") as raw:
                val_rank=float(raw["y"][-1])

            with history_path.open("a") as f:
                f.write(f"{epoch+1},{step},{acc:.6f},{val_rank:.4f},{zero:.6f}\n")

            print(f"[VAL] epoch={epoch+1} step={step} HW_acc={acc:.4f} mean_rank={val_rank:.2f} zero={zero:.2%}")

            return val_rank
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
        alpha_mine=alpha_mine,
        negative_mode=(args.negative_mode),
        legacy_label_2_id=label_2_id,
        mixed_violation_prob=args.mixed_violation_prob,
        loss_reduction=args.loss_reduction,
        pair_mode=args.pair_mode,
        validation_fn=validation_fn,
        val_every_steps=args.val_every_steps
    )   

    if args.val_size > 0:
        val_ckpt=ckpt_path.parent/"triplet_val_best.pt"
        assert val_ckpt.is_file(), f"Validation checkpoint not found: {val_ckpt}"
        model.load_state_dict(torch.load(val_ckpt, map_location="cpu", weights_only=True))
        model.to(device).eval()
        print(f"[FINAL ATTACK] Using validation-selected checkpoint: {val_ckpt}")

    # ============================  =============================================
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
    data_dict=load_dataset(data_path=data_path, attack_size=args.attack_size, which_one="test")
    attack_traces, attack_label, attack_plaintext, attack_real_key=dissemble_data_dict(data_dict=data_dict, tracewindow=args.tracewindow, which_one="test")

    # =========================================================================
    # Attack
    # ========================================================================= 
    attack_probabilities = predict_knn_prob(
        model,
        classifier,
        attack_traces,
        leakage_model=leakage_model
    )

    attack_expected=get_labels(attack_plaintext, int(attack_real_key[target_byte]), target_byte, leakage_model)
    attack_pred=np.argmax(attack_probabilities,axis=1)
    attack_acc=accuracy_score(attack_expected,attack_pred)
    print(f"TripletPower attack classification accuracy: {attack_acc:.6f}")
    # =========================================================================
    # Key-rank
    # =========================================================================
    ranking_curve(preds=attack_probabilities, key=attack_real_key, plaintext=attack_plaintext, target_byte=target_byte, \
                rank_root=rank_root, leakage_model=leakage_model, trace_num_max=args.trace_num_max, num_averaged=args.num_averaged)

    print("TripletPower pipeline finished.")

if __name__=="__main__":
    main()

