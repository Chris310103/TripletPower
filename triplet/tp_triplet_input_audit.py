#!/usr/bin/env python3
"""Audit the existing TripletPower input pipeline at one fixed checkpoint.

Run from the TripletPower repository root. This is evaluation-only: it does not
train, modify checkpoints, or alter any of the existing mining implementations.
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tools.model_zoo_pytorch import build_cnn_best
from triplet.triplet_pytorch import (
    getCLSidDict, build_positive_pairs, build_negatives,
    build_similarities, extract_embeddings,
)


def get_args():
    ap = argparse.ArgumentParser(description="Audit real TripletPower A/P/N selection and nonzero losses")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--negative_mode", default="true_semihard",
                    choices=["true_semihard", "current", "tf_legacy", "random_valid", "mixed_valid"])
    ap.add_argument("--pair_sampling", default="original", choices=["original", "uniform", "hw_sqrt"])
    ap.add_argument("--epoch_number", type=int, default=1,
                    help="1-based epoch for reproducing resampled A/P pairs (not a history replay)")
    ap.add_argument("--n_traces", type=int, default=500)
    ap.add_argument("--attack_size", type=int, default=10000)
    ap.add_argument("--target_byte", type=int, default=2)
    ap.add_argument("--sample_num_limit", type=int, default=300)
    ap.add_argument("--tracewindow", type=int, nargs=2, default=[1800, 2800])
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha_value", type=float, default=0.5)
    ap.add_argument("--alpha_mine", type=float, default=0.8)
    ap.add_argument("--selected_indices_path", default=None,
                    help="Only needed if the original run used fixed profiling indices")
    return ap.parse_args()


def main():
    args = get_args()
    if args.epoch_number < 1:
        raise ValueError("--epoch_number must be >=1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    saved_indices = ckpt.parent / "profiling_indices.npy"
    if saved_indices.is_file():
        expected_idx = np.asarray(np.load(saved_indices), dtype=np.int64)
        if args.selected_indices_path is not None:
            given_idx = np.asarray(np.load(args.selected_indices_path), dtype=np.int64)
            if not np.array_equal(given_idx, expected_idx):
                raise ValueError("Provided profiling indices disagree with the checkpoint indices")
        else:
            # The original script draws from the profiling pool before creating A/P pairs.
            # This reproduces that draw for the TP .npz loader's fixed 200k split.
            if str(args.data_path).endswith(".npz"):
                regenerated = np.random.RandomState(args.seed).choice(
                    200000-args.attack_size, size=args.n_traces, replace=False)
                if not np.array_equal(regenerated, expected_idx):
                    raise ValueError("Saved indices differ from seed-based selection. "
                                     "Pass --selected_indices_path with the ORIGINAL input index file")

    x_n, labels_n, traces, labels, cls_to_ids, id_to_cls = getCLSidDict(
        data_path=args.data_path, n_traces=args.n_traces,
        attack_size=args.attack_size, sample_num_limit=args.sample_num_limit,
        leakage_model="HW", target_byte=args.target_byte,
        selected_indices=np.load(args.selected_indices_path) if args.selected_indices_path else None,
        tracewindow=tuple(args.tracewindow),
    )

    a_ids, p_ids = build_positive_pairs(sorted(cls_to_ids), cls_to_ids)
    aa = np.asarray(a_ids, dtype=np.int64)
    pp = np.asarray(p_ids, dtype=np.int64)
    label_ids = np.asarray([id_to_cls[i] for i in range(len(traces))], dtype=np.int64)
    pool_ids = list(set(a_ids) | set(p_ids))  # Exact pool construction used by training.
    pool_set = set(pool_ids)

    print(f"\n[INPUT] profiling traces={len(x_n)} triplet traces={len(traces)} ")
    print(f"[INPUT] positive pairs={len(aa)} neg pool={len(pool_ids)} ")
    print(f"[CHECK A/P] same_HW={np.mean(label_ids[aa] == label_ids[pp]):.2%} "
          f"different_trace={np.mean(aa != pp):.2%}")
    assert np.all(label_ids[aa] == label_ids[pp]), "A/P CLASS BUG"
    assert np.all(aa != pp), "A/P ID BUG"

    if args.pair_sampling != "original":
        counts = np.bincount(label_ids[aa], minlength=9)
        if args.pair_sampling == "uniform":
            weights = np.ones(len(aa), dtype=np.float64)
        else:
            weights = 1.0 / np.sqrt(np.maximum(counts[label_ids[aa]], 200))
        weights /= weights.sum()
        rng = np.random.default_rng(args.seed+1009)
        for _ in range(args.epoch_number):
            chosen = rng.choice(len(aa), size=len(aa), replace=True, p=weights)
        aa, pp = aa[chosen], pp[chosen]

    used = len(aa)//args.batch_size * args.batch_size
    aa, pp = aa[:used], pp[:used]
    print(f"[INPUT] sampled mode={args.pair_sampling} epoch_number={args.epoch_number}; "
          f"consumed={used} pairs, dropped={len(a_ids)-used}")

    # Also check epoch-0 random-negative behavior on one actual batch.
    random.seed(args.seed + 19)
    init_stats = defaultdict(int)
    n0 = build_negatives(
        aa[:args.batch_size].tolist(), pp[:args.batch_size].tolist(), pool_ids,
        id_to_cls, args.alpha_mine, all_sims=None, negative_mode=args.negative_mode,
        legacy_label_2_id=cls_to_ids, stats=init_stats,
    )
    n0 = np.asarray(n0, dtype=np.int64)
    print(f"[EPOCH 0] random batch A/P same={np.mean(label_ids[aa[:len(n0)]] == label_ids[pp[:len(n0)]]):.2%} "
          f"A/N same_HW={np.mean(label_ids[aa[:len(n0)]] == label_ids[n0]):.2%}")
    if args.negative_mode in ("true_semihard", "random_valid", "mixed_valid"):
        assert not np.any(label_ids[aa[:len(n0)]] == label_ids[n0]), "Epoch-0 invalid negative"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cnn_best((traces.shape[1], 1), emb_size=256, classification=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    x_tensor = torch.from_numpy(traces).float().unsqueeze(-1).to(device)
    sims = build_similarities(model, x_tensor, batch_size=128)
    embs = torch.from_numpy(extract_embeddings(traces, model, batch_size=128))
    del model, x_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    valid_neg = {cls: [i for i in pool_ids if label_ids[i] != cls] for cls in set(id_to_cls.values())}
    stats = defaultdict(int)
    mined = []
    random.seed(args.seed + 29)  # audit-only randomness, not an exact epoch history replay
    for start in range(0, used, args.batch_size):
        end = start + args.batch_size
        part = build_negatives(
            aa[start:end].tolist(), pp[start:end].tolist(), pool_ids,
            id_to_cls, args.alpha_mine, all_sims=sims,
            negative_mode=args.negative_mode,
            valid_neg_ids_by_class=valid_neg, stats=stats,
            legacy_label_2_id=cls_to_ids,
        )
        mined.extend(part)
    nn = np.asarray(mined, dtype=np.int64)

    assert len(nn) == used, "N count mismatch"
    assert all(int(i) in pool_set for i in nn), "N outside negative pool"
    actual_same_n = int(np.count_nonzero(label_ids[aa] == label_ids[nn]))
    print(f"\n[CHECK N] mode={args.negative_mode} checked={used} "
          f"outside_pool=0 same_HW={actual_same_n} ({actual_same_n/used:.2%})")
    print("[MINER STATS] " + ", ".join(f"{k}={v} ({v/used:.2%})"
                                      for k, v in sorted(stats.items()) if v != 0))
    if args.negative_mode in ("true_semihard", "random_valid", "mixed_valid"):
        assert actual_same_n == 0, "Invalid negative selected"

    s_pos_snapshot = sims[aa, pp]
    s_neg_snapshot = sims[aa, nn]
    if args.negative_mode == "true_semihard":
        in_band = (s_neg_snapshot > s_pos_snapshot-args.alpha_mine) & (s_neg_snapshot < s_pos_snapshot)
        selected_semihard = int(in_band.sum())
        expected_semihard = stats["semihard"]
        print(f"[MINER RULE] semihard-selected={selected_semihard}; "
              f"miner-report={expected_semihard}; fallback={stats['fallback_random_valid']}")
        assert selected_semihard == expected_semihard, "True-semi-hard candidates violate mining rule"
    if args.negative_mode == "current":
        permitted = (s_neg_snapshot > s_pos_snapshot-args.alpha_mine) & (label_ids[aa] != label_ids[nn])
        print(f"[MINER RULE] candidate-compatible={int(permitted.sum())} "
              f"reported-candidate={stats['hard'] + stats['semihard']} "
              f"fallback={stats['fallback_random']}")
        assert int(permitted.sum()) >= stats['hard'] + stats['semihard'], "Current mined illegal candidates"

    # Reuse the same cosine formula as training on actual model embeddings.
    sp = F.cosine_similarity(embs[aa], embs[pp], dim=1)
    sn = F.cosine_similarity(embs[aa], embs[nn], dim=1)
    losses = torch.clamp(sn-sp+args.alpha_value, min=0)
    nonzero = losses > 0
    print(f"\n[CHECK LOSS] alpha_loss={args.alpha_value} alpha_mine={args.alpha_mine}")
    print(f"[CHECK LOSS] positive-loss={nonzero.float().mean().item():.2%} "
          f"mean_loss={losses.mean().item():.6f} "
          f"pos_sim={sp.mean().item():.4f} neg_sim={sn.mean().item():.4f}")
    if args.negative_mode == "true_semihard":
        flag = torch.from_numpy(in_band)
        if flag.any():
            print(f"[CHECK LOSS] positive-loss-among-mined={nonzero[flag].float().mean().item():.2%} "
                  f"mined_fraction={flag.float().mean().item():.2%}")
    print("\nDONE. This audits the miner at a FIXED checkpoint, not the exact historical training batches.")


if __name__ == "__main__":
    main()
