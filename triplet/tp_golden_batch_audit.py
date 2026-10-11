#!/usr/bin/env python3
"""Cross-environment TripletPower Golden Batch audit.

One file, two independent Python environments. No TensorFlow is imported by
--stage torch; no PyTorch is imported by --stage tf.

Run from repo root:

  # activate the PyTorch environment
  python tp_golden_split_audit.py --stage torch \
      --data_path '../Target 1/X1_K1_200k_L11.npz' \
      --checkpoint 'Output/.../triplet_val_best.pt' \
      --out 'audit/tp_golden.npz'

  # activate the TensorFlow environment
  python tp_golden_split_audit.py --stage tf --out 'audit/tp_golden.npz'

Nothing in training or checkpoint paths is modified; the sole generated file
is --out, containing input batches, imported weights and selected numerical
measurements. This is a two-step optimizer comparison from a *new* state, not
an exact restart of historical training.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# The script can be placed either in the repository root or under triplet/.
_here = Path(__file__).resolve().parent
_repo_root = _here if (_here / "tools").is_dir() else _here.parent
if not (_repo_root / "tools").is_dir():
    raise RuntimeError("Place this script in the TripletPower root or triplet/ directory")
sys.path.insert(0, str(_repo_root))

# TF layer name, PyTorch module name, and whether the weight is Conv1d.
WEIGHT_LAYER_NAMES = (
    "block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1",
    "block5_conv1", "fc1", "fc2", "predictions",
)
ACTIVATION_NAMES = (
    "block1_conv1", "block1_pool", "block2_conv1", "block2_pool",
    "block3_conv1", "block3_pool", "block4_conv1", "block4_pool",
    "block5_conv1", "block5_pool", "fc1", "fc2", "predictions",
)


def to_tf_layout(tensor, is_conv):
    arr = tensor.detach().cpu().numpy()
    return arr.transpose(2, 1, 0) if is_conv else arr.T


def build_triplets(path, start, end, pool_size, count, seed):
    from tools.loadData import get_labels
    with np.load(path, allow_pickle=False) as f:
        traces = np.asarray(f["power_trace"][:pool_size, start:end], dtype=np.float32)
        plaintext = np.asarray(f["plain_text"][:pool_size])
        key = np.asarray(f["key"])
    if key.ndim != 1:
        raise ValueError("This audit currently requires a fixed-key NPZ dataset")
    labels = np.asarray(get_labels(plaintext, int(key[2]), 2, "HW"), dtype=np.int64)
    groups = defaultdict(list)
    for i, hw in enumerate(labels):
        groups[int(hw)].append(i)
    valid = [hw for hw, ids in groups.items() if len(ids) >= 2]
    if len(valid) < 2:
        raise ValueError("Need >=2 classes with >=2 traces in the selected pool")
    rng = np.random.default_rng(seed)
    a, p, n = [], [], []
    for _ in range(count):
        cls = int(rng.choice(valid))
        aa, pp = rng.choice(groups[cls], 2, replace=False)
        neg_cls = int(rng.choice([v for v in groups if v != cls]))
        nn = int(rng.choice(groups[neg_cls]))
        a.append(int(aa)); p.append(int(pp)); n.append(nn)
    assert np.all(labels[a] == labels[p]) and np.all(labels[a] != labels[n])
    return [traces[idx, :, None].copy() for idx in (a, p, n)]


def torch_layers(model):
    return {
        "block1_conv1": (model.feature_extractor[1], True),
        "block2_conv1": (model.feature_extractor[4], True),
        "block3_conv1": (model.feature_extractor[7], True),
        "block4_conv1": (model.feature_extractor[10], True),
        "block5_conv1": (model.feature_extractor[13], True),
        "fc1": (model.fc[1], False),
        "fc2": (model.fc[3], False),
        "predictions": (model.output_layer[0], False),
    }


def torch_activations(model, x, torch):
    modules = [
        model.feature_extractor[2], model.feature_extractor[3],
        model.feature_extractor[5], model.feature_extractor[6],
        model.feature_extractor[8], model.feature_extractor[9],
        model.feature_extractor[11], model.feature_extractor[12],
        model.feature_extractor[14], model.feature_extractor[15],
        model.fc[2], model.fc[4], model.output_layer[1],
    ]
    results = {}
    handles = []
    for name, module in zip(ACTIVATION_NAMES, modules):
        handles.append(module.register_forward_hook(
            lambda _m, _inp, out, label=name:
            results.__setitem__(label, out.detach().cpu().numpy().copy())
        ))
    try:
        with torch.no_grad():
            model(torch.from_numpy(x))
    finally:
        for h in handles:
            h.remove()
    for key, val in list(results.items()):
        if val.ndim == 3:
            results[key] = val.transpose(0, 2, 1)
    return results


def sampled_gradient_indices(grad, seed=42):
    """Random probes plus top-magnitude probes: do not miss sparse gradients."""
    flat = np.asarray(grad).ravel()
    n = len(flat)
    rng = np.random.default_rng(seed)
    random_ids = rng.choice(n, min(n, 512), replace=False)
    if n <= 32:
        top_ids = np.arange(n)
    else:
        top_ids = np.argpartition(np.abs(flat), -32)[-32:]
    return np.unique(np.concatenate((random_ids, top_ids))).astype(np.int64)


def run_torch(args):
    import torch
    import torch.nn.functional as F
    from tools.model_zoo_pytorch import build_cnn_best

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    a, p, n = build_triplets(
        args.data_path, *args.tracewindow, args.pool_size, args.triplets, args.seed
    )
    model = build_cnn_best((a.shape[1], 1), 256, classification=False)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        print("Checkpoint:", args.checkpoint)
    else:
        print("WARNING: no checkpoint provided; auditing an initialized network")
    model.cpu().eval()
    layers = torch_layers(model)

    payload = {
        "a": a, "p": p, "n": n,
        "learning_rate": np.array(args.lr, dtype=np.float32),
    }
    for name, (module, conv) in layers.items():
        payload[f"w__{name}"] = to_tf_layout(module.weight, conv).copy()
        payload[f"b__{name}"] = module.bias.detach().cpu().numpy().copy()

    activations = torch_activations(model, a, torch)
    for name, val in activations.items():
        payload[f"act__{name}"] = val.copy()

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.lr, alpha=0.9, eps=1e-7,
        momentum=0.0, centered=False,
    )
    model.train()
    for step in (1, 2):
        optimizer.zero_grad()
        aa, pp, nn = [model(torch.from_numpy(x)) for x in (a, p, n)]
        pos_sim = F.cosine_similarity(aa, pp, dim=1)
        neg_sim = F.cosine_similarity(aa, nn, dim=1)
        loss = torch.relu(neg_sim - pos_sim + 0.5).mean()
        payload[f"loss__{step}"] = np.asarray(loss.detach().item(), dtype=np.float64)
        loss.backward()
        for name, (module, conv) in layers.items():
            grad = to_tf_layout(module.weight.grad, conv).copy()
            idx = sampled_gradient_indices(grad, args.seed + step)
            payload[f"gi__{step}__{name}"] = idx
            payload[f"gv__{step}__{name}"] = grad.ravel()[idx].copy()
            payload[f"gn__{step}__{name}"] = np.array(np.linalg.norm(grad.ravel().astype(np.float64)))
            payload[f"gm__{step}__{name}"] = np.array(np.max(np.abs(grad)))
        optimizer.step()
        for name, (module, conv) in layers.items():
            new_w = to_tf_layout(module.weight, conv)
            idx = payload[f"gi__{step}__{name}"]
            payload[f"wv__{step}__{name}"] = new_w.ravel()[idx].copy()
            diff = new_w.astype(np.float64) - payload[f"w__{name}"].astype(np.float64) if step == 1 else None
            if diff is not None:
                payload[f"update_norm__{step}__{name}"] = np.array(np.linalg.norm(diff.ravel()))
        print(f"Torch step {step}: triplet loss={loss.item():.8f}")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    print(f"Saved {output} ({output.stat().st_size / 1024**2:.1f} MiB)")
    print("Next: switch to your TensorFlow environment and run --stage tf --out", output)


def compare(label, actual, reference):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        print(f"{label:30s} SHAPE ERROR: {a.shape} vs {b.shape}")
        return float('inf'), float('inf')
    err = a - b
    absmax = float(np.max(np.abs(err))) if a.size else 0.
    rel = float(np.linalg.norm(err.ravel()) / max(np.linalg.norm(b.ravel()), 1e-12))
    print(f"{label:30s} max_abs={absmax:.3e} relative_l2={rel:.3e}")
    return absmax, rel


def run_tf(args):
    import tensorflow as tf
    from tools.model_zoo import cnn_best

    print("TensorFlow version:", tf.__version__)
    tf.random.set_seed(args.seed)
    if hasattr(tf.config, "threading"):
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)

    with np.load(args.out, allow_pickle=False) as saved:
        data = {name: saved[name] for name in saved.files}
    a, p, n = (data[x] for x in ("a", "p", "n"))
    model = cnn_best((a.shape[1], 1), emb_size=256, classification=False)
    for name in WEIGHT_LAYER_NAMES:
        model.get_layer(name).set_weights([data[f"w__{name}"], data[f"b__{name}"]])

    print("\n[1] Forward layer comparisons")
    probe = tf.keras.Model(model.inputs, [model.get_layer(k).output for k in ACTIVATION_NAMES])
    tf_outs = probe(tf.convert_to_tensor(a), training=False)
    largest_forward_rel = 0.
    for name, output in zip(ACTIVATION_NAMES, tf_outs):
        _, relative = compare(name, output.numpy(), data[f"act__{name}"])
        largest_forward_rel = max(largest_forward_rel, relative)

    lr = float(data["learning_rate"])
    rms_slots = [
        tf.Variable(tf.zeros_like(v), trainable=False)
        for v in model.trainable_variables
    ]
    print("\nOptimizer: TF 2.3 style RMSprop")
    max_loss_diff = 0.
    for step in (1, 2):
        with tf.GradientTape() as tape:
            aa, pp, nn = (model(tf.convert_to_tensor(x), training=True) for x in (a, p, n))
            pos_sim = tf.keras.layers.Dot(axes=-1, normalize=True)([aa, pp])
            neg_sim = tf.keras.layers.Dot(axes=-1, normalize=True)([aa, nn])
            loss = tf.reduce_mean(tf.nn.relu(neg_sim - pos_sim + 0.5))
        variables = model.trainable_variables
        grads = tape.gradient(loss, variables)
        print(f"\n[2] Gradients for optimizer step {step}")
        diff, _ = compare("triplet loss", loss.numpy(), data[f"loss__{step}"])
        max_loss_diff = max(max_loss_diff, diff)
        for name in WEIGHT_LAYER_NAMES:
            tf_weight = model.get_layer(name).trainable_weights[0]
            index = next((i for i, v in enumerate(variables) if v is tf_weight), None)
            if index is None:
                # Some Keras versions wrap the same variable in different objects.
                matches = [i for i, v in enumerate(variables) if v.name == tf_weight.name]
                if len(matches) != 1:
                    raise RuntimeError(f"Cannot locate variable {tf_weight.name}")
                index = matches[0]
            grad = grads[index]
            if grad is None:
                raise RuntimeError(f"No gradient for TF layer {name}")
            grad_np = np.asarray(grad.numpy())
            idx = data[f"gi__{step}__{name}"]
            compare("grad samples " + name, grad_np.ravel()[idx], data[f"gv__{step}__{name}"])
            compare("grad norm " + name, np.linalg.norm(grad_np.ravel().astype(np.float64)), data[f"gn__{step}__{name}"])
            compare("grad max  " + name, np.max(np.abs(grad_np)), data[f"gm__{step}__{name}"])

        for v, g, rms in zip(variables, grads, rms_slots):
            rms.assign(0.9 * rms + 0.1 * tf.square(g))
            v.assign_sub(lr * g / (tf.sqrt(rms) + 1e-7))
        print(f"\n[3] Updated weights for optimizer step {step}")
        for name in WEIGHT_LAYER_NAMES:
            weight = model.get_layer(name).get_weights()[0]
            idx = data[f"gi__{step}__{name}"]
            compare("weight samples " + name, weight.ravel()[idx], data[f"wv__{step}__{name}"])

    print("\nAUDIT FINISHED. Inspect earliest significant discrepancy.")
    print("Note: same checkpoint, fixed triplets, optimizer state starts at zero in BOTH frameworks.")
    print("This does not verify negative mining, batch ordering, or representation generalization.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("torch", "tf"), required=True)
    parser.add_argument("--out", default="audit/tp_golden.npz")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tracewindow", nargs=2, type=int, default=[1800, 2800])
    parser.add_argument("--pool_size", type=int, default=1000)
    parser.add_argument("--triplets", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    if args.stage == "torch":
        if not args.data_path:
            parser.error("--stage torch requires --data_path")
        run_torch(args)
    else:
        run_tf(args)


if __name__ == "__main__":
    main()

