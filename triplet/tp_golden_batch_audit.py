#!/usr/bin/env python3
"""TripletPower: one-batch, same-weights TensorFlow vs PyTorch differential audit.

Run from the TripletPower repository root (where tools/ is importable):
    python tp_golden_batch_audit.py --data_path '../Target 1/X1_K1_200k_L11.npz'

Does not train a model or modify any existing training/checkpoint file.
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU: avoid differing GPU algorithms
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import random
import numpy as np
import torch
import tensorflow as tf
from collections import defaultdict

from tools.loadData import get_labels
from tools.model_zoo import cnn_best as build_tf_cnn
from tools.model_zoo_pytorch import build_cnn_best as build_torch_cnn
from triplet.triplet_pytorch import cosine_triplet_loss


def compare(label, actual, reference):
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        raise AssertionError(f"{label}: shape {a.shape} != {b.shape}")
    delta = a - b
    mx = float(np.max(np.abs(delta)))
    rel = float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(b.ravel()), 1e-12))
    print(f"{label:22s} max_abs={mx:.4e}  relative_l2={rel:.4e}")
    return mx, rel


def to_tf_array(tensor, is_conv):
    arr = tensor.detach().cpu().numpy()
    return arr.transpose(2, 1, 0) if is_conv else arr.T


def setup_models(length, checkpoint=None):
    p = build_torch_cnn(input_shape=(length, 1), emb_size=256, classification=False)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        p.load_state_dict(state, strict=True)
        print("Loaded PyTorch checkpoint:", checkpoint)
        print("Note: optimizer state is reset; this checks a fresh RMSprop step, not an exact resumed update.")
    p.eval()
    t = build_tf_cnn(input_shape=(length, 1), emb_size=256, classification=False)
    # (TF named layer, PyTorch module, is_conv)
    layers = [
        ("block1_conv1", p.feature_extractor[1], True),
        ("block2_conv1", p.feature_extractor[4], True),
        ("block3_conv1", p.feature_extractor[7], True),
        ("block4_conv1", p.feature_extractor[10], True),
        ("block5_conv1", p.feature_extractor[13], True),
        ("fc1", p.fc[1], False),
        ("fc2", p.fc[3], False),
        ("predictions", p.output_layer[0], False),
    ]
    for name, pt_layer, is_conv in layers:
        tf_layer = t.get_layer(name)
        tf_layer.set_weights([
            to_tf_array(pt_layer.weight, is_conv),
            pt_layer.bias.detach().cpu().numpy(),
        ])
        compare(f"weight {name}", to_tf_array(pt_layer.weight, is_conv), tf_layer.get_weights()[0])
    return p, t, layers


def triplets_from_real_traces(path, start, end, pool_size, count, seed):
    with np.load(path, allow_pickle=False) as raw:
        x = np.asarray(raw["power_trace"][:pool_size, start:end], dtype=np.float32)
        pt = np.asarray(raw["plain_text"][:pool_size])
        key = np.asarray(raw["key"])
    if key.ndim != 1:
        raise ValueError("This audit expects a fixed-key NPZ dataset")
    labs = np.asarray(get_labels(pt, int(key[2]), 2, "HW"), dtype=np.int64)
    groups = defaultdict(list)
    for i, cls in enumerate(labs):
        groups[int(cls)].append(i)
    valid = [c for c, ids in groups.items() if len(ids) >= 2]
    if len(valid) < 2:
        raise ValueError("Not enough HW classes with 2 or more traces")
    rng = np.random.default_rng(seed)
    a, pos, neg = [], [], []
    for _ in range(count):
        cls = int(rng.choice(valid))
        aa, pp = rng.choice(groups[cls], size=2, replace=False)
        neg_cls = int(rng.choice([c for c in groups if c != cls]))
        nn = int(rng.choice(groups[neg_cls]))
        a.append(int(aa)); pos.append(int(pp)); neg.append(nn)
    assert np.all(labs[a] == labs[pos]) and np.all(labs[a] != labs[neg])
    print(f"Fixed real-data triplets: {len(a)}  pool={len(x)}  window=[{start}, {end})")
    return x[a, :, None].copy(), x[pos, :, None].copy(), x[neg, :, None].copy()


def forward_intermediates(p_model, t_model, x):
    # TF conv has integrated ReLU, so compare to Torch post-ReLU, not Conv1d pre-ReLU.
    pt_modules = [
        p_model.feature_extractor[2], p_model.feature_extractor[3],
        p_model.feature_extractor[5], p_model.feature_extractor[6],
        p_model.feature_extractor[8], p_model.feature_extractor[9],
        p_model.feature_extractor[11], p_model.feature_extractor[12],
        p_model.feature_extractor[14], p_model.feature_extractor[15],
        p_model.fc[2], p_model.fc[4], p_model.output_layer[1],
    ]
    names = [
        "block1_conv1", "block1_pool", "block2_conv1", "block2_pool",
        "block3_conv1", "block3_pool", "block4_conv1", "block4_pool",
        "block5_conv1", "block5_pool", "fc1", "fc2", "predictions",
    ]
    seen = {}
    handlers = []
    for name, module in zip(names, pt_modules):
        handlers.append(module.register_forward_hook(
            lambda mod, ins, out, key=name: seen.__setitem__(key, out.detach().numpy().copy())
        ))
    try:
        with torch.no_grad():
            p_model(torch.from_numpy(x))
    finally:
        for h in handlers:
            h.remove()
    probe = tf.keras.Model(t_model.inputs, [t_model.get_layer(n).output for n in names])
    tf_outputs = probe(tf.convert_to_tensor(x), training=False)
    print("\n[1] Layer-by-layer forward comparison (same weights, same traces)")
    for name, tf_val in zip(names, tf_outputs):
        pt_val = seen[name]
        if pt_val.ndim == 3:  # PT is N,C,T; TF is N,T,C
            pt_val = pt_val.transpose(0, 2, 1)
        compare(name, pt_val, tf_val.numpy())


def get_tf_gradient(var, variables, gradients):
    for v, g in zip(variables, gradients):
        if v is var:
            if g is None:
                raise ValueError(f"No TensorFlow gradient for {var.name}")
            return g.numpy()
    raise KeyError(f"TensorFlow variable not found: {var.name}")


def gradient_and_step(p_model, t_model, layers, a, pos, neg, lr):
    p_model.train()
    pt_opt = torch.optim.RMSprop(p_model.parameters(), lr=lr, alpha=0.9, eps=1e-7,
                                 momentum=0.0, centered=False)
    tf_opt = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=0.9, epsilon=1e-7,
                                         momentum=0.0, centered=False)

    for step in (1, 2):
        pt_opt.zero_grad()
        ap = p_model(torch.from_numpy(a))
        pp = p_model(torch.from_numpy(pos))
        np_ = p_model(torch.from_numpy(neg))
        pt_loss = cosine_triplet_loss(ap, pp, np_, alpha=0.5, reduction="mean")
        pt_loss.backward()

        a_tf, p_tf, n_tf = (tf.convert_to_tensor(v) for v in (a, pos, neg))
        with tf.GradientTape() as tape:
            at = t_model(a_tf, training=True)
            pt = t_model(p_tf, training=True)
            nt = t_model(n_tf, training=True)
            # The SAME Dot(normalize=True) operation that professor's original code uses.
            pos_sim = tf.keras.layers.Dot(axes=-1, normalize=True)([at, pt])
            neg_sim = tf.keras.layers.Dot(axes=-1, normalize=True)([at, nt])
            tf_loss = tf.reduce_mean(tf.nn.relu(neg_sim - pos_sim + 0.5))
        vars_tf = t_model.trainable_variables
        grads_tf = tape.gradient(tf_loss, vars_tf)

        print(f"\n[2] Gradient comparison, fixed triplets, step {step}")
        compare("triplet loss", pt_loss.detach().numpy(), tf_loss.numpy())
        for name, pt_layer, is_conv in layers:
            tf_layer = t_model.get_layer(name)
            grad_t = get_tf_gradient(tf_layer.trainable_weights[0], vars_tf, grads_tf)
            grad_p = to_tf_array(pt_layer.weight.grad, is_conv)
            compare(f"grad {name}", grad_p, grad_t)

        tf_opt.apply_gradients(zip(grads_tf, vars_tf))
        pt_opt.step()
        print(f"\n[3] Updated-weight comparison after RMSprop step {step}")
        for name, pt_layer, is_conv in layers:
            wtf = t_model.get_layer(name).get_weights()[0]
            compare(f"updated {name}", to_tf_array(pt_layer.weight, is_conv), wtf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--tracewindow", type=int, nargs=2, default=[1800, 2800])
    parser.add_argument("--pool_size", type=int, default=1000)
    parser.add_argument("--triplets", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tf.random.set_seed(args.seed)
    print("TensorFlow:", tf.__version__, "PyTorch:", torch.__version__)
    a, p, n = triplets_from_real_traces(args.data_path, *args.tracewindow,
                                       args.pool_size, args.triplets, args.seed)
    pt, tf_model, layers = setup_models(a.shape[1], args.checkpoint)
    forward_intermediates(pt, tf_model, a)
    gradient_and_step(pt, tf_model, layers, a, p, n, args.lr)
    print("\nCompleted. Large first-layer errors imply a forward/model mismatch; ")
    print("first gradient-only errors imply loss/backward mismatch; ")
    print("first post-update errors imply optimizer/update mismatch.")
    print("A match DOES NOT prove miners, data selection, or generalization are correct.")


if __name__ == "__main__":
    main()
