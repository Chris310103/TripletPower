"""TensorFlow TripletPower baseline: same 500 profiling traces as PyTorch.

Replace triplet/train_tf_tp_baseline.py in the TripletPower repository.
Run from the repository root with the existing triplet-tf environment.

Uses professor's original triplet.train() (including original positive-pair
sampling, generator, negative mining, model, and optimizer). This file only
supplies the same inputs as the PyTorch comparison and evaluates independent
train/validation cosine separation at the final epoch.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Must precede TensorFlow imports.

from pathlib import Path
from types import SimpleNamespace
import random
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

import triplet.triplet as legacy_triplet
from tools.loadData import get_labels


# Compatibility with TensorFlow 2.21 + tf-keras; leave legacy trainer unchanged.
def tf21_identity_loss(y_true, y_pred):
    return K.mean(y_pred)


def tf21_cosine_triplet_loss(inputs):
    positive_sim, negative_sim = inputs
    return K.maximum(0.0, negative_sim - positive_sim + float(legacy_triplet.alpha_value))


legacy_triplet.identity_loss = tf21_identity_loss
legacy_triplet.cosine_triplet_loss = tf21_cosine_triplet_loss


# Exactly the same profiling/validation indices used in the PyTorch runs.
DATA_PATH = Path("../Target 1/X1_K1_200k_L11.npz")
INDICES_DIR = Path(
    "Output/triplet_pytorch/profiling/"
    "tp_N500_ep24_no_topk_uniform_val/ckpt"
)
TRAIN_INDICES_PATH = INDICES_DIR / "profiling_indices.npy"
VAL_INDICES_PATH = INDICES_DIR / "validation_indices.npy"

RUN_NAME = "tp_tf_same500_ep8_compare"
RUN_ROOT = Path("Output/triplet_tensorflow/original_baseline") / RUN_NAME
MODEL_DIR = RUN_ROOT / "feat_model"

N_TRACES = 500
EPOCHS = 8
U = 300
SEED = 42
TARGET_BYTE = 2
LEAKAGE_MODEL = "HW"
TRACE_START, TRACE_END = 1800, 2800
ATTACK_SIZE = 10000
N_PAIR_SAMPLES = 10000


def cosine_health(name, feat_model, x, labels):
    """Match train_torch.embedding_health() cosine-gap sampling."""
    embeddings = feat_model.predict(x[..., None], batch_size=256, verbose=0)
    labels = np.asarray(labels, dtype=np.int64)
    assert embeddings.shape[0] == len(labels)

    norms = np.linalg.norm(embeddings, axis=1)
    usable = norms >= 1e-8
    direction = embeddings / np.maximum(norms[:, None], 1e-8)

    rng = np.random.default_rng(2026)
    a = rng.integers(0, len(embeddings), size=N_PAIR_SAMPLES)
    b = rng.integers(0, len(embeddings), size=N_PAIR_SAMPLES)

    valid = (a != b) & usable[a] & usable[b]
    cosines = np.einsum("ij,ij->i", direction[a], direction[b])
    same = valid & (labels[a] == labels[b])
    diff = valid & (labels[a] != labels[b])
    if not same.any() or not diff.any():
        raise RuntimeError(f"{name}: insufficient valid same/different HW pairs")

    pos = float(cosines[same].mean())
    neg = float(cosines[diff].mean())
    print(f"[{name}] pos={pos:.4f} neg={neg:.4f} gap={pos-neg:.4f} "
          f"zero_fraction={(~usable).mean():.2%}", flush=True)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    for path in (DATA_PATH, TRAIN_INDICES_PATH, VAL_INDICES_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path} (run from ~/TripletPower)")

    train_idx = np.asarray(np.load(TRAIN_INDICES_PATH), dtype=np.int64)
    val_idx = np.asarray(np.load(VAL_INDICES_PATH), dtype=np.int64)
    if len(train_idx) != N_TRACES or len(np.unique(train_idx)) != N_TRACES:
        raise ValueError("profiling_indices.npy is not 500 distinct trace indices")
    if len(np.unique(val_idx)) != len(val_idx):
        raise ValueError("validation_indices.npy contains repeated indices")
    if np.intersect1d(train_idx, val_idx).size:
        raise ValueError("Profiling and validation indices overlap")

    # Read raw TP traces and plaintext exactly once. No normalization.
    with np.load(DATA_PATH) as raw:
        traces = raw["power_trace"]
        plaintext = raw["plain_text"]
        key = raw["key"]

    train_pool_size = len(traces) - ATTACK_SIZE
    if train_pool_size <= 0:
        raise ValueError("Dataset is smaller than ATTACK_SIZE")
    if (train_idx.min() < 0 or val_idx.min() < 0 or
            train_idx.max() >= train_pool_size or val_idx.max() >= train_pool_size):
        raise ValueError("Profiling/validation indices must be in the non-attack pool")
    if plaintext.shape[0] != len(traces):
        raise ValueError("Trace/plaintext length mismatch")

    train_x = traces[train_idx, TRACE_START:TRACE_END].astype(np.float32)
    val_x = traces[val_idx, TRACE_START:TRACE_END].astype(np.float32)
    train_y = np.asarray(get_labels(
        plaintext[train_idx], int(key[TARGET_BYTE]), TARGET_BYTE, LEAKAGE_MODEL
    ), dtype=np.int64)
    val_y = np.asarray(get_labels(
        plaintext[val_idx], int(key[TARGET_BYTE]), TARGET_BYTE, LEAKAGE_MODEL
    ), dtype=np.int64)

    assert train_x.shape == (N_TRACES, TRACE_END - TRACE_START)
    assert val_x.shape[1] == TRACE_END - TRACE_START
    assert set(train_y).issubset(set(range(9)))
    assert set(val_y).issubset(set(range(9)))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("ORIGINAL TF TRIPLETPower / MATCHED PYTORCH PROFILING SET")
    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))
    print("Data:", DATA_PATH)
    print("N / U / epochs:", N_TRACES, U, EPOCHS)
    print("Train indices:", TRAIN_INDICES_PATH)
    print("Validation indices:", VAL_INDICES_PATH)
    print("Train shape:", train_x.shape, "Val shape:", val_x.shape)
    print("Train HW counts:", np.bincount(train_y, minlength=9))
    print("Attack set excluded from training and validation")
    print("=" * 72, flush=True)

    opts = SimpleNamespace(
        epochs=EPOCHS,
        nsamples=U,
        target_byte=TARGET_BYTE,
        leakage_model=LEAKAGE_MODEL,
    )

    # Professor's ORIGINAL training function: original A/P generator,
    # legacy negative mining, batch order, network, and RMSprop.
    start = time.time()
    final_model = legacy_triplet.train(opts, train_x, train_y, str(MODEL_DIR) + "/")
    print("Training seconds:", round(time.time() - start, 2))

    final_path = MODEL_DIR / "final_model.h5"
    final_model.save(str(final_path))
    print("Saved TF FINAL:", final_path, flush=True)

    # Compare the same cosine metric used by train_torch.py.
    cosine_health("TF TRAIN", final_model, train_x, train_y)
    cosine_health("TF VAL", final_model, val_x, val_y)
    print("DONE: no kNN/Key Rank; validation is disjoint from training.")


if __name__ == "__main__":
    main()