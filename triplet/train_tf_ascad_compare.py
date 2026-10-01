import os

# MUST be before TensorFlow import
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path
from types import SimpleNamespace
import random
import time

import numpy as np
import tensorflow as tf

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import triplet.triplet as legacy_triplet

from tools.ascad_loader import (
    load_dataset,
    dissemble_data_dict,
)
from tools.loadData import get_labels
from tools.key_rank_new import ranking_curve


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/ASCAD.h5"

RUN_NAME = "ascad_hw_N50000_ep20_tensorflow_legacy"

OUTPUT_ROOT = Path(
    "Output/triplet_tensorflow/profiling"
)

RUN_ROOT = OUTPUT_ROOT / RUN_NAME
MODEL_DIR = RUN_ROOT / "feat_model"
RANK_DIR = RUN_ROOT / "ranking"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RANK_DIR.mkdir(parents=True, exist_ok=True)

N_TRACES = 50000
EPOCHS = 20
U = 300

TARGET_BYTE = 2
LEAKAGE_MODEL = "HW"

N_NEIGHBORS = 10

TRACE_NUM_MAX = 5000
NUM_AVERAGED = 5

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


print("=" * 80)
print("TF LEGACY TRIPLETPower / ASCAD COMPARISON")
print("=" * 80)

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))


# ============================================================
# LOAD PROFILING DATA
# ============================================================

profiling_dict = load_dataset(
    DATA_PATH,
    which_one="train",
)

(
    x_all,
    stored_labels,
    plaintext_all,
    key,
) = dissemble_data_dict(
    profiling_dict,
    tracewindow=(0, 700),
    which_one="train",
)

x_all = np.asarray(x_all)
stored_labels = np.asarray(stored_labels)
plaintext_all = np.asarray(plaintext_all)
key = np.asarray(key)


# Match PyTorch getCLSidDict:
# with N=50000 this is a permutation of all traces.
selected_indices = np.random.choice(
    len(x_all),
    size=N_TRACES,
    replace=False,
)

x_train = x_all[selected_indices]
plaintext_train = plaintext_all[selected_indices]
stored_train = stored_labels[selected_indices]


key_byte = int(
    key[TARGET_BYTE]
)

train_y = get_labels(
    plaintext_train,
    key_byte,
    TARGET_BYTE,
    LEAKAGE_MODEL,
)

train_y = np.asarray(
    train_y,
    dtype=np.int64,
)


# ============================================================
# LABEL SANITY
# ============================================================

stored_hw = np.array(
    [
        int(v).bit_count()
        for v in stored_train
    ],
    dtype=np.int64,
)

match = np.mean(
    stored_hw == train_y
)

print("Label match:", match)
print("Profiling traces:", x_train.shape)
print(
    "HW counts:",
    np.bincount(
        train_y,
        minlength=9,
    ),
)

assert match == 1.0


# ============================================================
# OPTIONS EXPECTED BY ORIGINAL triplet.py
# ============================================================

opts = SimpleNamespace(
    epochs=EPOCHS,
    nsamples=U,
    target_byte=TARGET_BYTE,
    leakage_model=LEAKAGE_MODEL,
)


# ============================================================
# TRAIN ORIGINAL TF IMPLEMENTATION
#
# IMPORTANT:
# This deliberately uses professor's old source directly:
# - old getClsIdDict
# - old limitData
# - old generator
# - old negative mining
# - old mapping behavior
# - old cnn_best
# ============================================================

print("\n" + "=" * 80)
print("TRAIN ORIGINAL TENSORFLOW IMPLEMENTATION")
print("=" * 80)

print("N =", N_TRACES)
print("U =", U)
print("epochs =", EPOCHS)


start = time.time()


# Old source concatenates strings for CSVLogger,
# so keep the trailing slash.
model_dir_string = str(MODEL_DIR) + "/"


final_model = legacy_triplet.train(
    opts,
    x_train,
    train_y,
    model_dir_string,
)


print(
    "Training seconds:",
    time.time() - start,
)


# Old source returns FINAL in-memory model.
final_model_path = (
    MODEL_DIR / "final_model.h5"
)

final_model.save(
    str(final_model_path)
)

print(
    "Saved final model:",
    final_model_path,
)


# ============================================================
# TRAIN kNN USING ALL N PROFILING TRACES
# ============================================================

print("\nExtracting profiling embeddings...")


x_train_3d = x_train.reshape(
    (
        len(x_train),
        x_train.shape[1],
        1,
    )
)


train_embeddings = final_model.predict(
    x_train_3d,
    batch_size=256,
    verbose=1,
)


norms = np.linalg.norm(
    train_embeddings,
    axis=1,
)

print(
    "Embedding norm mean:",
    float(norms.mean()),
)

print(
    "Zero embedding fraction:",
    float(
        np.mean(
            norms < 1e-8
        )
    ),
)


knn = KNeighborsClassifier(
    n_neighbors=N_NEIGHBORS,
    weights="distance",
    metric="cosine",
    algorithm="brute",
)

knn.fit(
    train_embeddings,
    train_y,
)


# ============================================================
# LOAD ATTACK DATA
# ============================================================

attack_dict = load_dataset(
    DATA_PATH,
    which_one="test",
)

(
    x_attack,
    stored_attack_labels,
    plaintext_attack,
    attack_key,
) = dissemble_data_dict(
    attack_dict,
    tracewindow=(0, 700),
    which_one="test",
)

x_attack = np.asarray(x_attack)
plaintext_attack = np.asarray(plaintext_attack)
attack_key = np.asarray(attack_key)


attack_expected = get_labels(
    plaintext_attack,
    int(
        attack_key[
            TARGET_BYTE
        ]
    ),
    TARGET_BYTE,
    LEAKAGE_MODEL,
)


# ============================================================
# ATTACK
# ============================================================

print("\nExtracting attack embeddings...")


x_attack_3d = x_attack.reshape(
    (
        len(x_attack),
        x_attack.shape[1],
        1,
    )
)


attack_embeddings = final_model.predict(
    x_attack_3d,
    batch_size=256,
    verbose=1,
)


attack_pred = knn.predict(
    attack_embeddings
)


attack_acc = accuracy_score(
    attack_expected,
    attack_pred,
)


print("\n" + "=" * 80)
print("TENSORFLOW FINAL MODEL RESULT")
print("=" * 80)

print(
    "Attack HW accuracy:",
    attack_acc,
)


attack_probs = knn.predict_proba(
    attack_embeddings
)


print(
    "Zero probability fraction:",
    float(
        np.mean(
            attack_probs == 0
        )
    ),
)


# ============================================================
# SAME RANKING CODE USED BY PYTORCH
# ============================================================

rank_curve = ranking_curve(
    preds=attack_probs,
    key=attack_key,
    plaintext=plaintext_attack,
    target_byte=TARGET_BYTE,
    rank_root=str(RANK_DIR),
    leakage_model=LEAKAGE_MODEL,
    trace_num_max=TRACE_NUM_MAX,
    num_averaged=NUM_AVERAGED,
)


rank_curve = np.asarray(
    rank_curve
)


print(
    "Minimum rank:",
    float(
        rank_curve.min()
    ),
)

print(
    "Final rank:",
    float(
        rank_curve[-1]
    ),
)


zero_idx = np.where(
    rank_curve == 0
)[0]


if len(zero_idx):
    print(
        "First rank-0 trace:",
        int(
            zero_idx[0] + 1
        ),
    )
else:
    print(
        "Never reached rank 0."
    )


np.savez(
    RUN_ROOT / "summary.npz",
    attack_accuracy=np.array(
        [attack_acc]
    ),
    min_rank=np.array(
        [rank_curve.min()]
    ),
    final_rank=np.array(
        [rank_curve[-1]]
    ),
)


print("\nDONE")
print(
    "Results:",
    RUN_ROOT
)
