import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path
from types import SimpleNamespace
import random
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import triplet.triplet as legacy_triplet
from tools.loadData import get_labels
from tools.key_rank_new import ranking_curve


def tf21_identity_loss(y_true, y_pred):
    return K.mean(y_pred)


def tf21_cosine_triplet_loss(X):
    positive_sim, negative_sim = X
    return K.maximum(0.0, negative_sim - positive_sim + float(legacy_triplet.alpha_value))

legacy_triplet.identity_loss = tf21_identity_loss
legacy_triplet.cosine_triplet_loss = tf21_cosine_triplet_loss

DATA_PATH="../Target 1/X1_K1_200k_L11.npz"
RUN_NAME="tp_hw_N500_ep100_tensorflow_legacy_1800_2800"
OUTPUT_ROOT=Path("Output/triplet_tensorflow/profiling")

N_TRACES=500
ATTACK_SIZE=10000
EPOCHS=100
U=300

TARGET_BYTE=2
LEAKAGE_MODEL="HW"

TRACE_START=1800
TRACE_END=2800

N_NEIGHBORS=10

TRACE_NUM_MAX=5000
NUM_AVERAGED=5

SEED=42

RUN_ROOT = OUTPUT_ROOT / RUN_NAME
MODEL_DIR = RUN_ROOT / "feat_model"
RANK_DIR = RUN_ROOT / "ranking"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RANK_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("RMSprop:", legacy_triplet.RMSprop)

raw = np.load(DATA_PATH)
traces = np.asarray(raw["power_trace"])
plaintext = np.asarray(raw["plain_text"])
key = np.asarray(raw["key"])

train_pool_size = len(traces) - ATTACK_SIZE
if train_pool_size <= 0:
    raise ValueError("ATTACK_SIZE must be smaller than dataset size")

selected_indices = np.random.choice(train_pool_size, size=N_TRACES, replace=False)

x_train = traces[selected_indices, TRACE_START:TRACE_END]
plaintext_train = plaintext[selected_indices]
train_y = np.asarray(get_labels(plaintext_train, int(key[TARGET_BYTE]), TARGET_BYTE, LEAKAGE_MODEL), dtype=np.int64)

x_attack = traces[train_pool_size:, TRACE_START:TRACE_END]
plaintext_attack = plaintext[train_pool_size:]
attack_expected = np.asarray(get_labels(plaintext_attack, int(key[TARGET_BYTE]), TARGET_BYTE, LEAKAGE_MODEL), dtype=np.int64)

print("Dataset:", DATA_PATH)
print("Train pool:", train_pool_size)
print("Profiling traces:", x_train.shape)
print("Attack traces:", x_attack.shape)
print("Key:", key)
print("Target key byte:", int(key[TARGET_BYTE]))
print("HW counts:", np.bincount(train_y, minlength=9))

opts = SimpleNamespace(epochs=EPOCHS, nsamples=U, target_byte=TARGET_BYTE, leakage_model=LEAKAGE_MODEL)
model_dir_string = str(MODEL_DIR) + "/"

start = time.time()
final_model = legacy_triplet.train(opts, x_train, train_y, model_dir_string)
print("Training seconds:", time.time() - start)

# ============================================================
# Profiling embeddings + kNN
# ============================================================
x_train_3d=x_train.reshape((len(x_train), x_train.shape[1], 1))
train_embeddings=final_model.predict(x_train_3d, batch_size=256, verbose=1)

train_norms=np.linalg.norm(train_embeddings, axis=1)
print("Training embedding shape:", train_embeddings.shape)
print("Training embedding norm mean:", float(train_norms.mean()))
print("Training zero embedding fraction:", float(np.mean(train_norms < 1e-8)))

knn=KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights="distance", metric="cosine", algorithm="brute")
knn.fit(train_embeddings, train_y)

print("KNN classes:", knn.classes_)

# ============================================================
# Attack embeddings
# ============================================================
x_attack_3d=x_attack.reshape((len(x_attack), x_attack.shape[1], 1))
attack_embeddings=final_model.predict(x_attack_3d, batch_size=256, verbose=1)

attack_pred=knn.predict(attack_embeddings)
attack_acc=accuracy_score(attack_expected, attack_pred)

print("TensorFlow TripletPower attack classification accuracy:", attack_acc)

# ============================================================
# kNN probabilities
# ============================================================
needed_classes=set(range(9))
missing_classes=needed_classes-set(train_y.astype(int))

if missing_classes:
    print("Missing HW classes:", sorted(missing_classes))
    missing_embeddings=np.zeros((len(missing_classes), train_embeddings.shape[1]), dtype=train_embeddings.dtype)
    missing_labels=np.asarray(sorted(missing_classes), dtype=train_y.dtype)
    train_embeddings=np.concatenate((train_embeddings, missing_embeddings), axis=0)
    train_y=np.concatenate((train_y, missing_labels), axis=0)

knn=KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights="distance", metric="cosine", algorithm="brute")
knn.fit(train_embeddings, train_y)

print("KNN classes:", knn.classes_)
attack_pred=knn.predict(attack_embeddings)
attack_acc=accuracy_score(attack_expected, attack_pred)
print("TensorFlow TripletPower attack classification accuracy:", attack_acc)

# ============================================================
# Key rank
# ============================================================
attack_probs=knn.predict_proba(attack_embeddings)

print("Attack probability shape:", attack_probs.shape)

rank_curve=ranking_curve(preds=attack_probs, key=key, plaintext=plaintext_attack, target_byte=TARGET_BYTE, rank_root=str(RANK_DIR), leakage_model=LEAKAGE_MODEL, trace_num_max=TRACE_NUM_MAX, num_averaged=NUM_AVERAGED)

rank_curve=np.asarray(rank_curve)

print("Minimum rank:", float(rank_curve.min()))
print("Final rank:", float(rank_curve[-1]))

zero_idx=np.where(rank_curve == 0)[0]

if len(zero_idx) > 0:
    print("First rank-0 trace:", int(zero_idx[0] + 1))
else:
    print("Never reached rank 0.")

np.savez(RUN_ROOT / "summary.npz", attack_accuracy=np.array([attack_acc]), min_rank=np.array([rank_curve.min()]), final_rank=np.array([rank_curve[-1]]))

print("TensorFlow TripletPower pipeline finished.")
print("Results:", RUN_ROOT)