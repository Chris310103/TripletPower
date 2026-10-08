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

DATA_PATH = "../Target 1/X1_K1_200k_L11.npz"
RUN_NAME = "tp_hw_N2000_ep5_tensorflow_legacy"
OUTPUT_ROOT = Path("Output/triplet_tensorflow/profiling")
RUN_ROOT = OUTPUT_ROOT / RUN_NAME
MODEL_DIR = RUN_ROOT / "feat_model"
RANK_DIR = RUN_ROOT / "ranking"

N_TRACES = 2000
ATTACK_SIZE = 10000
EPOCHS = 5
U = 300
TARGET_BYTE = 2
LEAKAGE_MODEL = "HW"
TRACE_START = 1800
TRACE_END = 2800
N_NEIGHBORS = 10
TRACE_NUM_MAX = 500
NUM_AVERAGED = 5
SEED = 42

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