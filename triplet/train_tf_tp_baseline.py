import os
os.environ["TF_USE_LEGACY_KERAS"]="1"

from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

import triplet.triplet as legacy_triplet
import triplet.test as legacy_test

def tf21_identity_loss(y_true, y_pred):
    return K.mean(y_pred)

def tf21_cosine_triplet_loss(X):
    positive_sim, negative_sim=X
    return K.maximum(0.0, negative_sim-positive_sim+float(legacy_triplet.alpha_value))

legacy_triplet.identity_loss=tf21_identity_loss
legacy_triplet.cosine_triplet_loss=tf21_cosine_triplet_loss

DATA_PATH="../Target 1/X1_K1_200k_L11.npz"
RUN_NAME="tp_tf_original_baseline_N500_ep100"
OUTPUT_ROOT=Path("Output/triplet_tensorflow/original_baseline")
RUN_ROOT=OUTPUT_ROOT/RUN_NAME
MODEL_DIR=RUN_ROOT/"feat_model"
KNN_MODEL_DIR=RUN_ROOT/"knn_model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
KNN_MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_NUM=500
TEST_NUM=5000
EPOCHS=100
U=300
TARGET_BYTE=2
LEAKAGE_MODEL="HW"
ATTACK_WINDOW="1800_2800"
PREPROCESS=""

print("="*80)
print("ORIGINAL TENSORFLOW TRIPLETPower BASELINE")
print("="*80)
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("Dataset:", DATA_PATH)
print("Train traces:", TRAIN_NUM)
print("Test traces:", TEST_NUM)
print("Epochs:", EPOCHS)
print("U:", U)
print("Attack window:", ATTACK_WINDOW)

train_opts=SimpleNamespace(input=DATA_PATH, output=str(RUN_ROOT), epochs=EPOCHS, nsamples=U, target_byte=TARGET_BYTE, leakage_model=LEAKAGE_MODEL, attack_window=ATTACK_WINDOW, eval_type="", trace_num=TRAIN_NUM, preprocess=PREPROCESS, pre_train=False)

train_x, train_y, plaintext_train, train_key=legacy_triplet.load_data(train_opts)

print("Training X shape:", train_x.shape)
print("Training y shape:", train_y.shape)
print("Training key:", train_key)
print("Training HW counts:", np.bincount(np.asarray(train_y, dtype=np.int64), minlength=9))

model_dir_string=str(MODEL_DIR)+"/"

start=time.time()
feat_model=legacy_triplet.train(train_opts, train_x, train_y, model_dir_string)
print("Training seconds:", time.time()-start)

train_x_3d=train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
train_x_feat=feat_model.predict(train_x_3d)

print("Training feature shape:", train_x_feat.shape)

needed_classes=set(range(9))

if needed_classes != set(train_y):
    missing_class=needed_classes-set(train_y)
    missing_data=[]
    missing_label=[]

    print("Missing classes:", missing_class)

    for label in missing_class:
        missing_label.append(label)
        missing_data.append([0]*train_x_feat.shape[1])

    train_x_feat=np.concatenate((train_x_feat, missing_data), axis=0)
    train_y=np.concatenate((train_y, missing_label), axis=0)

legacy_triplet.train_one_knn(train_opts, str(KNN_MODEL_DIR), train_x_feat, train_y)

test_opts=SimpleNamespace(input=DATA_PATH, root_dir=str(RUN_ROOT), cross_dev=False, attack_window=ATTACK_WINDOW, preprocess=PREPROCESS, test_num=TEST_NUM, target_byte=TARGET_BYTE, leakage_model=LEAKAGE_MODEL, eval_type="")

print("="*80)
print("ORIGINAL TEST PIPELINE")
print("="*80)

x_test_feat, plaintext_test, test_key=legacy_test.get_x_feat(test_opts)

print("Test feature shape:", x_test_feat.shape)
print("Test plaintext shape:", plaintext_test.shape)
print("Test key:", test_key)

legacy_test.compute_knn_ranks(test_opts, x_test_feat, plaintext_test, test_key)

print("="*80)
print("DONE")
print("Results:", RUN_ROOT)
print("="*80)