from pathlib import Path
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from tools.loadData import get_labels
from tools.model_zoo_pytorch import build_cnn_best
from tools.key_rank_new import ranking_curve
from triplet.triplet_pytorch import limit_per_class, build_positive_pairs, train_tripletpower, train_knn, predict_knn_prob

DATA_PATH="../Target 1/X1_K1_200k_L11.npz"
RUN_NAME="tp_pytorch_sanity_N500_ep100_first5000"

OUTPUT_ROOT=Path("Output/triplet_pytorch/sanity")
RUN_ROOT=OUTPUT_ROOT/RUN_NAME
CKPT_PATH=RUN_ROOT/"ckpt"/"triplet_best.pt"
RANK_ROOT=RUN_ROOT/"ranking"

N_TRACES=500
TEST_NUM=5000
EPOCHS=100
U=300
TARGET_BYTE=2
LEAKAGE_MODEL="HW"
TRACE_START=1800
TRACE_END=2800
BATCH_SIZE=100
LEARNING_RATE=1e-5
ALPHA_VALUE=0.5
N_NEIGHBORS=10
TRACE_NUM_MAX=5000
NUM_AVERAGED=100
NEGATIVE_MODE="tf_legacy"
SEED=42

CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
RANK_ROOT.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    device=torch.device("cuda")
elif torch.backends.mps.is_available():
    device=torch.device("mps")
else:
    device=torch.device("cpu")

print("="*80)
print("PYTORCH FIRST500 -> FIRST5000 SANITY BASELINE")
print("="*80)
print("Device:", device)
print("Dataset:", DATA_PATH)

raw=np.load(DATA_PATH)
traces=np.asarray(raw["power_trace"])
plaintext=np.asarray(raw["plain_text"])
key=np.asarray(raw["key"])

x_train=traces[:N_TRACES, TRACE_START:TRACE_END]
plaintext_train=plaintext[:N_TRACES]
train_y=np.asarray(get_labels(plaintext_train, int(key[TARGET_BYTE]), TARGET_BYTE, LEAKAGE_MODEL), dtype=np.int64)

x_attack=traces[:TEST_NUM, TRACE_START:TRACE_END]
plaintext_attack=plaintext[:TEST_NUM]
attack_expected=np.asarray(get_labels(plaintext_attack, int(key[TARGET_BYTE]), TARGET_BYTE, LEAKAGE_MODEL), dtype=np.int64)

print("Training traces:", x_train.shape)
print("Attack traces:", x_attack.shape)
print("Key:", key)
print("Target key byte:", int(key[TARGET_BYTE]))
print("HW counts:", np.bincount(train_y, minlength=9))

x_limited, labels_limited, label_2_id, id_2_label, selected_u_indices=limit_per_class(x_train, train_y, U)

print("Triplet subset:", x_limited.shape)
print("Triplet HW counts:", np.bincount(labels_limited.astype(np.int64), minlength=9))

a_ids, p_ids=build_positive_pairs(sorted(label_2_id.keys()), label_2_id)

print("Number of positive pairs:", len(a_ids))

model=build_cnn_best(input_shape=(x_limited.shape[1], 1), emb_size=256, classification=False)
model=model.to(device)

model, loss_log=train_tripletpower(model=model, all_traces=x_limited, a_ids=a_ids, p_ids=p_ids, id_2_label=id_2_label, device=device, ckpt_path=CKPT_PATH, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, alpha_value=ALPHA_VALUE, negative_mode=NEGATIVE_MODE, legacy_label_2_id=label_2_id)

print("="*80)
print("TRAIN kNN")
print("="*80)

classifier=train_knn(model=model, traces=x_train, labels=train_y, n_neighbors=N_NEIGHBORS, leakage_model=LEAKAGE_MODEL)

train_probabilities=predict_knn_prob(model, classifier, x_train, leakage_model=LEAKAGE_MODEL)
train_pred=np.argmax(train_probabilities, axis=1)
train_acc=accuracy_score(train_y, train_pred)

print("PyTorch kNN training accuracy:", train_acc)

print("="*80)
print("ATTACK")
print("="*80)

attack_probabilities=predict_knn_prob(model, classifier, x_attack, leakage_model=LEAKAGE_MODEL)
attack_pred=np.argmax(attack_probabilities, axis=1)
attack_acc=accuracy_score(attack_expected, attack_pred)

print("PyTorch attack classification accuracy:", attack_acc)
print("Attack probability shape:", attack_probabilities.shape)

random.seed(SEED)

rank_curve=ranking_curve(preds=attack_probabilities, key=key, plaintext=plaintext_attack, target_byte=TARGET_BYTE, rank_root=RANK_ROOT, leakage_model=LEAKAGE_MODEL, trace_num_max=TRACE_NUM_MAX, num_averaged=NUM_AVERAGED)

rank_curve=np.asarray(rank_curve)

print("Minimum rank:", float(rank_curve.min()))
print("Final rank:", float(rank_curve[-1]))

zero_idx=np.where(rank_curve == 0)[0]

if len(zero_idx) > 0:
    print("First rank-0 trace:", int(zero_idx[0]+1))
else:
    print("Never reached rank 0.")

np.savez(RUN_ROOT/"summary.npz", attack_accuracy=np.asarray([attack_acc]), train_accuracy=np.asarray([train_acc]), min_rank=np.asarray([rank_curve.min()]), final_rank=np.asarray([rank_curve[-1]]))

print("="*80)
print("DONE")
print("Results:", RUN_ROOT)
print("="*80)