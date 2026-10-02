import random
import numpy as np
import torch

from tools.ascad_loader import load_dataset, dissemble_data_dict
from tools.model_zoo_pytorch import build_cnn_best
from triplet.triplet_pytorch import getCLSidDict, train_knn, predict_knn_prob
from tools.key_rank_new import Sbox, HW_byte, create_hw_label_mapping


DATA_PATH = "data/ASCAD.h5"
CKPT = "Output/triplet_pytorch/profiling/true_semihard_2000_normalized/ckpt/triplet_final.pt"

N_TRACES = 2000
SAMPLE_LIMIT = 300
TARGET_BYTE = 2
K = 10
SEED = 42
TRACE_MAX = 500


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    x_n, labels_n, x_limited, _, _, _ = getCLSidDict(
        data_path=DATA_PATH,
        n_traces=N_TRACES,
        sample_num_limit=SAMPLE_LIMIT,
        leakage_model="HW",
        target_byte=TARGET_BYTE
    )

    model = build_cnn_best(
        input_shape=(x_limited.shape[1], 1),
        emb_size=256,
        classification=False
    ).to(device)

    state = torch.load(CKPT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    print("checkpoint:", CKPT)

    knn = train_knn(
        model=model,
        traces=x_n,
        labels=labels_n,
        n_neighbors=K,
        leakage_model="HW"
    )

    data = load_dataset(DATA_PATH, which_one="test")

    attack_traces, _, plaintext, key = dissemble_data_dict(
        data,
        tracewindow=(0, 700),
        which_one="test"
    )

    probs = predict_knn_prob(
        model,
        knn,
        attack_traces,
        leakage_model="HW"
    )

    real_key = int(key[TARGET_BYTE])
    hw_mapping = create_hw_label_mapping()

    random.seed(SEED)
    indices = list(range(len(plaintext)))
    random.shuffle(indices)
    indices = indices[:TRACE_MAX]

    scores = np.zeros(256, dtype=np.float64)

    checkpoints = {1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 500}

    print("real key:", real_key, hex(real_key))
    print()
    print("traces | rank | real_score | best_wrong | gap | top_key")

    for i, idx in enumerate(indices, start=1):
        pt = int(plaintext[idx, TARGET_BYTE])

        for guess in range(256):
            sbox_out = Sbox[pt ^ guess]
            hw = HW_byte[sbox_out]

            p = probs[idx, hw] / len(hw_mapping[hw])
            scores[guess] += np.log(p + 1e-40)

        ranked = np.argsort(scores)[::-1]
        rank = int(np.where(ranked == real_key)[0][0])

        wrong_scores = scores.copy()
        wrong_scores[real_key] = -np.inf

        best_wrong_key = int(np.argmax(wrong_scores))
        best_wrong_score = wrong_scores[best_wrong_key]

        gap = scores[real_key] - best_wrong_score

        if i in checkpoints:
            print(
                f"{i:6d} | "
                f"{rank:4d} | "
                f"{scores[real_key]:10.3f} | "
                f"{best_wrong_score:10.3f} | "
                f"{gap:9.3f} | "
                f"{int(ranked[0]):3d} (0x{int(ranked[0]):02x})"
            )

    print()
    print("final rank:", rank)
    print("final gap:", gap)
    print("best wrong key:", best_wrong_key, hex(best_wrong_key))


if __name__ == "__main__":
    main()
