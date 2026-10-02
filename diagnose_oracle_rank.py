import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

from tools.ascad_loader import load_dataset, dissemble_data_dict
from tools.loadData import get_labels
from tools.key_rank_new import ranking_curve


DATA_PATH = "data/ASCAD.h5"
TARGET_BYTE = 2
TRACE_NUM_MAX = 500
EPS = 1e-6


def main():
    data = load_dataset(DATA_PATH, which_one="test")

    _, _, plaintext, key = dissemble_data_dict(
        data,
        tracewindow=(0, 700),
        which_one="test"
    )

    real_key = int(key[TARGET_BYTE])

    labels = np.asarray(
        get_labels(
            plaintext,
            real_key,
            TARGET_BYTE,
            "HW"
        ),
        dtype=np.int64
    )

    probs = np.full(
        (len(labels), 9),
        EPS,
        dtype=np.float64
    )

    probs[np.arange(len(labels)), labels] = 1.0 - 8 * EPS

    print("real key:", real_key, hex(real_key))
    print("oracle acc:", np.mean(np.argmax(probs, axis=1) == labels))

    random.seed(42)
    np.random.seed(42)

    ge = ranking_curve(
        preds=probs,
        key=key,
        plaintext=plaintext,
        target_byte=TARGET_BYTE,
        rank_root=Path(
            "Output/triplet_pytorch/profiling/rank/oracle_sanity"
        ),
        leakage_model="HW",
        trace_num_max=TRACE_NUM_MAX,
        num_averaged=1
    )

    ge = np.asarray(ge)

    print("first 20 ranks:", ge[:20])
    print("min rank:", ge.min())
    print("final rank:", ge[-1])

    zero = np.where(ge == 0)[0]

    if len(zero):
        print("first rank 0:", zero[0] + 1, "traces")
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")


if __name__ == "__main__":
    main()
