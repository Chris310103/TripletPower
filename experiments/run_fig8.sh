#!/bin/bash
set -e

DATA="$HOME/TripletPower/data/ASCAD.h5"

EPOCHS="${1:-100}"
BATCH_SIZE=100
TARGET_BYTE=2
LEAKAGE_MODEL="HW"
SEED=42
LR=1e-5

U=300
ALPHA=0.5
N_NEIGHBORS=10
NUM_AVERAGED=5

echo "Fig.8 experiment epochs: $EPOCHS"

for N in 250 2000 4000
do
    echo
    echo "============================================================"
    echo "Starting Fig.8 experiment for N=${N}"
    echo "============================================================"

    if [ "$N" -eq 250 ]; then
        TRACE_NUM_MAX=3000
    elif [ "$N" -eq 2000 ]; then
        TRACE_NUM_MAX=150
    elif [ "$N" -eq 4000 ]; then
        TRACE_NUM_MAX=50
    else
        echo "Unsupported N=${N}"
        exit 1
    fi

    RANK_NAME="fig8_N${N}"
    INDICES_PATH="Output/cnn_pytorch/N_${N}/selected_indices.npy"

    echo
    echo "------------------------------------------------------------"
    echo "[1/3] Training CNN for N=${N}"
    echo "------------------------------------------------------------"

    python -m cnn.train_cnn_torch \
        --data_path "$DATA" \
        --n_traces "$N" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --target_byte "$TARGET_BYTE" \
        --leakage_model "$LEAKAGE_MODEL" \
        --learning_rate "$LR" \
        --seed "$SEED"

    echo
    echo "------------------------------------------------------------"
    echo "[2/3] Testing CNN / computing key rank for N=${N}"
    echo "------------------------------------------------------------"

    python -m cnn.test_cnn_torch \
        --data_path "$DATA" \
        --n_traces "$N" \
        --target_byte "$TARGET_BYTE" \
        --leakage_model "$LEAKAGE_MODEL" \
        --trace_num_max "$TRACE_NUM_MAX" \
        --num_averaged "$NUM_AVERAGED" \
        --seed "$SEED"

    echo
    echo "------------------------------------------------------------"
    echo "[3/3] Training TripletPower / computing key rank for N=${N}"
    echo "------------------------------------------------------------"

    python -m triplet.train_torch \
        --data_path "$DATA" \
        --rank_name "$RANK_NAME" \
        --n_traces "$N" \
        --sample_num_limit "$U" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --target_byte "$TARGET_BYTE" \
        --leakage_model "$LEAKAGE_MODEL" \
        --learning_rate "$LR" \
        --alpha_value "$ALPHA" \
        --n_neighbors "$N_NEIGHBORS" \
        --seed "$SEED" \
        --selected_indices_path "$INDICES_PATH" \
        --trace_num_max "$TRACE_NUM_MAX" \
        --num_averaged "$NUM_AVERAGED"

    echo
    echo "============================================================"
    echo "Finished N=${N}"
    echo "============================================================"
done

echo
echo "============================================================"
echo "All training finished. Plotting Fig.8..."
echo "============================================================"

python experiments/plot_fig8.py \
    --n_values 250 2000 4000 \
    --cnn_rank_paths \
        "Output/cnn_pytorch/N_250/rank/ranking_raw_data.npz" \
        "Output/cnn_pytorch/N_2000/rank/ranking_raw_data.npz" \
        "Output/cnn_pytorch/N_4000/rank/ranking_raw_data.npz" \
    --triplet_rank_paths \
        "Output/triplet_pytorch/profiling/rank/fig8_N250/ranking_raw_data.npz" \
        "Output/triplet_pytorch/profiling/rank/fig8_N2000/ranking_raw_data.npz" \
        "Output/triplet_pytorch/profiling/rank/fig8_N4000/ranking_raw_data.npz" \
    --output "Output/fig8/fig8_full.png"

echo
echo "============================================================"
echo "Fig.8 experiment finished."
echo "Output: Output/fig8/fig8_full.png"
echo "============================================================"