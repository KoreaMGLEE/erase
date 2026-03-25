#!/bin/bash
# Plan 005: MNLI additional models - LR search + Confidence
# Phase 1: Small models on separate GPUs, then large models on 2 GPUs
set -e
export HF_HOME=~/hf_home

BASE=/workspace/erase
MNLI_SCRIPT=$BASE/scripts/train_mnli_proxy.py
T5_SCRIPT=$BASE/scripts/train_mnli_t5_proxy.py
OUT=$BASE/outputs/plan5

mkdir -p $OUT/{lr_search,confidence,logs}

# ============================================================
# Helper: run one model through LR search + confidence
# ============================================================
run_model() {
    local model=$1
    local script=$2
    local gpu=$3
    local bs=$4
    local bf16_flag=$5
    shift 5
    local lrs=("$@")

    echo ""
    echo "============================================"
    echo "[$model] Starting on GPU $gpu ($(date))"
    echo "============================================"

    # Phase 1: LR Search
    echo "--- Phase 1: LR Search ---"
    for lr in "${lrs[@]}"; do
        local outdir=$OUT/lr_search/${model}_lr${lr}
        if [ -f "$outdir/results.json" ]; then
            echo "  SKIP: $outdir exists"
            continue
        fi
        echo "  $model lr=$lr"
        python3 $script \
            --model $model --split split1 --lr $lr \
            --epochs 1 --batch_size $bs --seed 1 --gpu $gpu \
            --mode lr_search \
            $bf16_flag \
            --output_dir $outdir
    done

    # Find best LR
    BEST_LR=""
    BEST_ACC=0
    for lr in "${lrs[@]}"; do
        local rfile=$OUT/lr_search/${model}_lr${lr}/results.json
        if [ ! -f "$rfile" ]; then continue; fi
        acc=$(python3 -c "import json; print(json.load(open('$rfile'))['val_acc'])")
        echo "  $model lr=$lr val_acc=$acc"
        better=$(python3 -c "print(1 if float('$acc') > $BEST_ACC else 0)")
        if [ "$better" = "1" ]; then BEST_ACC=$acc; BEST_LR=$lr; fi
    done
    echo ">>> Best $model LR: $BEST_LR (val_acc=$BEST_ACC)"

    # Phase 2: Confidence
    echo "--- Phase 2: Confidence (best_lr=$BEST_LR) ---"
    for split in split1 split2 split3; do
        local outdir=$OUT/confidence/${model}_${split}
        if [ -f "$outdir/avg_conf.json" ]; then
            echo "  SKIP: $outdir exists"
            continue
        fi
        echo "  $model $split"
        python3 $script \
            --model $model --split $split --lr $BEST_LR \
            --epochs 1 --batch_size $bs --seed 1 --gpu $gpu \
            --mode confidence \
            $bf16_flag \
            --output_dir $outdir
    done

    # Merge 90K confidence
    python3 -c "
import json, numpy as np, os
all_conf = {}
for s in ['split1', 'split2', 'split3']:
    path = '$OUT/confidence/${model}_' + s + '/avg_conf.json'
    if os.path.exists(path):
        with open(path) as f:
            all_conf.update(json.load(f))
if all_conf:
    confs = np.array(list(all_conf.values()))
    print(f'$model 90K: mean={confs.mean():.4f}, std={confs.std():.4f}, >0.95={np.mean(confs>0.95)*100:.1f}%, <0.40={np.mean(confs<0.4)*100:.1f}%')
    with open('$OUT/confidence/${model}_90k_avg_conf.json', 'w') as f:
        json.dump(all_conf, f)
else:
    print('$model: no confidence data')
"
    echo "[$model] DONE ($(date))"
}

echo "============================================"
echo "Plan 005: MNLI Additional Models"
echo "Start: $(date)"
echo "============================================"

# ============================================================
# Stage 1: Small Pythia models (GPU 0) + T5 small/base (GPU 1)
# ============================================================
echo ""
echo "=== Stage 1: Small models (parallel on 2 GPUs) ==="

# GPU 0: pythia-14m, pythia-31m
(
    run_model pythia-14m $MNLI_SCRIPT 0 16 "" 3e-5 5e-5 1e-4 3e-4 1e-3
    run_model pythia-31m $MNLI_SCRIPT 0 16 "" 3e-5 5e-5 1e-4 3e-4 1e-3
    echo "[GPU 0 Stage 1] DONE ($(date))"
) &
PID_GPU0=$!

# GPU 1: t5-v1_1-small, t5-v1_1-base
(
    run_model t5-v1_1-small $T5_SCRIPT 1 16 "" 3e-5 5e-5 1e-4 3e-4 1e-3
    run_model t5-v1_1-base $T5_SCRIPT 1 16 "" 1e-5 3e-5 5e-5 1e-4 3e-4
    echo "[GPU 1 Stage 1] DONE ($(date))"
) &
PID_GPU1=$!

wait $PID_GPU0
wait $PID_GPU1
echo "=== Stage 1 COMPLETE ==="

# ============================================================
# Stage 2: T5 large (GPU 0) + T5 xl (GPU 1, bf16 LoRA)
# ============================================================
echo ""
echo "=== Stage 2: Medium models (parallel on 2 GPUs) ==="

(
    run_model t5-v1_1-large $T5_SCRIPT 0 16 "" 1e-5 2e-5 3e-5 5e-5 1e-4
    echo "[GPU 0 Stage 2] DONE ($(date))"
) &
PID_GPU0=$!

(
    run_model t5-v1_1-xl $T5_SCRIPT 1 4 "--bf16" 5e-5 1e-4 3e-4 1e-3 3e-3
    echo "[GPU 1 Stage 2] DONE ($(date))"
) &
PID_GPU1=$!

wait $PID_GPU0
wait $PID_GPU1
echo "=== Stage 2 COMPLETE ==="

# ============================================================
# Stage 3: Large models (2 GPU each, sequential)
# ============================================================
echo ""
echo "=== Stage 3: Large models (2 GPU, sequential) ==="

# pythia-12b: LoRA + bf16, try single GPU first with grad_ckpt
echo "--- pythia-12b ---"
run_model pythia-12b $MNLI_SCRIPT 0 4 "--bf16" 5e-5 1e-4 3e-4 1e-3 3e-3

# t5-v1_1-xxl: LoRA + bf16
echo "--- t5-v1_1-xxl ---"
run_model t5-v1_1-xxl $T5_SCRIPT 0 2 "--bf16" 5e-5 1e-4 3e-4 1e-3 3e-3

echo ""
echo "============================================"
echo "Plan 005: ALL DONE ($(date))"
echo "============================================"
