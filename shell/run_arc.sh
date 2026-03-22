#!/bin/bash
# ARC Plan 2 - Per-model LR search
# Usage: bash run_arc.sh <gpu_id>
# GPU 0: bert-base, pythia-410m
# GPU 1: bert-large, pythia-1b

set -e
export HF_HOME=~/hf_home

GPU=$1
BASE=/workspace/erase
SCRIPT=$BASE/scripts/train_arc_proxy.py
OUT=$BASE/outputs/plan2/lr_search

if [ -z "$GPU" ]; then
    echo "Usage: bash run_arc.sh <gpu_id>"
    exit 1
fi

declare -A LR_CANDIDATES
LR_CANDIDATES[bert-base]="1e-5 2e-5 3e-5 5e-5 1e-4"
LR_CANDIDATES[bert-large]="1e-5 2e-5 3e-5 5e-5 1e-4"
LR_CANDIDATES[pythia-410m]="1e-5 2e-5 3e-5 5e-5 1e-4"
LR_CANDIDATES[pythia-1b]="5e-5 1e-4 3e-4 1e-3 3e-3"

# GPU assignments
if [ "$GPU" = "0" ]; then
    MODELS="bert-base pythia-410m"
elif [ "$GPU" = "1" ]; then
    MODELS="bert-large pythia-1b"
else
    echo "GPU must be 0 or 1"
    exit 1
fi

# BF16 flag for large models
needs_bf16() {
    case $1 in
        pythia-1b|pythia-1.4b|pythia-2.8b|pythia-6.9b) echo "--bf16" ;;
        *) echo "" ;;
    esac
}

for model in $MODELS; do
    echo ""
    echo "============================================"
    echo "[ARC] [$model] Starting on GPU $GPU ($(date))"
    echo "============================================"

    BF16_FLAG=$(needs_bf16 $model)

    for lr in ${LR_CANDIDATES[$model]}; do
        echo "  $model lr=$lr"
        python3 $SCRIPT \
            --model $model --lr $lr \
            --epochs 10 --batch_size 8 --seed 1 --gpu $GPU \
            $BF16_FLAG \
            --output_dir $OUT/${model}_lr${lr}
    done

    # Find best LR (by challenge_acc)
    BEST_LR=""
    BEST_ACC=0
    for lr in ${LR_CANDIDATES[$model]}; do
        acc=$(python3 -c "import json; d=json.load(open('$OUT/${model}_lr${lr}/results.json')); print(d['best_challenge_acc'])")
        overall=$(python3 -c "import json; d=json.load(open('$OUT/${model}_lr${lr}/results.json')); print(d['best_overall_acc'])")
        epoch=$(python3 -c "import json; d=json.load(open('$OUT/${model}_lr${lr}/results.json')); print(d['best_epoch'])")
        echo "  $model lr=$lr challenge_acc=$acc overall_acc=$overall (best_epoch=$epoch)"
        better=$(python3 -c "print(1 if $acc > $BEST_ACC else 0)")
        if [ "$better" = "1" ]; then
            BEST_ACC=$acc
            BEST_LR=$lr
        fi
    done
    echo ">>> Best $model LR: $BEST_LR (challenge_acc=$BEST_ACC)"
    echo "[ARC] [$model] DONE ($(date))"
done

echo ""
echo "============================================"
echo "GPU $GPU: ARC ALL DONE ($(date))"
echo "============================================"
