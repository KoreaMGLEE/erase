#!/bin/bash
# ARC LR search + confidence for all models
# Usage: bash run_arc_all.sh <role>
set -e

ROLE=$1
SCRIPT=/workspace/erase/scripts/train_arc_all_models.py
OUT=/workspace/erase/outputs/plan2_v2

declare -A LR_MAP
# Small BERT/Pythia
LR_MAP[bert-mini]="3e-5 5e-5 1e-4 3e-4 1e-3"
LR_MAP[bert-small]="3e-5 5e-5 1e-4 3e-4 1e-3"
LR_MAP[bert-medium]="3e-5 5e-5 1e-4 3e-4 1e-3"
LR_MAP[pythia-14m]="3e-5 5e-5 1e-4 3e-4 1e-3"
LR_MAP[pythia-31m]="3e-5 5e-5 1e-4 3e-4 1e-3"
LR_MAP[pythia-70m]="3e-5 5e-5 1e-4 3e-4 1e-3"
# Medium
LR_MAP[bert-base]="1e-5 2e-5 3e-5 5e-5 1e-4"
LR_MAP[bert-large]="1e-5 2e-5 3e-5 5e-5 1e-4"
LR_MAP[pythia-160m]="1e-5 2e-5 3e-5 5e-5 1e-4"
LR_MAP[pythia-410m]="1e-5 2e-5 3e-5 5e-5 1e-4"
# T5 v1.1
LR_MAP[t5-v1_1-small]="3e-5 5e-5 1e-4 3e-4"
LR_MAP[t5-v1_1-base]="5e-5 1e-4 2e-4 3e-4"
LR_MAP[t5-v1_1-large]="3e-5 5e-5 1e-4"
LR_MAP[t5-v1_1-xl]="5e-5 1e-4 3e-4"
# LoRA Pythia
LR_MAP[pythia-1b]="5e-5 1e-4 3e-4 1e-3"
LR_MAP[pythia-1.4b]="5e-5 1e-4 3e-4 1e-3"
LR_MAP[pythia-2.8b]="5e-5 1e-4 3e-4 1e-3"
LR_MAP[pythia-6.9b]="5e-5 1e-4 3e-4 1e-3"

needs_bf16() {
    case $1 in
        t5-v1_1-xl|pythia-2.8b|pythia-6.9b) echo "--bf16" ;;
        *) echo "" ;;
    esac
}

get_bs() {
    case $1 in
        pythia-6.9b) echo "2" ;;
        t5-v1_1-xl|pythia-1.4b|pythia-2.8b) echo "4" ;;
        *) echo "8" ;;
    esac
}

run_model() {
    local model=$1 gpu=$2
    local bf16=$(needs_bf16 $model)
    local bs=$(get_bs $model)
    local lrs=${LR_MAP[$model]}

    echo ""
    echo "============================================"
    echo "[ARC] [$model] Starting (GPU $gpu) $(date)"
    echo "============================================"

    # LR Search
    echo "--- LR Search ---"
    for lr in $lrs; do
        local outdir=$OUT/lr_search/${model}_lr${lr}
        mkdir -p $outdir
        if [ -f "$outdir/results.json" ]; then echo "SKIP $lr"; continue; fi
        echo "  $model lr=$lr"
        python3 $SCRIPT --model $model --lr $lr --epochs 5 --batch_size $bs --seed 1 --gpu $gpu $bf16 --output_dir $outdir
    done

    # Find best LR
    local best_lr="" best_acc=0
    for lr in $lrs; do
        local rfile=$OUT/lr_search/${model}_lr${lr}/results.json
        if [ ! -f "$rfile" ]; then continue; fi
        local acc=$(python3 -c "import json; print(json.load(open('$rfile'))['best_overall_acc'])")
        echo "  $model lr=$lr acc=$acc"
        local better=$(python3 -c "print(1 if float('$acc') > $best_acc else 0)")
        if [ "$better" = "1" ]; then best_acc=$acc; best_lr=$lr; fi
    done
    echo ">>> Best $model: $best_lr ($best_acc)"

    # Confidence: 3 seeds
    echo "--- Confidence (seeds 1,2,3) ---"
    for seed in 1 2 3; do
        local outdir=$OUT/confidence/${model}_seed${seed}
        mkdir -p $outdir
        if [ -f "$outdir/results.json" ]; then echo "SKIP seed=$seed"; continue; fi
        echo "  $model seed=$seed"
        python3 $SCRIPT --model $model --lr $best_lr --epochs 5 --batch_size $bs --seed $seed --gpu $gpu $bf16 --output_dir $outdir
    done

    echo "[ARC] [$model] DONE $(date)"
}

case $ROLE in
    local0)
        # pythia-6.9b, t5-base, pythia-31m
        for m in pythia-6.9b t5-v1_1-base pythia-31m; do
            run_model $m 0
        done
        ;;
    local1)
        # t5-xl, bert-large, pythia-160m, pythia-70m
        for m in t5-v1_1-xl bert-large pythia-160m pythia-70m; do
            run_model $m 1
        done
        ;;
    remote0)
        # pythia-2.8b, pythia-1b, bert-base, bert-medium, bert-small
        for m in pythia-2.8b pythia-1b bert-base bert-medium bert-small; do
            run_model $m 0
        done
        ;;
    remote1)
        # t5-large, pythia-1.4b, pythia-410m, t5-small, bert-mini, pythia-14m
        for m in t5-v1_1-large pythia-1.4b pythia-410m t5-v1_1-small bert-mini pythia-14m; do
            run_model $m 1
        done
        ;;
esac

echo "============================================"
echo "$ROLE ALL DONE $(date)"
echo "============================================"
