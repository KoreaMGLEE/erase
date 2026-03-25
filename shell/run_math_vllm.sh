#!/bin/bash
# Plan 4: MATH LR Search with vLLM eval
# Usage: bash run_math_vllm.sh <gpu_id>
# GPU 0: pythia-1b, pythia-6.9b
# GPU 1: qwen2.5-1.5b, pythia-2.8b
set -e
export HF_HOME=~/hf_home
export CUDA_VISIBLE_DEVICES=$1

GPU=$1
BASE=/workspace/erase
SCRIPT=$BASE/scripts/train_math_vllm.py
OUT=$BASE/outputs/plan4/lr_search

if [ -z "$GPU" ]; then echo "Usage: bash run_math_vllm.sh <gpu_id>"; exit 1; fi

LRS="5e-5 1e-4 3e-4 1e-3 3e-3"

needs_bf16() {
    case $1 in
        pythia-2.8b|pythia-6.9b) echo "--bf16" ;;
        *) echo "" ;;
    esac
}

get_batch_size() {
    case $1 in
        pythia-6.9b) echo "2" ;;
        *) echo "4" ;;
    esac
}

if [ "$GPU" = "0" ]; then
    MODELS="pythia-1b pythia-6.9b"
elif [ "$GPU" = "1" ]; then
    MODELS="qwen2.5-1.5b pythia-2.8b"
else
    echo "GPU must be 0 or 1"; exit 1
fi

for model in $MODELS; do
    BF16=$(needs_bf16 $model)
    BS=$(get_batch_size $model)

    echo ""
    echo "============================================"
    echo "[MATH] [$model] Starting on GPU $GPU ($(date))"
    echo "============================================"

    for lr in $LRS; do
        outdir=$OUT/${model}_lr${lr}
        if [ -f "$outdir/results.json" ]; then
            echo "  SKIP: $outdir exists"
            continue
        fi
        echo "  $model lr=$lr"
        # Use gpu 0 internally since CUDA_VISIBLE_DEVICES remaps
        python3 $SCRIPT \
            --model $model --lr $lr \
            --epochs 3 --batch_size $BS --grad_accum 4 \
            --seed 1 --gpu 0 --eval_samples 500 \
            --use_vllm \
            $BF16 --output_dir $outdir
    done

    BEST_LR=""
    BEST_ACC=0
    for lr in $LRS; do
        rfile=$OUT/${model}_lr${lr}/results.json
        if [ ! -f "$rfile" ]; then continue; fi
        acc=$(python3 -c "import json; print(json.load(open('$rfile'))['best_accuracy'])")
        echo "  $model lr=$lr acc=$acc"
        better=$(python3 -c "print(1 if float('$acc') > $BEST_ACC else 0)")
        if [ "$better" = "1" ]; then BEST_ACC=$acc; BEST_LR=$lr; fi
    done
    echo ">>> Best $model LR: $BEST_LR (acc=$BEST_ACC)"
    echo "[MATH] [$model] DONE ($(date))"
done

echo "GPU $GPU: MATH ALL DONE ($(date))"
