#!/bin/bash
# ARC Pythia models only (LoRA ≥1b)
# Usage: bash run_arc_pythia.sh <gpu_id>
# GPU 0: pythia-1b, pythia-2.8b
# GPU 1: pythia-1.4b, pythia-6.9b

set -e
export HF_HOME=~/hf_home

GPU=$1
BASE=/workspace/erase
SCRIPT=$BASE/scripts/train_arc_proxy.py
OUT=$BASE/outputs/plan2

if [ -z "$GPU" ]; then echo "Usage: bash run_arc_pythia.sh <gpu_id>"; exit 1; fi

declare -A LR_CANDIDATES
LR_CANDIDATES[pythia-1b]="5e-5 1e-4 3e-4 1e-3 3e-3"
LR_CANDIDATES[pythia-1.4b]="5e-5 1e-4 3e-4 1e-3 3e-3"
LR_CANDIDATES[pythia-2.8b]="5e-5 1e-4 3e-4 1e-3 3e-3"
LR_CANDIDATES[pythia-6.9b]="5e-5 1e-4 3e-4 1e-3 3e-3"

if [ "$GPU" = "0" ]; then
    MODELS="pythia-1b pythia-2.8b"
elif [ "$GPU" = "1" ]; then
    MODELS="pythia-1.4b pythia-6.9b"
else
    echo "GPU must be 0 or 1"; exit 1
fi

needs_bf16() {
    case $1 in
        pythia-2.8b|pythia-6.9b) echo "--bf16" ;;
        *) echo "" ;;
    esac
}

get_batch_size() {
    case $1 in
        pythia-6.9b) echo "4" ;;
        *) echo "8" ;;
    esac
}

run_model() {
    local model=$1
    local BF16=$(needs_bf16 $model)
    local BS=$(get_batch_size $model)

    echo ""
    echo "============================================"
    echo "[ARC] [$model] Starting on GPU $GPU ($(date))"
    echo "============================================"

    # Phase 1: LR Search
    echo "--- Phase 1: LR Search ---"
    for lr in ${LR_CANDIDATES[$model]}; do
        echo "  $model lr=$lr"
        python3 $SCRIPT --model $model --lr $lr \
            --epochs 10 --batch_size $BS --seed 1 --gpu $GPU \
            $BF16 --output_dir $OUT/lr_search_v2/${model}_lr${lr}
    done

    BEST_LR=""
    BEST_ACC=0
    for lr in ${LR_CANDIDATES[$model]}; do
        acc=$(python3 -c "import json; d=json.load(open('$OUT/lr_search_v2/${model}_lr${lr}/results.json')); print(d['best_overall_acc'])")
        challenge=$(python3 -c "import json; d=json.load(open('$OUT/lr_search_v2/${model}_lr${lr}/results.json')); print(d['best_challenge_acc'])")
        echo "  $model lr=$lr overall=$acc challenge=$challenge"
        better=$(python3 -c "print(1 if $acc > $BEST_ACC else 0)")
        if [ "$better" = "1" ]; then BEST_ACC=$acc; BEST_LR=$lr; fi
    done
    echo ">>> Best $model LR: $BEST_LR (overall_acc=$BEST_ACC)"

    # Phase 2: Confidence (3 seeds)
    echo "--- Phase 2: Confidence (best_lr=$BEST_LR, seeds=1,2,3) ---"
    for seed in 1 2 3; do
        echo "  $model seed=$seed"
        python3 $SCRIPT --model $model --lr $BEST_LR \
            --epochs 10 --batch_size $BS --seed $seed --gpu $GPU \
            $BF16 --output_dir $OUT/confidence/${model}_seed${seed}
    done

    python3 -c "
import json, numpy as np
model = '$model'
OUT = '$OUT'
for seed in [1, 2, 3]:
    d = json.load(open(f'{OUT}/confidence/{model}_seed{seed}/results.json'))
    print(f'  seed={seed}: overall={d[\"best_overall_acc\"]:.4f}, challenge={d[\"best_challenge_acc\"]:.4f}, best_epoch={d[\"best_epoch\"]}')
avg_o = np.mean([json.load(open(f'{OUT}/confidence/{model}_seed{s}/results.json'))['best_overall_acc'] for s in [1,2,3]])
avg_c = np.mean([json.load(open(f'{OUT}/confidence/{model}_seed{s}/results.json'))['best_challenge_acc'] for s in [1,2,3]])
std_o = np.std([json.load(open(f'{OUT}/confidence/{model}_seed{s}/results.json'))['best_overall_acc'] for s in [1,2,3]])
std_c = np.std([json.load(open(f'{OUT}/confidence/{model}_seed{s}/results.json'))['best_challenge_acc'] for s in [1,2,3]])
print(f'{model} avg: overall={avg_o:.4f}±{std_o:.4f}, challenge={avg_c:.4f}±{std_c:.4f}')
"
    echo "[ARC] [$model] DONE ($(date))"
}

for model in $MODELS; do
    run_model $model
done

echo "GPU $GPU: ARC Pythia ALL DONE ($(date))"
