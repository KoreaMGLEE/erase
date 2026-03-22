#!/bin/bash
# Remaining experiments: per-model Phase1(LR search) → Phase2(confidence) pipeline
# Usage: bash run_remaining.sh <gpu_id>
# GPU 0: pythia-70m, bert-small, pythia-1b, pythia-410m, pythia-2.8b
# GPU 1: bert-mini, bert-medium, pythia-1.4b, bert-large, pythia-6.9b

set -e
export HF_HOME=~/hf_home

GPU=$1
BASE=/workspace/erase
SCRIPT=$BASE/scripts/train_mnli_proxy.py
OUT=$BASE/outputs/plan0

if [ -z "$GPU" ]; then
    echo "Usage: bash run_remaining.sh <gpu_id>"
    exit 1
fi

# Model-specific LR candidates
declare -A LR_CANDIDATES
# < 110M
LR_CANDIDATES[bert-mini]="3e-5 5e-5 1e-4 3e-4 1e-3"
LR_CANDIDATES[bert-small]="3e-5 5e-5 1e-4 3e-4 1e-3"
LR_CANDIDATES[bert-medium]="3e-5 5e-5 1e-4 3e-4 1e-3"
LR_CANDIDATES[pythia-70m]="3e-5 5e-5 1e-4 3e-4 1e-3"
# >= 110M
LR_CANDIDATES[bert-large]="1e-5 2e-5 3e-5 5e-5 1e-4"
LR_CANDIDATES[pythia-410m]="1e-5 2e-5 3e-5 5e-5 1e-4"
# >= 1B LoRA
LR_CANDIDATES[pythia-1b]="5e-5 1e-4 3e-4 1e-3 3e-3"
LR_CANDIDATES[pythia-1.4b]="5e-5 1e-4 3e-4 1e-3 3e-3"
LR_CANDIDATES[pythia-2.8b]="5e-5 1e-4 3e-4 1e-3 3e-3"
LR_CANDIDATES[pythia-6.9b]="5e-5 1e-4 3e-4 1e-3 3e-3"

# GPU assignments (balanced ~16.5hr each)
if [ "$GPU" = "0" ]; then
    MODELS="pythia-70m bert-small pythia-1b pythia-410m pythia-2.8b"
elif [ "$GPU" = "1" ]; then
    MODELS="bert-mini bert-medium pythia-1.4b bert-large pythia-6.9b"
else
    echo "GPU must be 0 or 1"
    exit 1
fi

run_model() {
    local model=$1
    echo ""
    echo "============================================"
    echo "[$model] Starting on GPU $GPU ($(date))"
    echo "============================================"

    # Phase 1: LR Search
    echo "--- Phase 1: LR Search ---"
    for lr in ${LR_CANDIDATES[$model]}; do
        echo "  $model lr=$lr"
        python3 $SCRIPT \
            --model $model --split split1 --lr $lr \
            --epochs 1 --batch_size 16 --seed 1 --gpu $GPU \
            --mode lr_search \
            --output_dir $OUT/lr_search/${model}_lr${lr}
    done

    # Find best LR
    BEST_LR=""
    BEST_ACC=0
    for lr in ${LR_CANDIDATES[$model]}; do
        acc=$(python3 -c "import json; d=json.load(open('$OUT/lr_search/${model}_lr${lr}/results.json')); print(d['val_acc'])")
        echo "  $model lr=$lr val_acc=$acc"
        better=$(python3 -c "print(1 if $acc > $BEST_ACC else 0)")
        if [ "$better" = "1" ]; then
            BEST_ACC=$acc
            BEST_LR=$lr
        fi
    done
    echo ">>> Best $model LR: $BEST_LR (val_acc=$BEST_ACC)"

    # Phase 2: Confidence
    echo "--- Phase 2: Confidence (best_lr=$BEST_LR) ---"
    for split in split1 split2 split3; do
        echo "  $model $split"
        python3 $SCRIPT \
            --model $model --split $split --lr $BEST_LR \
            --epochs 1 --batch_size 16 --seed 1 --gpu $GPU \
            --mode confidence \
            --output_dir $OUT/confidence/${model}_${split}
    done

    # Merge 90K confidence
    python3 -c "
import json, numpy as np, os
all_conf = {}
for s in ['split1', 'split2', 'split3']:
    path = '$OUT/confidence/${model}_' + s + '/avg_conf.json'
    with open(path) as f:
        conf = json.load(f)
    all_conf.update(conf)
confs = np.array(list(all_conf.values()))
print(f'$model 90K: mean={confs.mean():.4f}, std={confs.std():.4f}, >0.95={np.mean(confs>0.95)*100:.1f}%, <0.40={np.mean(confs<0.4)*100:.1f}%')
with open('$OUT/confidence/${model}_90k_avg_conf.json', 'w') as f:
    json.dump(all_conf, f)
"

    echo "[$model] DONE ($(date))"
    echo ""
}

echo "============================================"
echo "GPU $GPU: Starting models: $MODELS"
echo "Start time: $(date)"
echo "============================================"

for model in $MODELS; do
    run_model $model
done

echo "============================================"
echo "GPU $GPU: ALL DONE ($(date))"
echo "============================================"
