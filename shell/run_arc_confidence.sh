#!/bin/bash
# ARC confidence extraction - all models, 3 seeds
# Best LRs from existing LR search results (002 plan)
# Unstable models use stabilized LR

SCRIPT=/workspace/erase/scripts/train_arc_confidence.py
OUT=/workspace/erase/outputs/plan2_v2/confidence_v2

# Model -> best_lr mapping
# bert-large uses lr=1e-5 (stabilized)
# t5-v1_1-large uses lr=3e-5 (best stable)
declare -A MODEL_LR=(
    ["bert-mini"]=5e-5
    ["bert-small"]=5e-5
    ["bert-medium"]=5e-5
    ["bert-base"]=5e-5
    ["bert-large"]=1e-5
    ["t5-v1_1-small"]=3e-5
    ["t5-v1_1-base"]=1e-4
    ["t5-v1_1-large"]=3e-5
    ["t5-v1_1-xl"]=1e-4
    ["pythia-14m"]=1e-3
    ["pythia-31m"]=1e-4
    ["pythia-70m"]=3e-4
    ["pythia-160m"]=2e-5
    ["pythia-410m"]=1e-5
    ["pythia-1b"]=1e-4
    ["pythia-1.4b"]=5e-5
    ["pythia-2.8b"]=5e-5
    ["pythia-6.9b"]=5e-5
)

# bf16 models
declare -A USE_BF16=(
    ["pythia-2.8b"]=1
    ["pythia-6.9b"]=1
    ["t5-v1_1-xl"]=1
)

GPU=$1  # 0 or 1
shift
MODELS=("$@")

for model in "${MODELS[@]}"; do
    lr=${MODEL_LR[$model]}
    bf16_flag=""
    if [[ -n "${USE_BF16[$model]}" ]]; then
        bf16_flag="--bf16"
    fi
    for seed in 1 2 3; do
        outdir=$OUT/${model}_seed${seed}
        if [ -f "$outdir/avg_conf.json" ]; then
            echo "SKIP $model seed=$seed"
            continue
        fi
        echo "=== $model seed=$seed lr=$lr GPU=$GPU ==="
        mkdir -p $outdir
        python3 $SCRIPT --model $model --lr $lr --epochs 5 --batch_size 8 --seed $seed --gpu $GPU --output_dir $outdir $bf16_flag
    done
done
echo "DONE $(date)"
