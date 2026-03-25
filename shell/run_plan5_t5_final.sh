#!/bin/bash
# Plan 5 T5 v1.1 final — sentinel + Adafactor + warmup
# Usage: bash run_plan5_t5_final.sh <role>
# role: local0, local1, remote0, remote1
set -e

ROLE=$1
SCRIPT=/workspace/erase/scripts/train_mnli_t5v11_final.py
OUT=/workspace/erase/outputs/plan5

run_lr_search() {
    local model=$1 gpu=$2 bs=$3 bf16=$4
    shift 4; local lrs=("$@")
    echo "=== $model LR Search (GPU $gpu) ==="
    for lr in "${lrs[@]}"; do
        local outdir=$OUT/lr_search/${model}_sentinel_lr${lr}
        mkdir -p $outdir
        if [ -f "$outdir/results.json" ]; then echo "SKIP $lr"; continue; fi
        echo "  $model lr=$lr"
        python3 $SCRIPT --model $model --split split1 --lr $lr --epochs 1 --batch_size $bs --seed 1 --gpu $gpu $bf16 --mode lr_search --output_dir $outdir
    done
    local best_lr="" best_acc=0
    for lr in "${lrs[@]}"; do
        local rfile=$OUT/lr_search/${model}_sentinel_lr${lr}/results.json
        if [ ! -f "$rfile" ]; then continue; fi
        local acc=$(python3 -c "import json; print(json.load(open('$rfile'))['val_acc'])")
        echo "  $model lr=$lr val_acc=$acc"
        local better=$(python3 -c "print(1 if float('$acc') > $best_acc else 0)")
        if [ "$better" = "1" ]; then best_acc=$acc; best_lr=$lr; fi
    done
    echo ">>> Best $model: $best_lr ($best_acc)"
    echo $best_lr
}

run_confidence() {
    local model=$1 gpu=$2 best_lr=$3 bs=$4 bf16=$5
    echo "=== $model Confidence (GPU $gpu, lr=$best_lr) ==="
    for split in split1 split2 split3; do
        local outdir=$OUT/confidence/${model}_sentinel_${split}
        mkdir -p $outdir
        if [ -f "$outdir/avg_conf.json" ]; then echo "SKIP $split"; continue; fi
        echo "  $model $split"
        python3 $SCRIPT --model $model --split $split --lr $best_lr --epochs 1 --batch_size $bs --seed 1 --gpu $gpu $bf16 --mode confidence --output_dir $outdir
    done
    python3 -c "
import json, numpy as np, os
all_conf = {}
for s in ['split1', 'split2', 'split3']:
    path = '$OUT/confidence/${model}_sentinel_' + s + '/avg_conf.json'
    if os.path.exists(path):
        with open(path) as f: all_conf.update(json.load(f))
if all_conf:
    confs = np.array(list(all_conf.values()))
    print(f'$model 90K: mean={confs.mean():.4f}, std={confs.std():.4f}, >0.95={np.mean(confs>0.95)*100:.1f}%')
    with open('$OUT/confidence/${model}_90k_avg_conf.json', 'w') as f: json.dump(all_conf, f)
"
    echo "[$model] DONE"
}

case $ROLE in
    local0)
        # t5-v1_1-base: 5e-5 fixed, confidence only
        run_confidence t5-v1_1-base 0 5e-5 16 ""
        # t5-v1_1-small: LR search + confidence
        BEST=$(run_lr_search t5-v1_1-small 0 16 "" 3e-5 5e-5 1e-4 | tail -1)
        run_confidence t5-v1_1-small 0 $BEST 16 ""
        ;;
    local1)
        # t5-v1_1-large: LR search + confidence
        BEST=$(run_lr_search t5-v1_1-large 1 16 "" 3e-5 5e-5 1e-4 | tail -1)
        run_confidence t5-v1_1-large 1 $BEST 16 ""
        ;;
    remote0)
        # t5-v1_1-xl: LR search + confidence (bf16 LoRA)
        BEST=$(run_lr_search t5-v1_1-xl 0 4 "--bf16" 3e-5 5e-5 1e-4 | tail -1)
        run_confidence t5-v1_1-xl 0 $BEST 4 "--bf16"
        ;;
    remote1)
        # pythia-12b: already uses train_mnli_proxy.py, skip here
        echo "pythia-12b uses separate script"
        ;;
esac
