#!/bin/bash
# T5 v1.1 Adafactor distributed across 4 GPUs
# $1 = role: local0, local1, remote0, remote1
set -e

ROLE=$1
BASE=/workspace/erase
T5_SCRIPT=$BASE/scripts/train_mnli_t5_proxy.py
OUT=$BASE/outputs/plan5

run_lr_search() {
    local model=$1 gpu=$2 lr=$3 bs=$4 bf16=$5
    local outdir=$OUT/lr_search/${model}_adafactor_lr${lr}
    if [ -f "$outdir/results.json" ]; then echo "SKIP $model lr=$lr"; return; fi
    echo "  $model adafactor lr=$lr (GPU $gpu)"
    python3 $T5_SCRIPT --model $model --split split1 --lr $lr --epochs 1 --batch_size $bs --seed 1 --gpu $gpu --adafactor $bf16 --mode lr_search --output_dir $outdir
}

run_confidence() {
    local model=$1 gpu=$2 best_lr=$3 bs=$4 bf16=$5
    for split in split1 split2 split3; do
        local outdir=$OUT/confidence/${model}_adafactor_${split}
        if [ -f "$outdir/avg_conf.json" ]; then echo "SKIP $model $split"; continue; fi
        echo "  $model $split (GPU $gpu)"
        python3 $T5_SCRIPT --model $model --split $split --lr $best_lr --epochs 1 --batch_size $bs --seed 1 --gpu $gpu --adafactor $bf16 --mode confidence --output_dir $outdir
    done
    python3 -c "
import json, numpy as np, os
all_conf = {}
for s in ['split1', 'split2', 'split3']:
    path = '$OUT/confidence/${model}_adafactor_' + s + '/avg_conf.json'
    if os.path.exists(path):
        with open(path) as f: all_conf.update(json.load(f))
if all_conf:
    confs = np.array(list(all_conf.values()))
    print(f'$model 90K: mean={confs.mean():.4f}, std={confs.std():.4f}, >0.95={np.mean(confs>0.95)*100:.1f}%')
    with open('$OUT/confidence/${model}_90k_avg_conf.json', 'w') as f: json.dump(all_conf, f)
"
}

find_best_lr() {
    local model=$1
    local best_lr="" best_acc=0
    for lr in 1e-4 3e-4 5e-4 1e-3 3e-3; do
        local rfile=$OUT/lr_search/${model}_adafactor_lr${lr}/results.json
        if [ ! -f "$rfile" ]; then continue; fi
        local acc=$(python3 -c "import json; print(json.load(open('$rfile'))['val_acc'])")
        echo "  $model lr=$lr val_acc=$acc"
        local better=$(python3 -c "print(1 if float('$acc') > $best_acc else 0)")
        if [ "$better" = "1" ]; then best_acc=$acc; best_lr=$lr; fi
    done
    echo ">>> Best $model: $best_lr ($best_acc)"
    echo $best_lr
}

case $ROLE in
    local0)
        echo "=== Local GPU 0: t5-small LR+conf, t5-base conf ==="
        # t5-small LR search
        for lr in 1e-4 3e-4 5e-4 1e-3 3e-3; do run_lr_search t5-v1_1-small 0 $lr 16 ""; done
        BEST=$(find_best_lr t5-v1_1-small | tail -1)
        run_confidence t5-v1_1-small 0 $BEST 16 ""
        # t5-base confidence
        run_confidence t5-v1_1-base 0 3e-4 16 ""
        echo "=== Local GPU 0 DONE ==="
        ;;
    local1)
        echo "=== Local GPU 1: t5-xl LR+conf ==="
        for lr in 1e-4 3e-4 5e-4 1e-3 3e-3; do run_lr_search t5-v1_1-xl 1 $lr 4 "--bf16"; done
        BEST=$(find_best_lr t5-v1_1-xl | tail -1)
        run_confidence t5-v1_1-xl 1 $BEST 4 "--bf16"
        echo "=== Local GPU 1 DONE ==="
        ;;
    remote0)
        echo "=== Remote GPU 0: t5-large LR (3 LRs) ==="
        for lr in 1e-4 3e-4 5e-4; do run_lr_search t5-v1_1-large 0 $lr 16 ""; done
        echo "=== Remote GPU 0 LR DONE ==="
        ;;
    remote1)
        echo "=== Remote GPU 1: t5-large LR (2 LRs) ==="
        for lr in 1e-3 3e-3; do run_lr_search t5-v1_1-large 1 $lr 16 ""; done
        echo "=== Remote GPU 1 LR DONE ==="
        ;;
esac
