#!/bin/bash
set -e
source /venv/main/bin/activate
export HF_HOME=/workspace/hf_home
cd /workspace/erase

OUT=outputs/plan5
T5_SCRIPT=scripts/train_mnli_t5_proxy.py

# GPU 0: t5-v1_1-base wider LR + confidence
run_base() {
    model=t5-v1_1-base
    echo "=== $model wider LR (GPU 0) ==="
    for lr in 5e-4 1e-3 3e-3; do
        outdir=$OUT/lr_search/${model}_lr${lr}
        if [ -f "$outdir/results.json" ]; then echo "SKIP $lr"; continue; fi
        echo "  $model lr=$lr"
        python3 $T5_SCRIPT --model $model --split split1 --lr $lr --epochs 1 --batch_size 16 --seed 1 --gpu 0 --mode lr_search --output_dir $outdir
    done

    BEST_LR=""
    BEST_ACC=0
    for lr in 1e-5 3e-5 5e-5 1e-4 3e-4 5e-4 1e-3 3e-3; do
        rfile=$OUT/lr_search/${model}_lr${lr}/results.json
        if [ ! -f "$rfile" ]; then continue; fi
        acc=$(python3 -c "import json; print(json.load(open('$rfile'))['val_acc'])")
        echo "  $model lr=$lr val_acc=$acc"
        better=$(python3 -c "print(1 if float('$acc') > $BEST_ACC else 0)")
        if [ "$better" = "1" ]; then BEST_ACC=$acc; BEST_LR=$lr; fi
    done
    echo ">>> Best $model LR: $BEST_LR (val_acc=$BEST_ACC)"

    for split in split1 split2 split3; do
        outdir=$OUT/confidence/${model}_${split}_v2
        if [ -f "$outdir/avg_conf.json" ]; then echo "SKIP $split"; continue; fi
        echo "  $model $split"
        python3 $T5_SCRIPT --model $model --split $split --lr $BEST_LR --epochs 1 --batch_size 16 --seed 1 --gpu 0 --mode confidence --output_dir $outdir
    done
    echo "[$model] DONE ($(date))"
}

# GPU 1: t5-v1_1-large split2 re-run
run_large_fix() {
    model=t5-v1_1-large
    BEST_LR=1e-4
    echo "=== $model split2 fix (GPU 1) ==="
    for seed in 2 3; do
        outdir=$OUT/confidence/${model}_split2_seed${seed}
        if [ -f "$outdir/avg_conf.json" ]; then echo "SKIP seed=$seed"; continue; fi
        echo "  $model split2 seed=$seed"
        python3 $T5_SCRIPT --model $model --split split2 --lr $BEST_LR --epochs 1 --batch_size 16 --seed $seed --gpu 1 --mode confidence --output_dir $outdir
    done
    echo "[$model split2 fix] DONE ($(date))"
}

echo "=== Starting remote experiments ($(date)) ==="
run_base &
run_large_fix &
wait
echo "=== ALL REMOTE DONE ($(date)) ==="
