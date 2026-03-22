#!/bin/bash
# Plan 3: Difficulty-based training
# Usage: bash run_plan3.sh <gpu_id>
# GPU 0: bert-large (all experiments)
# GPU 1: pythia-1b (all experiments)
set -e
export HF_HOME=~/hf_home

GPU=$1
BASE=/workspace/erase
SCRIPT=$BASE/scripts/train_arc_plan3.py
DATA=$BASE/outputs/plan3/data
OUT=$BASE/outputs/plan3

if [ -z "$GPU" ]; then echo "Usage: bash run_plan3.sh <gpu_id>"; exit 1; fi

if [ "$GPU" = "0" ]; then
    MODEL="bert-large"
    LR="1e-5"
    BF16=""
    BS=8
elif [ "$GPU" = "1" ]; then
    MODEL="pythia-1b"
    LR="1e-4"
    BF16=""
    BS=8
else
    echo "GPU must be 0 or 1"; exit 1
fi

run_condition() {
    local exp=$1
    local cond=$2
    local indices_file=$3
    local outdir=$OUT/$exp/${MODEL}_${cond}

    if [ -f "$outdir/results.json" ]; then
        echo "  SKIP: $outdir already exists"
        return
    fi

    echo "  $MODEL $cond ($(date))"
    python3 $SCRIPT \
        --model $MODEL --condition $cond \
        --train_indices $indices_file \
        --data_file $DATA/matched_train.json \
        --lr $LR --epochs 10 --batch_size $BS --seed 1 --gpu $GPU \
        $BF16 --output_dir $outdir
}

echo "============================================"
echo "Plan 3: $MODEL on GPU $GPU ($(date))"
echo "============================================"

# --- Exp 1 ---
echo ""
echo "=== Experiment 1: Human Difficulty Subsets ==="
for cond in easy_only medium_only hard_only random_Neasy random_Nmed full; do
    # Extract condition indices from exp1_conditions.json
    python3 -c "
import json
with open('$DATA/exp1_conditions.json') as f:
    d = json.load(f)
with open('/tmp/plan3_indices.json', 'w') as f:
    json.dump(d['$cond'], f)
"
    run_condition exp1 $cond /tmp/plan3_indices.json
done

# --- Exp 2 ---
echo ""
echo "=== Experiment 2: Equal Size (N_hard=287) ==="
for cond in hard easy_sub medium_sub random; do
    python3 -c "
import json
with open('$DATA/exp2_conditions.json') as f:
    d = json.load(f)
with open('/tmp/plan3_indices.json', 'w') as f:
    json.dump(d['$cond'], f)
"
    run_condition exp2 $cond /tmp/plan3_indices.json
done

# --- Exp 3 ---
echo ""
echo "=== Experiment 3: Model Confidence Selection ==="
python3 -c "
import json
with open('$DATA/exp3_conditions.json') as f:
    d = json.load(f)
print(','.join(d.keys()))
" > /tmp/plan3_exp3_models.txt

IFS=',' read -ra CONF_MODELS < /tmp/plan3_exp3_models.txt
for conf_model in "${CONF_MODELS[@]}"; do
    for cond in model_easy_top_Neasy model_easy_top_Nhard model_hard_bottom_Neasy model_hard_bottom_Nhard; do
        python3 -c "
import json
with open('$DATA/exp3_conditions.json') as f:
    d = json.load(f)
with open('/tmp/plan3_indices.json', 'w') as f:
    json.dump(d['$conf_model']['$cond'], f)
"
        run_condition exp3 "${conf_model}_${cond}" /tmp/plan3_indices.json
    done
done

echo ""
echo "============================================"
echo "Plan 3: $MODEL ALL DONE ($(date))"
echo "============================================"
