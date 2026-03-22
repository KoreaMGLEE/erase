#!/bin/bash
set -e

BASE=/workspace/erase
OUT=$BASE/outputs/plan0
SCRIPT=$BASE/scripts/train_mnli_proxy.py

echo "============================================"
echo "Step 1: Data Preparation"
echo "============================================"
python3 $BASE/scripts/prepare_data.py

echo ""
echo "============================================"
echo "Step 2: Phase 1 - LR Search (bert-base GPU:0, t5-base GPU:1)"
echo "============================================"

# BERT LR candidates
BERT_LRS="1e-5 2e-5 3e-5 5e-5 1e-4"
# T5 LR candidates
T5_LRS="1e-5 3e-5 5e-5 1e-4 3e-4"

# Run bert-base LR search on GPU 0 (background)
echo "Starting bert-base LR search on GPU 0..."
for lr in $BERT_LRS; do
    echo "  bert-base lr=$lr"
    python3 $SCRIPT \
        --model bert-base --split split1 --lr $lr \
        --epochs 1 --batch_size 16 --seed 1 --gpu 0 \
        --mode lr_search \
        --output_dir $OUT/lr_search/bert-base_lr${lr}
done &
PID_BERT=$!

# Run t5-base LR search on GPU 1 (background)
echo "Starting t5-base LR search on GPU 1..."
for lr in $T5_LRS; do
    echo "  t5-base lr=$lr"
    python3 $SCRIPT \
        --model t5-base --split split1 --lr $lr \
        --epochs 1 --batch_size 16 --seed 1 --gpu 1 \
        --mode lr_search \
        --output_dir $OUT/lr_search/t5-base_lr${lr}
done &
PID_T5=$!

# Wait for both
wait $PID_BERT
echo "bert-base LR search done!"
wait $PID_T5
echo "t5-base LR search done!"

echo ""
echo "============================================"
echo "Step 3: Select Best LR"
echo "============================================"

# Find best LR for bert-base
BEST_BERT_LR=""
BEST_BERT_ACC=0
for lr in $BERT_LRS; do
    acc=$(python3 -c "import json; d=json.load(open('$OUT/lr_search/bert-base_lr${lr}/results.json')); print(d['val_acc'])")
    echo "bert-base lr=$lr val_acc=$acc"
    better=$(python3 -c "print(1 if $acc > $BEST_BERT_ACC else 0)")
    if [ "$better" = "1" ]; then
        BEST_BERT_ACC=$acc
        BEST_BERT_LR=$lr
    fi
done
echo ">>> Best bert-base LR: $BEST_BERT_LR (val_acc=$BEST_BERT_ACC)"

# Find best LR for t5-base
BEST_T5_LR=""
BEST_T5_ACC=0
for lr in $T5_LRS; do
    acc=$(python3 -c "import json; d=json.load(open('$OUT/lr_search/t5-base_lr${lr}/results.json')); print(d['val_acc'])")
    echo "t5-base lr=$lr val_acc=$acc"
    better=$(python3 -c "print(1 if $acc > $BEST_T5_ACC else 0)")
    if [ "$better" = "1" ]; then
        BEST_T5_ACC=$acc
        BEST_T5_LR=$lr
    fi
done
echo ">>> Best t5-base LR: $BEST_T5_LR (val_acc=$BEST_T5_ACC)"

# Save LR selection
python3 -c "
import json
summary = {
    'bert-base': {'best_lr': '$BEST_BERT_LR', 'val_acc': $BEST_BERT_ACC},
    't5-base': {'best_lr': '$BEST_T5_LR', 'val_acc': $BEST_T5_ACC},
}
with open('$OUT/lr_search/lr_selection_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('LR selection saved!')
"

echo ""
echo "============================================"
echo "Step 4: Phase 2 - Confidence Measurement (3 splits)"
echo "============================================"

# bert-base on GPU 0
echo "Starting bert-base confidence runs on GPU 0..."
for split in split1 split2 split3; do
    echo "  bert-base $split lr=$BEST_BERT_LR"
    python3 $SCRIPT \
        --model bert-base --split $split --lr $BEST_BERT_LR \
        --epochs 1 --batch_size 16 --seed 1 --gpu 0 \
        --mode confidence \
        --output_dir $OUT/confidence/bert-base_${split}
done &
PID_BERT2=$!

# t5-base on GPU 1
echo "Starting t5-base confidence runs on GPU 1..."
for split in split1 split2 split3; do
    echo "  t5-base $split lr=$BEST_T5_LR"
    python3 $SCRIPT \
        --model t5-base --split $split --lr $BEST_T5_LR \
        --epochs 1 --batch_size 16 --seed 1 --gpu 1 \
        --mode confidence \
        --output_dir $OUT/confidence/t5-base_${split}
done &
PID_T52=$!

wait $PID_BERT2
echo "bert-base confidence done!"
wait $PID_T52
echo "t5-base confidence done!"

echo ""
echo "============================================"
echo "Step 5: Merge 90K Confidence"
echo "============================================"

python3 -c "
import json, numpy as np, os

OUT = '$OUT'
data_dir = os.path.join(OUT, 'data')

# Load split indices
splits = {}
for s in ['split1', 'split2', 'split3']:
    with open(os.path.join(data_dir, f'{s}_indices.json')) as f:
        splits[s] = json.load(f)

for model in ['bert-base', 't5-base']:
    all_conf = {}
    for s in ['split1', 'split2', 'split3']:
        conf_path = os.path.join(OUT, 'confidence', f'{model}_{s}', 'avg_conf.json')
        with open(conf_path) as f:
            conf = json.load(f)
        all_conf.update(conf)

    # Save merged 90K confidence
    save_path = os.path.join(OUT, 'confidence', f'{model}_90k_avg_conf.json')
    with open(save_path, 'w') as f:
        json.dump(all_conf, f)

    confs = np.array(list(all_conf.values()))
    print(f'{model} 90K confidence: mean={confs.mean():.4f}, std={confs.std():.4f}, '
          f'>0.95={np.mean(confs>0.95):.3f}, <0.40={np.mean(confs<0.4):.3f}')
    print(f'  Saved: {save_path} ({len(all_conf)} examples)')
"

echo ""
echo "============================================"
echo "EXPERIMENT COMPLETE"
echo "============================================"
