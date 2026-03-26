#!/bin/bash
# GPU 1: C5_union, C6_dedup (6 runs)
python3 /workspace/erase/scripts/experiment_009_train_arc_t5xl.py \
  --gpu 1 \
  --conditions C5_union C6_dedup \
  --seeds 1 2 3 \
  --data_dir arc_k5 \
  --result_tag arc_t5xl_k5 \
  2>&1 | tee /workspace/erase/outputs/plan9/logs/arc_t5xl_gpu1.log
