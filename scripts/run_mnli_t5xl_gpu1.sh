#!/bin/bash
# GPU 1: C4_intersect, C5_union, C6_dedup (9 runs)
python3 /workspace/erase/scripts/experiment_009_train_mnli_t5xl.py \
  --gpu 1 \
  --conditions C4_intersect C5_union C6_dedup \
  --splits split1 split2 split3 \
  --result_tag mnli_t5xl_k5 \
  2>&1 | tee /workspace/erase/outputs/plan9/logs/mnli_t5xl_gpu1.log
