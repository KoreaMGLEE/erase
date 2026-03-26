#!/bin/bash
# GPU 0: C1_full, C2_random, C3_self_easy_bert (9 runs)
python3 /workspace/erase/scripts/experiment_009_train_arc_t5xl.py \
  --gpu 0 \
  --conditions C1_full C2_random C3_self_easy_bert \
  --seeds 1 2 3 \
  --data_dir arc_k5 \
  --result_tag arc_t5xl_k5 \
  2>&1 | tee /workspace/erase/outputs/plan9/logs/arc_t5xl_gpu0.log
