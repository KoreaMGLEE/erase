#!/bin/bash
# GPU 0: C1_full, C2_random, C3_self_easy_t5xl (9 runs)
python3 /workspace/erase/scripts/experiment_009_train_mnli_t5xl.py \
  --gpu 0 \
  --conditions C1_full C2_random C3_self_easy_t5xl \
  --splits split1 split2 split3 \
  --result_tag mnli_t5xl_k5 \
  2>&1 | tee /workspace/erase/outputs/plan9/logs/mnli_t5xl_gpu0.log
