#!/bin/bash
# Monitor GPUs and launch ARC experiments when idle.
# Checks every 30 minutes. Launches run_arc.sh on each idle GPU.
# Usage: bash gpu_monitor.sh

set -e
export HF_HOME=~/hf_home

BASE=/workspace/erase
LOG_DIR=$BASE/outputs/plan2/logs
mkdir -p $LOG_DIR

GPU0_STARTED=false
GPU1_STARTED=false

# Check if ARC already completed
check_arc_done() {
    local gpu=$1
    if [ "$gpu" = "0" ]; then
        # GPU 0 runs bert-base, pythia-410m
        if [ -f "$BASE/outputs/plan2/lr_search/pythia-410m_lr1e-4/results.json" ]; then
            return 0
        fi
    elif [ "$gpu" = "1" ]; then
        # GPU 1 runs bert-large, pythia-1b
        if [ -f "$BASE/outputs/plan2/lr_search/pythia-1b_lr3e-3/results.json" ]; then
            return 0
        fi
    fi
    return 1
}

is_gpu_idle() {
    local gpu=$1
    local util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu | tr -d ' ')
    local mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu | tr -d ' ')
    # Idle = utilization 0% AND memory < 1000 MiB
    if [ "$util" -le 5 ] && [ "$mem" -lt 1000 ]; then
        return 0
    fi
    return 1
}

echo "============================================"
echo "GPU Monitor started at $(date)"
echo "Waiting for idle GPUs to launch ARC experiments..."
echo "Checking every 30 minutes."
echo "============================================"

while true; do
    # Check GPU 0
    if [ "$GPU0_STARTED" = false ]; then
        if check_arc_done 0; then
            echo "[$(date)] GPU 0: ARC already completed, skipping."
            GPU0_STARTED=true
        elif is_gpu_idle 0; then
            echo "[$(date)] GPU 0 is idle! Launching ARC experiments..."
            nohup bash $BASE/shell/run_arc.sh 0 > $LOG_DIR/gpu0_arc.log 2>&1 &
            GPU0_STARTED=true
            echo "[$(date)] GPU 0: ARC launched (PID: $!)"
        else
            echo "[$(date)] GPU 0: still busy"
        fi
    fi

    # Check GPU 1
    if [ "$GPU1_STARTED" = false ]; then
        if check_arc_done 1; then
            echo "[$(date)] GPU 1: ARC already completed, skipping."
            GPU1_STARTED=true
        elif is_gpu_idle 1; then
            echo "[$(date)] GPU 1 is idle! Launching ARC experiments..."
            nohup bash $BASE/shell/run_arc.sh 1 > $LOG_DIR/gpu1_arc.log 2>&1 &
            GPU1_STARTED=true
            echo "[$(date)] GPU 1: ARC launched (PID: $!)"
        else
            echo "[$(date)] GPU 1: still busy"
        fi
    fi

    # Exit if both started
    if [ "$GPU0_STARTED" = true ] && [ "$GPU1_STARTED" = true ]; then
        echo "[$(date)] Both GPUs handled. Monitor exiting."
        break
    fi

    echo "[$(date)] Sleeping 30 minutes..."
    sleep 1800
done
