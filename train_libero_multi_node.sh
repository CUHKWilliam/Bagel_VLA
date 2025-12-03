#!/bin/bash

MODEL_SIZE=16
TOKEN_NUM=3
NNODES=3
NPROC_PER_NODE=8
export WANDB_API_KEY=801795babe1e93ec2fce3084446d3e7163f7658a
# Calculate world size from environment variables
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/log_node${NODE_RANK}.txt"
exec > "$LOG_FILE" 2>&1

echo "=== Environment Check ==="
echo "Log File: $LOG_FILE"
echo "Time: $(date)"
echo ""

# Main variables
echo "=== Core Distributed Training Variables ==="
printf "%-20s = %s\n" "NPROC_PER_NODE" "${NPROC_PER_NODE:-NOT SET}"
printf "%-20s = %s\n" "MASTER_PORT" "${MASTER_PORT:-NOT SET}"
printf "%-20s = %s\n" "NNODES" "${NNODES:-NOT SET}"
printf "%-20s = %s\n" "NODE_RANK" "${NODE_RANK:-NOT SET}"
printf "%-20s = %s\n" "MASTER_ADDR" "${MASTER_ADDR:-NOT SET}"
echo ""

source /dataset_rc_mm/tangwl3@xiaopeng.com/anaconda3/bin/activate /dataset_rc_mm/tangwl3@xiaopeng.com/anaconda3
conda activate bagel_vla

# Verify activation
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Error: Failed to activate bagel_env!"
    exit 1
fi

echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
python --version
echo ""

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run training
/dataset_rc_mm/tangwl3@xiaopeng.com/anaconda3/envs/bagel_vla/bin/accelerate launch \
    --num_processes $WORLD_SIZE \
    --num_machines $NNODES \
    --machine_rank $NODE_RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --config_file /dataset_rc_mm/tangwl3@xiaopeng.com/Bagel_VLA/accelerate_config.yaml \
    /dataset_rc_mm/tangwl3@xiaopeng.com/Bagel_VLA/src/lerobot/scripts/train_dist.py \
    --dataset.repo_id=/dataset_rc_mm/share/datasets/huggingface.co/IPEC-COMMUNITY/bridge__lerobot \
    --policy.model_size=$MODEL_SIZE \
    --dataset.token_num=$TOKEN_NUM \
    --output_dir="/dataset_rc_mm/tangwl3@xiaopeng.com/Bagel_VLA/outputs/train/act-und-gen-ratio-1-0-1_model-und_data-tok-${TOKEN_NUM}G_model-param-${MODEL_SIZE}B" \
    --policy.push_to_hub=false \
    --policy.repo_id="lerobot/pi0" \
    --eval_freq=20000000 \
    --save_freq=1000 \
    --job_name="act-und-gen-ratio-1-0-1_model-und_data-tok-${TOKEN_NUM}G_model-param-${MODEL_SIZE}B" \
    --wandb.project="exp_gt_visual_gen" \
    --policy.use_ref false \
    --policy.type="pi0"
