#!/bin/bash

MODEL_SIZE=16
TOKEN_NUM=3
export WANDB_API_KEY=801795babe1e93ec2fce3084446d3e7163f7658a
# Calculate world size from environment variables
# Set GPU visibility
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=bond0.2000;  # 指定的socket协议网口，默认是eth0
export NCCL_SHM_DISABLE=1;  # 强制使用P2P协议，会自动使用IB协议或IP socket
export NCCL_P2P_DISABLE=1;  # 关闭p2p传输，使用NVLink or PCI可配置，默认可不配置
export NCCL_DEBUG=INFO # DEBUG打印日志的等级
# Run training
accelerate launch \
    --num_processes 16 \
    --num_machines 2 \
    --machine_rank 0 \
    --main_process_ip 10.0.100.92 \
    --main_process_port 29505 \
    --config_file accelerate_config.yaml \
    src/lerobot/scripts/train_dist.py \
    --dataset.repo_id="/mnt/data/dataset/ipec_datasets/bridge_orig_lerobot" \
    --policy.model_size=$MODEL_SIZE \
    --dataset.token_num=$TOKEN_NUM \
    --output_dir="outputs/train/act-und-gen-ratio-1-0-1_model-und_data-tok-${TOKEN_NUM}G_model-param-${MODEL_SIZE}B" \
    --policy.push_to_hub=false \
    --policy.repo_id="lerobot/pi0" \
    --eval_freq=20000000 \
    --save_freq=1000 \
    --job_name="act-und-gen-ratio-1-0-0_model-und_data-tok-${TOKEN_NUM}G_model-param-${MODEL_SIZE}B" \
    --wandb.project="exp_formal" \
    --policy.use_ref false \
    --policy.type="pi0"
