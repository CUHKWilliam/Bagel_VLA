MODEL_SIZE=4
TOKEN_NUM=0.2

CUDA_VISIBLE_DEVICES=6 accelerate launch --config_file accelerate_config_4xA6000.yaml \
  src/lerobot/scripts/train_dist.py \
  --dataset.repo_id=/root/data/lerobot/libero_spatial_no_noops_lerobot_v21 \
  --policy.model_size=$MODEL_SIZE \
  --dataset.token_num=$TOKEN_NUM \
  --output_dir="./outputs/train/act-und-gen-ratio-1-0-1_model-und_data-tok-${TOKEN_NUM}M_model-param-${MODEL_SIZE}B" \
  --policy.push_to_hub=false \
  --policy.repo_id="lerobot/pi0" \
  --eval_freq=1000 \
  --save_freq=10000 \
  --job_name="act-und-gen-ratio-1-0-1_model-und_data-tok-${TOKEN_NUM}M_model-param-${MODEL_SIZE}B" \
  --wandb.project="exp_scaling_law_libero" \
  --policy.type="pi0"
