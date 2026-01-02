MODEL_SIZE=16
TOKEN_NUM=30000
JOB_NAME="libero-spatial_from_scratch"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --config_file accelerate_config_single_node.yaml \
src/lerobot/scripts/train_dist.py \
--dataset.repo_id="/mnt/data/dataset/libero_spatial" \
--policy.model_size=$MODEL_SIZE \
--dataset.token_num=$TOKEN_NUM \
--output_dir="./outputs/train/$JOB_NAME" \
--policy.push_to_hub=false \
--policy.repo_id="lerobot/pi0" \
--eval_freq=20000000 \
--save_freq=1000 \
--job_name=$JOB_NAME \
--wandb.project="exp_formal" \
--policy.use_ref=false \
--policy.optimizer_lr=0.0001 \
--policy.type="pi0"
# --policy.path="outputs/train/phase-1_base-action-data-only/checkpoints/last/pretrained_model" 
# --policy.type="pi0"
# --config_path="outputs/train/phase-1_base-action-data-only/checkpoints/last/pretrained_model/train_config.json"	
# --resume true \
# --steps=200000000 \
# --policy.optimizer_lr=0.0001
