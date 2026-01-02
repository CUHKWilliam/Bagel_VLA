MODEL_SIZE=16
TOKEN_NUM=30000

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file accelerate_config_single_node.yaml \
src/lerobot/scripts/train_dist.py \
--dataset.repo_id="/mnt/data/dataset/ipec_datasets/viola_lerobot,/mnt/data/dataset/ipec_datasets/jaco_play_lerobot" \
--policy.model_size=$MODEL_SIZE \
--dataset.token_num=$TOKEN_NUM \
--output_dir="./outputs/train/phase-1_base-action-data-only_debug" \
--policy.push_to_hub=false \
--policy.repo_id="lerobot/pi0" \
--eval_freq=20000000 \
--save_freq=20000 \
--job_name="libero_init_from_scratch" \
--wandb.project="exp_formal" \
--policy.use_ref false \
--policy.type="pi0"
