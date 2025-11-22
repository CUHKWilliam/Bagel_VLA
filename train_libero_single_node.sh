MODEL_SIZE=16
TOKEN_NUM=1

CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file accelerate_config_single_node.yaml \
src/lerobot/scripts/train_dist.py \
--dataset.repo_id=/root/data/lerobot/bridge_orig_lerobot \
--policy.model_size=$MODEL_SIZE \
--dataset.token_num=$TOKEN_NUM \
--output_dir="./outputs/train/act-und-gen-ratio-1-0-1_model-und_data-tok-${TOKEN_NUM}G_model-param-${MODEL_SIZE}B" \
--policy.push_to_hub=false \
--policy.repo_id="lerobot/pi0" \
--eval_freq=100 \
--save_freq=10000 \
--job_name="act-und-gen-ratio-1-0-1_model-und_data-tok-${TOKEN_NUM}G_model-param-${MODEL_SIZE}B" \
--wandb.project="exp_scaling_law_libero" \
--policy.use_ref false \
--policy.type="pi0"
