MODEL_SIZE=16
TOKEN_NUM=30000

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file accelerate_config_single_node.yaml \
src/lerobot/scripts/train_dist.py \
--dataset.repo_id="/mnt/data/dataset/ipec_datasets/bridge_orig_lerobot" \
--policy.model_size=$MODEL_SIZE \
--dataset.token_num=$TOKEN_NUM \
--output_dir="./outputs/train/phase-1_base-action-data-only" \
--policy.push_to_hub=false \
--policy.repo_id="lerobot/pi0" \
--eval_freq=20000000 \
--save_freq=20000 \
--job_name="act-und-gen-ratio-1-0-0_model-und_data-tok-${TOKEN_NUM}G_model-param-${MODEL_SIZE}B" \
--wandb.project="exp_formal" \
--policy.use_ref false \
--policy.type="pi0"
