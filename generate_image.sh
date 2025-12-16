MODEL_SIZE=16
TOKEN_NUM=3

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file accelerate_config_single_node.yaml \
src/lerobot/scripts/generate_image.py \
--dataset.repo_id=/dataset_rc_mm/share/datasets/huggingface.co/IPEC-COMMUNITY/bridge__lerobot \
--policy.model_size=$MODEL_SIZE \
--dataset.token_num=$TOKEN_NUM \
--output_dir="./outputs/train/act-und-gen-ratio-0-0-1_model-und_data-tok-${TOKEN_NUM}G_model-param-${MODEL_SIZE}B_2" \
--policy.push_to_hub=false \
--policy.repo_id="lerobot/pi0" \
--eval_freq=20000000 \
--save_freq=1000 \
--job_name="act-und-gen-ratio-0-0-1_model-und_data-tok-${TOKEN_NUM}G_model-param-${MODEL_SIZE}B_2" \
--wandb.project="exp_gt_visual_gen" \
--policy.use_ref false \
--policy.type="pi0" \
--resume true \
--config_path="./outputs/train/act-und-gen-ratio-0-0-1_model-und_data-tok-${TOKEN_NUM}G_model-param-${MODEL_SIZE}B_2/checkpoints/100000/pretrained_model/train_config.json"
