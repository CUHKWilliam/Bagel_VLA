MODEL_SIZE=16
TOKEN_NUM=3

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file accelerate_config_single_node.yaml \
src/lerobot/scripts/train_dist.py \
--dataset.repo_id="/dataset_rc_mm/tangwl3@xiaopeng.com/datasets/libero_object" \
--policy.model_size=$MODEL_SIZE \
--dataset.token_num=$TOKEN_NUM \
--output_dir="./outputs/train/libero_object_init_from_scratch" \
--policy.push_to_hub=false \
--policy.repo_id="lerobot/pi0" \
--eval_freq=20000000 \
--save_freq=1000 \
--job_name="libero-object_init_from_scratch" \
--wandb.project="exp_formal" \
--policy.use_ref false \
--policy.type="pi0" \
--resume true \
--config_path="./outputs/train/libero_object_init_from_scratch/checkpoints/last/pretrained_model/train_config.json"
