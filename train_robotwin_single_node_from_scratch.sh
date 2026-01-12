export WANDB_API_KEY="d196633f749ba370ba23ee7d629a231e41694d2a"
export WANDB_ENTITY="jackson_personal"

TASK_NAME=${1}
MODEL_SIZE=16
TOKEN_NUM=30000
JOB_NAME="robotwin_${TASK_NAME}_demo_clean_50_from_scratch"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file accelerate_config_single_node.yaml \
src/lerobot/scripts/train_dist.py \
--dataset.repo_id="/mnt/data/dataset/RoboTwin2.0/lerobot/robotwin_${TASK_NAME}_demo_clean_50" \
--policy.model_size=$MODEL_SIZE \
--dataset.token_num=$TOKEN_NUM \
--output_dir="./outputs/train/$JOB_NAME" \
--policy.push_to_hub=false \
--policy.repo_id="lerobot/pi0" \
--eval_freq=20000000 \
--save_freq=300 \
--job_name=$JOB_NAME \
--wandb.project="exp_formal" \
--policy.action_dim=14 \
--policy.optimizer_lr=0.0001 \
--policy.type="pi0" \
--policy.use_ref=false 
# --resume true \
# --config_path="./outputs/train/$JOB_NAME/checkpoints/last/pretrained_model/train_config.json" 
