TASK_NAME=${1}
port=${2}
GPU_ID=${3}
ckpt_dir="outputs/train/robotwin_${TASK_NAME}_demo_clean_50_from_scratch/checkpoints/last/pretrained_model"
host=localhost

CUDA_VISIBLE_DEVICES=$GPU_ID python3 policy_serving/websocket_policy_server.py --checkpoint $ckpt_dir --host $host --port $port