source /dataset_rc_mm/tangwl3@xiaopeng.com/anaconda3/bin/activate /dataset_rc_mm/tangwl3@xiaopeng.com/anaconda3
conda activate bagel_vla
export WANDB_API_KEY="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"
# 可选 NCCL 环境（按你网络修改）=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export NCCL_IB_DISABLE=1
#export NCCL_P2P_DISABLE=1
#export NCCL_NET=Socket; # 数据传输协议，如果使用IB网卡协议，则不需要配置
#export NCCL_SOCKET_IFNAME=bond0.2000;  # 指定的socket协议网口，默认是eth0
#export NCCL_SHM_DISABLE=1;  # 强制使用P2P协议，会自动使用IB协议或IP socket
#export NCCL_SOCKET_NTHREADS=4;  # socket协议线程数，默认是1,范围1-16，数字越大数据传输越快
#export NCCL_P2P_DISABLE=0;  # 关闭p2p传输，使用NVLink or PCI可配置，默认可不配置
export NCCL_IB_DISABLE=1;  # 为1表示禁用IB协议，如果使用IB则设置为0
export NCCL_DEBUG=INFO # DEBUG打印日志的等级
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1          # 有 IB/ROCE 则改 0
# export NCCL_SOCKET_IFNAME=eth0    # 换成实际网卡名

MODEL_SIZE=16 ## no use, full param
TOKEN_NUM=1000000000000 # no use, full tokens
OUTPUT_DIR="./outputs/train/formal_train_all"
JOB_NAME="formal_train_all"
WANDB_PROJECT="debug"
NNODES=3
MASTER_PORT=23089
GPUS_PER_NODE=8

LOG_DIR="./logs/${JOB_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/node_${NODE_RANK}.log"


echo "=========================================="
echo "Starting training on Node ${NODE_RANK}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Log file: ${LOG_FILE}"
echo "Timestamp: ${TIMESTAMP}"
echo "=========================================="
echo ""

# Log environment info
echo "Environment Information:"
echo "NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NCCL_DEBUG=${NCCL_DEBUG}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo ""

# Start training with torchrun
echo "Launching torchrun..."
echo "Command: torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} src/lerobot/scripts/train_dist_torchrun.py ..."
echo ""

# 启动
torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=${GPUS_PER_NODE} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  src/lerobot/scripts/train_dist_torchrun.py \
    --dataset.repo_id="/dataset_rc_mm/share/datasets/modelscope.cn/Galaxea/Galaxea-Open-World-Dataset/lerobot_decompressed/*,/dataset_rc_mm/share/datasets/ml-site.cdn-apple.com/egodex_lerobot_gr00t/part1/*,/dataset_rc_mm/share/datasets/ml-site.cdn-apple.com/egodex_lerobot_gr00t/part3/*,/dataset_rc_mm/share/datasets/ml-site.cdn-apple.com/egodex_lerobot_gr00t/part4/*,/dataset_rc_mm/share/datasets/ml-site.cdn-apple.com/egodex_lerobot_gr00t/part5/*,/dataset_rc_mm/share/datasets/modelscope.cn/agibot_world/agibot_world_beta_gripper_top_head_lerobot_gr00t/agibotworld/*,/dataset_rc_mm/share/datasets/huggingface.co/nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot/*" \
    --policy.model_size=${MODEL_SIZE} \
    --dataset.token_num=${TOKEN_NUM} \
    --output_dir="${OUTPUT_DIR}" \
    --policy.push_to_hub=false \
    --policy.repo_id="lerobot/pi0" \
    --eval_freq=20000000 \
    --save_freq=3000 \
    --job_name="${JOB_NAME}" \
    --wandb.project="${WANDB_PROJECT}" \
    --policy.use_ref false \
    --policy.type="pi0" \
    --resume true \
    --config_path="./outputs/train/formal_train_all/checkpoints/last/pretrained_model/train_config.json"

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training completed on Node ${NODE_RANK}"
echo "Exit code: ${EXIT_CODE}"
echo "Log saved to: ${LOG_FILE}"
echo "=========================================="

exit ${EXIT_CODE}
#     --dataset.repo_id="/dataset_rc_mm/share/datasets/modelscope.cn/Galaxea/Galaxea-Open-World-Dataset/lerobot_decompressed/*,/dataset_rc_mm/share/datasets/ml-site.cdn-apple.com/egodex_lerobot_gr00t/part1/*,/dataset_rc_mm/share/datasets/ml-site.cdn-apple.com/egodex_lerobot_gr00t/part3/*,/dataset_rc_mm/share/datasets/ml-site.cdn-apple.com/egodex_lerobot_gr00t/part4/*,/dataset_rc_mm/share/datasets/ml-site.cdn-apple.com/egodex_lerobot_gr00t/part5/*,/dataset_rc_mm/share/datasets/modelscope.cn/agibot_world/agibot_world_beta_gripper_top_head_lerobot_gr00t/agibotworld/*,/dataset_rc_mm/share/datasets/huggingface.co/nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot/*" \
