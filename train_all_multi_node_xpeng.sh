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
OUTPUT_DIR="./outputs/train/train_all_formal"
JOB_NAME="train_all_formal"
WANDB_PROJECT="debug"
NNODES=2
GPUS_PER_NODE=1

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
echo "Environment Information:" | tee -a "${LOG_FILE}"
echo "NODE_RANK=${NODE_RANK}" | tee -a "${LOG_FILE}"
echo "MASTER_ADDR=${MASTER_ADDR}" | tee -a "${LOG_FILE}"
echo "MASTER_PORT=${MASTER_PORT}" | tee -a "${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_FILE}"
echo "NCCL_DEBUG=${NCCL_DEBUG}" | tee -a "${LOG_FILE}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Start training with torchrun
echo "Launching torchrun..." | tee -a "${LOG_FILE}"
echo "Command: torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} src/lerobot/scripts/train_dist.py ..." | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"


# 启动
torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=${GPUS_PER_NODE} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  src/lerobot/scripts/train_dist.py \
    --dataset.repo_id="/dataset_rc_mm/share/datasets/ml-site.cdn-apple.com/egodex_lerobot_gr00t/part5/wrap" \
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
    --config_path="./outputs/train/formal_train_all/checkpoints/last/pretrained_model/train_config.json" \
    2>&1 | tee -a "${LOG_FILE}"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "Training completed on Node ${NODE_RANK}" | tee -a "${LOG_FILE}"
echo "Exit code: ${EXIT_CODE}" | tee -a "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}