#!/bin/bash

# Create log file with timestamp and node rank
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NODE_RANK_VALUE=${NODE_RANK:-0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/log_node${NODE_RANK_VALUE}_${TIMESTAMP}.txt"

# Redirect ALL output to the log file
exec > "$LOG_FILE" 2>&1

echo "=== Environment Check ==="
echo "Log File: $LOG_FILE"
echo "Time: $(date)"
echo ""

# Main variables
echo "=== Core Distributed Training Variables ==="
printf "%-20s = %s\n" "NPROC_PER_NODE" "${NPROC_PER_NODE:-NOT SET}"
printf "%-20s = %s\n" "MASTER_PORT" "${MASTER_PORT:-NOT SET}"
printf "%-20s = %s\n" "NNODES" "${NNODES:-NOT SET}"
printf "%-20s = %s\n" "NODE_RANK" "${NODE_RANK:-NOT SET}"
printf "%-20s = %s\n" "MASTER_ADDR" "${MASTER_ADDR:-NOT SET}"
echo ""

# Network info
echo "=== Network Information ==="
echo "Hostname: $(hostname)"
echo "Full Hostname: $(hostname -f 2>/dev/null || echo "N/A")"
echo "IP Address(es): $(hostname -I 2>/dev/null || echo "N/A")"
echo ""

# GPU info
echo "=== GPU Information ==="
if command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi output:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "nvidia-smi not available"
fi
echo ""

# Python/PyTorch info
echo "=== Python/PyTorch Information ==="
if command -v python3 &> /dev/null; then
    python3 -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA Version: {torch.version.cuda}')
        print(f'Number of GPUs: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
except ImportError:
    print('PyTorch not installed')
"
else
    echo "Python3 not available"
fi

echo ""
echo "=== End of Report ==="
