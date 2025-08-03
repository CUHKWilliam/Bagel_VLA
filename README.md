# Bagel_VLA - Training Setup

This repository contains the training code for Bagel_VLA. Follow the steps below to configure and run the training process.

## Setup Instructions

### 1. Configuration
Before running the training script, you need to configure these files:

#### GPU Selection
Edit `train.sh` to set CUDA_VISIBLE_DEVICES:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Specify which GPUs to use 
```

#### Parallel Processing Configuration
Modify `num_processes` in `accelerate_config.yaml` to match your GPU count:
```
num_processes: 8    # Use 8 GPU for training
```

#### Set up proper batch_size
Modify `batch_size` in the file `./lerobot/configs/train.py` to fit your GPU memory

### 2. Training
```
bash train.py
```
