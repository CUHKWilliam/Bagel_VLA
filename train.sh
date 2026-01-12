CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --config_file accelerate_config.yaml  lerobot/scripts/train.py --policy.path=lerobot/pi0 --dataset.repo_id libero_90_no_noops_lerobot_all
