CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config.yaml  lerobot/scripts/train.py --policy.path=lerobot/pi0 --dataset.repo_id=aopolin-lv/libero_spatial_no_noops_lerobot_v21
