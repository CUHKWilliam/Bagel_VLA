CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config.yaml  lerobot/scripts/convert_ds_to_py_ckpt.py --policy.path=lerobot/pi0 --dataset.repo_id stack_boxes_dataset
