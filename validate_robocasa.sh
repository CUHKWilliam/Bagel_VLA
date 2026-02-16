CUDA_VISIBLE_DEVICES=0 python3 validate_robocasa.py \
--policy.path="./outputs/train/formal_train_scratch_gr00t_gr1_unified.PnPBottleToCabinetClose/checkpoints/00000285000/pretrained_model" \
--policy.repo_id="lerobot/pi0" \
--resume true \
--config_path="./outputs/train/formal_train_scratch_gr00t_gr1_unified.PnPBottleToCabinetClose/checkpoints/00000285000/pretrained_model/train_config.json" \
--policy.use_ref=false
