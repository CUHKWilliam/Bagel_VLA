python -m lerobot.scripts.train \
    --policy.type=pi0 \
    --dataset.repo_id=lerobot/pi \
    --env.type= \
    --env.task=AlohaTransferCube-v0 \
    --log_freq=25 \
    --save_freq=100 \
    --output_dir=outputs/train/run_metaworld4openpi0
