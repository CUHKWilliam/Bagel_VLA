MODEL_SIZE=7
TOKEN_NUM=300000000000000000

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file accelerate_config_single_node.yaml \
src/lerobot/scripts/train_dist.py \
--dataset.repo_id="/dataset_rc_mm/tangwl3@xiaopeng.com/datasets/libero_*" \
--policy.model_size=$MODEL_SIZE \
--dataset.token_num=$TOKEN_NUM \
--output_dir="./outputs/train/formal_libero_all" \
--policy.push_to_hub=false \
--policy.repo_id="lerobot/pi0" \
--eval_freq=20000000 \
--save_freq=3000 \
--job_name="formal_libero_all" \
--wandb.project="exp_formal" \
--policy.use_ref false \
--policy.type="pi0" \
--resume true \
--config_path="./outputs/train/formal_libero_all/checkpoints/last/pretrained_model/train_config.json"

# --dataset.repo_id="/dataset_rc_mm/share/datasets/modelscope.cn/agibot_world/agibot_world_beta_gripper_top_head_lerobot_gr00t/agibotworld/*,/publicdata-sh/huggingface.co/datasets/IPEC-COMMUNITY/*,/dataset_rc_mm/share/datasets/modelscope.cn/Galaxea/Galaxea-Open-World-Dataset/lerobot_decompressed/*"
#--dataset.video_repo_id="/mnt/data/dataset/something-something-v2,/mnt/data/dataset/ego4d" \
# --dataset.repo_id="/mnt/data/dataset/agibot_lerobot/agibotworld/task_327,/mnt/data/dataset/ipec_datasets/viola_lerobot,/mnt/data/dataset/galaxea/Pour_Leftovers_20250802_012" \
# --dataset.repo_id=/mnt/data/dataset/agibot_lerobot/agibotworld/*,/mnt/data/dataset/ipec_datasets/*,/mnt/data/dataset/galaxea/*
# --dataset.vqa_repo_id="/mnt/data/dataset/cambrian,/mnt/data/dataset/robot2vlm/data" \
# /dataset_rc_mm/share/datasets/modelscope.cn/Galaxea/Galaxea-Open-World-Dataset/lerobot_decompressed/*,/dataset_rc_mm/share/datasets/ml-site.cdn-apple.com/egodex_lerobot_gr00t/part1/*,/dataset_rc_mm/share/datasets/ml-site.cdn-apple.com/egodex_lerobot_gr00t/part2/*,/dataset_rc_mm/share/datasets/ml-site.cdn-apple.com/egodex_lerobot_gr00t/part3/*,/dataset_rc_mm/share/datasets/ml-site.cdn-apple.com/egodex_lerobot_gr00t/part4/*,/dataset_rc_mm/share/datasets/ml-site.cdn-apple.com/egodex_lerobot_gr00t/part5/*	
