import os
from pathlib import Path

def find_missing_episodes_jsonl(root_dir=None):
    """
    Iterate through all directories and find those missing meta/episodes.jsonl
    
    Args:
        root_dir (str): Root directory to search. If None, uses current directory.
    
    Returns:
        list: List of directories missing meta/episodes.jsonl
    """
    if root_dir is None:
        root_dir = os.getcwd()
    
    root_path = Path(root_dir)
    missing_dirs = []
    
    # Iterate through all directories
    for item in sorted(os.listdir(root_path)):
        item = Path(os.path.join(root_path, item))
        meta_episodes_path = item / 'meta' / 'episodes.jsonl'
         
        if not meta_episodes_path.exists():
            missing_dirs.append(str(item))
    print(missing_dirs)
    return missing_dirs
find_missing_episodes_jsonl('/dataset_rc_mm/share/datasets/modelscope.cn/agibot_world/agibot_world_beta_gripper_top_head_lerobot_gr00t/agibotworld')
