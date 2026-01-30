import json
import os
from pathlib import Path

def check_dataset_versions(base_path="/publicdata-sh/huggingface.co/datasets/IPEC-COMMUNITY"):
    """
    Check codebase_version for all datasets in IPEC-COMMUNITY.
    Returns lists of v2.0 and v2.1 datasets.
    """
    v20_datasets = []
    v21_datasets = []
    failed_datasets = []
    
    base_path = Path(base_path)
    
    # Iterate through all dataset directories
    for dataset_dir in sorted(base_path.iterdir()):
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        info_json_path = dataset_dir / "main" / "meta" / "info.json"
        
        try:
            if not info_json_path.exists():
                failed_datasets.append((dataset_name, "info.json not found"))
                continue
            
            with open(info_json_path, 'r') as f:
                info = json.load(f)
            
            version = info.get("codebase_version", "unknown")
            
            if version == "v2.0":
                v20_datasets.append(dataset_name)
                print(f"✓ {dataset_name}: v2.0")
            elif version == "v2.1":
                v21_datasets.append(dataset_name)
                print(f"✓ {dataset_name}: v2.1")
            else:
                print(f"? {dataset_name}: {version}")
                
        except json.JSONDecodeError:
            failed_datasets.append((dataset_name, "JSON parse error"))
        except Exception as e:
            failed_datasets.append((dataset_name, str(e)))
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nv2.0 datasets ({len(v20_datasets)}):")
    for ds in v20_datasets:
        print(f"  - {ds}")
    
    print(f"\nv2.1 datasets ({len(v21_datasets)}):")
    for ds in v21_datasets:
        print(f"  - {ds}")
    
    if failed_datasets:
        print(f"\nFailed to check ({len(failed_datasets)}):")
        for ds, reason in failed_datasets:
            print(f"  - {ds}: {reason}")
    
    return v20_datasets, v21_datasets, failed_datasets

if __name__ == "__main__":
    v20, v21, failed = check_dataset_versions()
