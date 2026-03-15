# cgmnet/utils/data_utils.py
import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Any

# Hardcoded configurations for common, known datasets
DATASET_TASK_TYPES = {
    'bace': 'cls', 'bbbp': 'cls', 'clintox': 'cls',
    'sider': 'cls', 'tox21': 'cls', 'toxcast': 'cls','metstab':'cls','estrogen':'cls',
    'esol': 'reg', 'freesolv': 'reg', 'lipo': 'reg', 'test_fint': 'reg',
}
DATASET_TASK_COUNTS = {
    'clintox': 2, 'sider': 27, 'tox21': 12, 'toxcast': 617,'metstab': 2,'estrogen':2,
}
DATASET_METRICS = {
    'bace': ['auc', 'ap'], 'bbbp': ['auc'], 'clintox': ['auc'],'metstab':['auc'],'estrogen':['auc'],
    'sider': ['auc'], 'tox21': ['auc'], 'toxcast': ['auc'],
    'esol': ['rmse', 'mae', 'r2'], 'freesolv': ['rmse', 'mae', 'r2'],
    'lipo': ['rmse', 'mae', 'r2'], 'test_fint': ['rmse', 'mae', 'r2'],
}

def get_task_config(dataset_name: str, dataset_root: Path) -> Dict[str, Any]:
    """
    Gets task configuration via a hybrid approach:
    1. Checks hardcoded dictionaries.
    2. If not found, falls back to loading 'config.json'.
    """
    if dataset_name in DATASET_TASK_TYPES:
        print(f"Info: Using hardcoded configuration for '{dataset_name}'.")
        return {
            "task_type": DATASET_TASK_TYPES[dataset_name],
            "n_tasks": DATASET_TASK_COUNTS.get(dataset_name, 1),
            "metrics": DATASET_METRICS.get(dataset_name, [])
        }
        
    dataset_dir = dataset_root / dataset_name
    config_path = dataset_dir / "config.json"
    print(f"Info: Config for '{dataset_name}' not in code. Loading from {config_path}")
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration for dataset '{dataset_name}' not found. "
            f"Please add it to dicts in 'data_utils.py' or create a 'config.json' in '{dataset_dir}'."
        )
    
    with config_path.open('r') as f: config = json.load(f)

    if not all(k in config for k in ['task_type', 'metrics']):
        raise ValueError("config.json is missing required keys: 'task_type', 'metrics'.")
    
    config.setdefault('n_tasks', 1)
    return config

def get_split_indices(split_dir: Path, scaffold_id: int = 0) -> Dict[str, np.ndarray]:
    """Loads train/validation/test indices from a .npy split file."""
    for name in [f'scaffold-seed-{scaffold_id}.npy', f'scaffold-{scaffold_id}.npy']:
        split_file = split_dir / name
        if split_file.exists():
            split_data = np.load(split_file, allow_pickle=True)
            if split_data.shape == (): return split_data.item()
            elif split_data.ndim == 1 and len(split_data) == 3:
                return {'train': split_data[0], 'valid': split_data[1], 'test': split_data[2]}
    raise FileNotFoundError(f"Split file for scaffold_id {scaffold_id} not found in {split_dir}")
