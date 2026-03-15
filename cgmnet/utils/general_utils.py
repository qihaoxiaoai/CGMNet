# cgmnet/utils/general_utils.py
"""
Contains general utility functions that are used across the project, such as
functions for ensuring experiment reproducibility.
"""
import os
import random
import numpy as np
import torch
import dgl


def set_random_seed(seed: int, n_threads: int = 1):
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.

    This function sets the seed for Python's `random`, `numpy`, `torch` (for both CPU
    and CUDA), and `dgl`. It also configures PyTorch's cuDNN settings for
    deterministic behavior.

    Args:
        seed (int): The random seed to use.
        n_threads (int): The number of threads for torch to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # These settings are necessary for full determinism with CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    torch.set_num_threads(n_threads)
    os.environ['PYTHONHASHSEED'] = str(seed)
