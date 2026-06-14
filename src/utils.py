import os
import random
import numpy as np

def set_seed(seed=42):
    """Sets the seed for reproducibility across standard libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
