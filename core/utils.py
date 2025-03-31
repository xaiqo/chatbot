import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np

from config import USE_GPU

def to_gpu(array):
    if USE_GPU and CUPY_AVAILABLE:
        return cp.asarray(array)
    return np.asarray(array)

def to_cpu(array):
    if USE_GPU and CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return array.get()
    return array