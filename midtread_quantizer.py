'''Mid-read (round) scalar quantization.'''

import numpy as np

name = "mid-tread"

def quantize(x: np.ndarray, quantization_step: float) -> np.ndarray:
    assert quantization_step > 0
    k = np.rint(x / quantization_step).astype(np.int)  # Quantization indexes
    #k = np.floor(x / quantization_step + 0.5).astype(np.int)  # Quantization indexes
    return k

def dequantize(k: np.ndarray, quantization_step: float) -> np.ndarray:
    y = quantization_step * k
    return y

def quan_dequan(x: np.ndarray, quantization_step:float) -> np.ndarray:
    k = quantize(x, quantization_step)
    y = dequantize(k, quantization_step)
    return y, k
