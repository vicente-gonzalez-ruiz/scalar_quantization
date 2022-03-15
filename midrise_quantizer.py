'''Mid-rise scalar quantization.'''

import numpy as np

name = "mid-rise"

def quantize(x: np.ndarray, quantization_step: float) -> np.ndarray:
    assert quantization_step > 0
    k = np.floor(x / quantization_step).astype(np.int)
    return k

def dequantize(k: np.ndarray, quantization_step: float) -> np.ndarray:
    y = quantization_step * (k + 0.5)
    return y

def quan_dequan(x: np.ndarray, quantization_step:float) -> np.ndarray:
    k = quantize(x, quantization_step)
    y = dequantize(k, quantization_step)
    return y, k
