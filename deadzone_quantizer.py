'''Deadzone scalar quantization.'''

import numpy as np

name = "dead-zone"

def quantize(x: np.ndarray, quantization_step: float) -> np.ndarray:
    assert quantization_step > 0
    k = (x / quantization_step).astype(np.int) # Quantization indexes
    #k = (x / quantization_step).astype(np.int32)
    #return k.astype(np.float32)
    return k

def dequantize(k: np.ndarray, quantization_step: float) -> np.ndarray:
    y = quantization_step * k
    return y

def quan_dequan(x: np.ndarray, quantization_step:float) -> np.ndarray:
    k = quantize(x, quantization_step)#.astype(np.int8)
    y = dequantize(k, quantization_step)
    return y, k
