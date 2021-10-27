'''Mid-rise scalar quantization.'''

import numpy as np

def quantize(x, quantization_step):
    k = np.floor(x / quantization_step).astype(np.int)
    return k

def dequantize(k, quantization_step):
    y = quantization_step * (k + 0.5)
    return y
