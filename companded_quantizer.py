'''Companded (using mu-law) scalar quantization.'''

import numpy as np
import deadzone_quantizer as deadzone

def muLaw_compress(x: np.ndarray, mu: float) -> np.ndarray:
    return np.log(1+mu*np.abs(x))/np.log(1+mu)*np.sign(x)

def muLaw_expand(y: np.ndarray, mu: float) -> np.ndarray:
    return (1/mu)*(((1+mu)**np.abs(y))-1)*np.sign(y)

def quantize(x: np.ndarray, quantization_step: float) -> np.ndarray:
    '''Companded mu-law deadzone quantizer'''
    mu = 255
    x_compressed = (32768*(muLaw_compress(x/32768, mu)))
    k = deadzone.quantize(x_compressed, quantization_step).astype(np.int16)
    return k

def dequantize(k: np.ndarray, quantization_step: float) -> np.ndarray:
    '''Companded mu-law deadzone dequantizer'''
    mu = 255
    z_compressed = deadzone.dequantize(k, quantization_step)
    y = np.round(32768*muLaw_expand(z_compressed/32768, mu)).astype(np.int16)
    return y

def quan_dequan(x: np.ndarray, quantization_step:float) -> np.ndarray:
    k = quantize(x, quantization_step)
    y = dequantize(k, quantization_step)
    return y, k
