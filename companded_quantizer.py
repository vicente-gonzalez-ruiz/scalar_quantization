'''Companded (using mu-law) scalar quantization.'''

import numpy as np
import deadzone_quantizer as deadzone

def muLaw_compress(x, mu):
    return np.log(1+mu*np.abs(x))/np.log(1+mu)*np.sign(x)

def muLaw_expand(y, mu):
    return (1/mu)*(((1+mu)**np.abs(y))-1)*np.sign(y)

def quantize(x, quantization_step):
    '''Companded mu-law deadzone quantizer'''
    mu = 255
    x_compressed = (32768*(muLaw_compress(x/32768, mu)))
    k = deadzone.quantize(x_compressed, quantization_step)
    return k

def dequantize(k, quantization_step):
    '''Companded mu-law deadzone dequantizer'''
    mu = 255
    z_compressed = deadzone.dequantize(k, quantization_step)
    y = np.round(32768*muLaw_expand(z_compressed/32768, mu)).astype(np.int16)
    return y
