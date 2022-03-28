'''Distortion computation.'''

import information
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim

def MSE(x, y):
    error_signal = x.astype(np.float64) - y
    return information.average_energy(error_signal)

def RMSE(x, y):
    return math.sqrt(MSE(x, y))

def SSIM(x, y):
    return ssim(x, y, data_range=y.max() - y.min(), full=False)
