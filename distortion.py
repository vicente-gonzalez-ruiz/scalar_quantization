'''Distortion computation.'''

import information
import numpy as np
import math

def MSE(x, y):
    error_signal = x.astype(np.float64) - y
    return information.average_energy(error_signal)

def RMSE(x, y):
    return math.sqrt(MSE(x, y))
