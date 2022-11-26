'''Uniform Scalar Deadzone Quantization.'''

import numpy as np
from quantization import Quantizer

name = "dead-zone"

class Deadzone_Quantizer(Quantizer):
    
    def quantize(self, x):
        k = (x / self.Q_step).astype(np.int)
        return k

    def dequantize(self, k):
        y = np.where(k < 0, self.Q_step * (k - 0.5), k)
        y = np.where(k > 0, self.Q_step * (k + 0.5), y)
        #y = self.Q_step * k
        return y
