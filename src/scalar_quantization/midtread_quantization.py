'''Mid-read (round) scalar quantization.'''

import numpy as np
from quantization import Quantizer

name = "mid-tread"

class Midtread_Quantizer(Quantizer):

    def quantize(self, x):
        k = np.rint(x / self.Q_step).astype(np.int)
        return k

    def dequantize(self, k):
        y = self.Q_step * k
        return y
