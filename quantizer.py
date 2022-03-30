'''Scalar Uniform Quantization.'''

import numpy as np

class Quantizer():
    
    def __init__(self, Q_step, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.Q_step = Q_step
        assert Q_step > 0
        
    def quan_dequan(self, x):
        k = self.quantize(x)
        y = self.dequantize(k)
        return y, k

    def get_representation_levels(self):
        quantization_indexes = np.linspace(self.min_val, (self.max_val + 1)//self.Q_step - 1, (self.max_val + 1)//self.Q_step).astype(np.uint8)
        representation_levels = self.dequantize(quantization_indexes)
        return representation_levels

    def get_decision_levels(self):
        range_input_values = np.linspace(self.min_val, self.max_val, self.max_val + 1).astype(np.uint8)
        range_input_indexes = self.quantize(range_input_values)
        DPCM = np.diff(range_input_indexes)
        decision_levels = np.where(DPCM != 0)
        extended_decision_levels = np.append(self.min_val, decision_levels)
        extended_decision_levels = np.append(extended_decision_levels, self.max_val)
        return extended_decision_levels
