import numpy as np
from scalar_quantization.LloydMax_quantization import LloydMax_Quantizer as Quantizer

if __name__ == "__main__":
    Q = Quantizer()
    x = np.linspace(0, 255, 10)
    y, k = Q.encode_and_decode(x)
    print(y)
