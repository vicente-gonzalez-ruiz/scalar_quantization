import numpy as np
from scalar_quantization.deadzone_quantization import Deadzone_Quantizer as Quantizer

if __name__ == "__main__":
    Q = Quantizer()
    x = np.linspace(-5, 5, 50)
    y, k = Q.encode_and_decode(x)
    for i in range(50):
        print(x[i], y[i])
