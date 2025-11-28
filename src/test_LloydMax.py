import numpy as np
from scalar_quantization.LloydMax_quantization import LloydMax_Quantizer as Quantizer

if __name__ == "__main__":
    '''
    number_of_bins = 8
    low = 0
    high = 255
    Q_step = (high - low + 1) // number_of_bins
    print("Q_step =", Q_step)
    x = np.random.randint(low, high, size=100)
    print(x)
    histogram, bin_edges = np.histogram(x, number_of_bins, range=(low, high))
    print("histogram =", histogram)
    print("bin_edges =", bin_edges)
    Q = Quantizer(Q_step, histogram)
    y, k = Q.encode_and_decode(x)
    print("y =", y)
    print("k =", k)
    '''

    # Generate a sample histogram (for example, using a Gaussian distribution)
    number_of_bins = 8
    low = -256
    high = 255
    Q_step = (high - low + 1) // number_of_bins
    print("Q_step =", Q_step)
    x = np.random.randint(low, high, size=100)

    histogram, _ = np.histogram(x, bins=(high-low+1), range=(-256, 255))

    # Create the Lloyd-Max quantizer with a quantization step and histogram
    q_step = 64
    quantizer = Quantizer(Q_step=q_step, counts=histogram, min_val=-256, max_val=255)
    # Print the representation levels (centroids)
    print("Representation levels (centroids):")
    print(quantizer.get_representation_levels())

    # Encode and decode an example signal
    example_signal = np.array([45, 150, 200, 100])
    quantized = quantizer.encode(x)
    decoded = quantizer.decode(quantized)

    print("\nExample signal:", x)
    print("Quantized signal:", quantized)
    print("Decoded signal:", decoded)
