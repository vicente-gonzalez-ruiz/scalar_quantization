'''Lloyd-Max scalar quantization. Use K-Means or K-Medoids (depending
on a parameter) to compute the centers (an average in the case of
K-Means and the best input (a medoid) in the case of K-Medoids).'''

import logging
import logging_config
logger = logging.getLogger(__name__)
#logging.basicConfig(format="[%(filename)s:%(lineno)s %(levelname)s probando %(funcName)s()] %(message)s")
##logger.setLevel(logging.CRITICAL)
##logger.setLevel(logging.ERROR)
logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

import numpy as np
from scipy.ndimage import uniform_filter1d
from quantization import Quantizer
import warnings

name = "Lloyd-Max"

class LloydMax_Quantizer(Quantizer):

    def __init__(self, Q_step, counts, min_val=0, max_val=255, max_iters=100, epsilon=1E5):
        '''Defines the working parameters of the quantizer:

        Q_step: quantization step size.

        counts: number of ocurrences of each possible input
        sample. None of such ocurrences should be 0 (although this
        issue is solved in this constructor). This will generate that
        the enumeration of the quantization indexes are not
        consecutive, and usually, will not start at 0 (see the
        notebook), but this should not be a problem for those entropy
        codecs that do not expect to find an input in the range [0,
        ...]. Notice, anyway, that the used centroids (the number of
        different quantization indexes used by the encoder in the
        quantization of the signal, that can be smaller than the
        number of bins) must be sent to the decoder.
        
        [min_val, max_val]: expected dynamic range of the input
        signal.

        max_iters: maximum number of iterations.

        '''
        super().__init__(Q_step, min_val, max_val)
        assert np.all(counts) > 0
        self.N_bins = (max_val + 1 - min_val) // Q_step
        initial_boundaries = np.linspace(min_val, max_val + 1, self.N_bins + 1)
        initial_centroids = 0.5 * (initial_boundaries[1:] + initial_boundaries[:-1])
        self.centroids = initial_centroids
        logger.info(f"initial_centroids={self.centroids}")
        self.boundaries = initial_boundaries
        logger.info(f"initial_boundaries={self.boundaries}")
        prev_b = np.zeros(self.boundaries.size)
        for j in range(max_iters):
            prev_b[:] = self.boundaries
            self._compute_boundaries()
            max_abs_error = np.max(np.abs(prev_b - self.boundaries))
            if (j>0) and (max_abs_error <= epsilon):
                break
            self._compute_centroids(counts)
        logger.info(f"centroids={self.centroids}")

    def _compute_boundaries(self):
        '''The new boundaries are in the middle points between the
        current centroids.

        '''
        logger.debug(f"centroids={self.centroids}")
        self.boundaries = uniform_filter1d(self.centroids, size=2, origin=-1)[:-1]
        self.boundaries = np.concatenate(([0], self.boundaries, [256]))
        logger.info(f"boundaries={self.boundaries}")

    def _compute_centroids(self, counts):
        '''Compute the centroid of each bin.'''
        centroids = []
        bin_size = self.Q_step
        logger.info(f"bin_size={bin_size}")
        for i in range(self.N_bins):
            b_i = i*bin_size
            b_i_1 = (i+1)*bin_size
            logger.debug(f"b_i={b_i} b_i_1={b_i_1}")
            # See from scipy.ndimage import center_of_mass
            mass = np.sum([j*counts[j] for j in range(b_i, b_i_1)])
            total_counts_in_bin = np.sum([counts[j] for j in range(b_i, b_i_1)])
            centroid = mass/total_counts_in_bin
            logger.debug(f"centroid={centroid}")
            centroids.append(centroid)
        self.centroids = np.array(centroids)
        logger.info(f"centroids={self.centroids}")

    def encode(self, x):
        '''Find the quantization indexes for the signal <x>.

        '''
        logger.debug(f"x.shape={x.shape}")
        k = np.searchsorted(self.boundaries, x, side="right") - 1
        logger.debug(f"k.shape={k.shape}")
        logger.debug(f"max(k)={np.max(k)}")
        return k

    def decode(self, k):
        '''Return the centroids corresponding to the quantization indexes
        <k>.

        '''
        logger.debug(f"k.shape={k.shape}")
        logger.debug(f"max(k)={np.max(k)}")
        logger.debug(f"centroids.shape={self.centroids.shape}")
        y = self.centroids[k]
        logger.debug(f"y.shape={y.shape}")
        return y

    def get_representation_levels(self):
        '''In a Lloyd-Max quantizer the representation levels are the
        centroids.

        '''
        return self.centroids
