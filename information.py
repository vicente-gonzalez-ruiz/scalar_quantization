'''Information estimation.'''

import numpy as np
import math
#import logging
#import debug
import os

import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger.setLevel(logging.CRITICAL)
logger.setLevel(logging.ERROR)
#logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

def energy(x):
    return np.sum(x.astype(np.double)*x.astype(np.double))

def average_energy(x):
    return energy(x)/np.size(x)

def entropy(sequence_of_symbols):
    logger.debug(f"sequence_of_symbols={sequence_of_symbols}")
    assert sequence_of_symbols.ndim == 1
    value, counts = np.unique(sequence_of_symbols, return_counts = True)
    probs = counts / len(sequence_of_symbols)
    n_classes = np.count_nonzero(probs)
    logger.info(f"sequence_of_symbols.max()={sequence_of_symbols.max()}")
    #logging.info("information.entropy: sequence_of_symbols.max() =", sequence_of_symbols.max())
    #print(f"n_clases={n_classes}")
    logger.info(f"sequence_of_symbols.min()={sequence_of_symbols.min()}")
    #logging.info("information.entropy: sequence_of_symbols.min() =", sequence_of_symbols.min())
    logger.info(f"n_classes={n_classes}")
    #logging.info("information.entropy: n_clases = ", n_classes)
    #debug.print("information.entropy: probs =", probs)

    if n_classes <= 1:
        return 0

    _entropy = 0.
    #print(f"probs={probs} {np.sum(probs)}")
    for i in probs:
        _entropy -= i * math.log(i, 2)
    #logging.debug("information.entropy: _entropy =", _entropy)
    logger.info(f"entropy={_entropy}")

    return _entropy

if __name__ == "__main__":

    sequence = np.array([0,1], dtype=np.uint8)
    print(sequence, entropy(sequence))

    sequence = np.array([0,1,2,3], dtype=np.uint8)
    print(sequence, entropy(sequence))

    sequence = np.array([0,1,1,2,3], dtype=np.uint8)
    print(sequence, entropy(sequence))

    sequence = np.array([0,-1,1,-2,3], dtype=np.uint8)
    print(sequence, entropy(sequence))

    sequence = np.arange(21, dtype=np.uint8)
    print(sequence, entropy(sequence))    

    sequence = np.arange(21, dtype=np.uint8).reshape(3,7)
    print(sequence, entropy(sequence))
    
    sequence = np.arange(21, dtype=np.uint8).reshape(3,7)
    print(sequence, PNG_BPP(sequence))
    
