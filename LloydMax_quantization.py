'''Lloyd-Max scalar quantization.'''

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
from sklearn import cluster
from quantization import Quantizer
import warnings

name = "Lloyd-Max"

class LloydMax_Quantizer(Quantizer):

    def __init__(self, x, Q_step, min_val=0, max_val=255):
        '''Creates the classifier using the samples in <X>. <Q_step> is the
        quantization step size, <min_val> is the minimum expected
        value faced by the classifier, and <max_val> the maximum
        value. Notice that <Q_step> is used only for providing an
        initial value for the centroids (the distance between the
        representation levels is not going to be <Q_step> in general).

        '''
        super().__init__(Q_step, min_val, max_val)
        self.N_clusters = (max_val + 1 - min_val) // Q_step
        initial_decision_levels = np.linspace(min_val, max_val + 1, self.N_clusters + 1).reshape(-1, 1)
        initial_centroids = 0.5 * (initial_decision_levels[1:] + initial_decision_levels[:-1])
        logger.info(f"initial_centroids={initial_centroids.squeeze()}")
        #self.classifier = cluster.KMeans(n_clusters=self.N_clusters, init=initial_centroids, n_init=1)
        #with warnings.catch_warnings():
         #   warnings.simplefilter("ignore")
            #self.classifier = cluster.KMeans(n_clusters=self.N_clusters, init=initial_centroids, algorithm="elkan")
        self.classifier = cluster.KMeans(n_clusters=self.N_clusters, init=initial_centroids, n_init=1)
        #self.classifier = cluster.KMeans(n_clusters=self.N_clusters)
        #np.random.seed(0)
        self.train(x)

    def train(self, x):
        flatten_x = x.reshape((-1, 1))
        #print("flatten_x.shape=", flatten_x.shape)
        self.classifier.fit(flatten_x)
        self.centroids = self.classifier.cluster_centers_.squeeze()
        #x_shape = x.shape
        #k = self.classifier.labels_
        #k.shape = x_shape
        idx = np.argsort(self.classifier.cluster_centers_.sum(axis=1))
        self.lut = np.zeros_like(idx)
        self.lut[idx] = np.arange(self.N_clusters)
        logger.info(f"lut={self.lut}")
        #logger.info(f"centroids={self.centroids}")
        #tmp = self.centroids[0]
        #self.centroids[0] = self.centroids[1]
        #self.centroids[1] = tmp
        logger.info(f"centroids={self.centroids}")
        argsort_lut = np.argsort(self.lut)
        sorted_centroids = self.centroids[argsort_lut]
        logger.info(f"sorted_centroids={sorted_centroids}")
        logger.info(f"labels={self.classifier.labels_} len={len(self.classifier.labels_)}")
        sorted_labels = self.lut[self.classifier.labels_]
        logger.info(f"sorted_labels={sorted_labels} len={len(sorted_labels)}")
        self.centroids[:] = sorted_centroids
        self.classifier.labels_ = sorted_labels

    def retrain(self, x):
        '''Retrain the classifier using previous centroids.'''
        self.classifier = cluster.KMeans(n_clusters=self.N_clusters, init=self.classifier.cluster_centers_, n_init=1)
        self.train(x)
        #classifier = cluster.KMeans(n_clusters=self.N_clusters, verbose=0)
        #flatten_x = x.reshape((-1, 1))
        #print("flatten_x.shape=", flatten_x.shape)
        #self.classifier.fit(flatten_x)
        #print("x.shape=",x.shape)
        #classifier.fit(flatten_x)
        #self.centroids = self.classifier.cluster_centers_.squeeze()
        #self.train(x)
        #flatten_X = X.reshape((-1, 1))
        #self.classifier.fit(flatten_X)
        #return classifier

    def quantize(self, x):
        '''Use the created classifier to find the quantization indexes
        (labels) for each point of <x>.

        '''
        k = self.classifier.predict(x.reshape((-1, 1)))
        k.shape = x.shape
        return k

    def dequantize(self, k):
        '''Return the centroids corresponding to the quantization indexes
        <k>.

        '''
        y = self.centroids[k]
        return y

    def get_representation_levels(self):
        '''In a Lloyd-Max quantizer the representation levels are the
        centroids computed by the classifier (K-Means).

        '''
        return self.centroids
