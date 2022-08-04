'''Lloyd-Max scalar quantization.'''

import numpy as np
from sklearn import cluster
from quantization import Quantizer

name = "Lloyd-Max"

class LloydMax_Quantizer(Quantizer):

    def __init__(self, X, Q_step, min_val=0, max_val=255):
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
        self.classifier = cluster.KMeans(n_clusters=self.N_clusters, init=initial_centroids)
        #self.classifier = cluster.KMeans(n_clusters=self.N_clusters)
        #np.random.seed(0)
        flatten_X = X.reshape((-1, 1))
        self.classifier.fit(flatten_X)
        self.centroids = self.classifier.cluster_centers_.squeeze()
        #x_shape = x.shape
        #k = self.classifier.labels_
        #k.shape = x_shape

    def retrain(self, X, centroids):
        '''Retrain the classifier using previous centroids.'''
        self.classifier = cluster.KMeans(n_clusters=self.N_clusters, init=centroids, verbose=0)
        flatten_X = X.reshape((-1, 1))
        self.classifier.fit(flatten_X)

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
