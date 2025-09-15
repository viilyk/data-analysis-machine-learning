from math import pi, exp, sqrt
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator

import numpy as np


class MyKNN(BaseEstimator):
    metric_options = {'minkowski', 'cosine', 'chebyshev'}
    kernel_options = {'uniform', 'triangular', 'epanechnikov', 'gaussian'}

    def __init__(
            self,
            fix_window=False,
            k=None,
            window_size=None,
            leaf_size=30,
            metric='minkowski',
            p=2,
            kernel='uniform',
    ):
        if not fix_window:
            if k is None:
                raise ValueError("Expected k parameter for var mode of window")
            self.neighbors = NearestNeighbors(n_neighbors=k + 1, radius=window_size,
                                              leaf_size=leaf_size, metric=metric,
                                              p=p)
        else:
            if window_size is None:
                raise ValueError("Expected window_size parameter for fix mode of window")
            self.neighbors = NearestNeighbors(n_neighbors=k, radius=window_size,
                                              leaf_size=leaf_size, metric=metric,
                                              p=p)

        if metric not in self.metric_options:
            raise ValueError("Unknown metric function")
        if kernel not in self.kernel_options:
            raise ValueError("Unknown kernel function")

        self.x_train = None
        self.y_train = None
        self.k = k
        self.window_size = window_size
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.kernel = kernel
        self.weights_objects = None
        self.fix_window = fix_window

    @staticmethod
    def kernel_function(x, kernel):
        match kernel:
            case 'uniform':
                return 1 / 2 if abs(x) < 1 else 0
            case 'triangular':
                return 1 - abs(x) if abs(x) < 1 else 0
            case 'epanechnikov':
                return 3 / 4 * (1 - x * x) if abs(x) < 1 else 0
            case _:
                return 1 / sqrt(2 * pi) * exp(-x ** 2 / 2)

    def _query(self, x):
        if not self.fix_window:
            dist, ind = self.neighbors.kneighbors(x)
            return dist[:, -1], dist[:, :-1], ind[:, :-1]

        dist, ind = self.neighbors.radius_neighbors(x)
        return None, dist, ind

    def fit(self, x, y, weights_objects=None):
        if len(x) != len(y):
            raise ValueError(
                "The number of objects does not match the number of values")
        if weights_objects is not None:
            if len(weights_objects) != len(x):
                raise ValueError(
                    "The number of weights does not match the number of objects")
            self.weights_objects = weights_objects
        else:
            self.weights_objects = np.ones(len(x))

        self.neighbors.fit(x)
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        classes = np.unique(self.y_train)
        y = []
        h_optional, distances, indexes = self._query(x)
        h = self.window_size
        for i in range(len(x)):
            ind = indexes[i]
            dist = distances[i]
            if not self.fix_window:
                h = h_optional[i]
            if len(ind) == 0:
                raise ValueError("The window size is too small.")
            c = np.zeros(classes.size, dtype=float)
            for j in range(len(ind)):
                for k in range(len(classes)):
                    if self.y_train[ind[j]] == classes[k]:
                        c[k] += MyKNN.kernel_function(dist[j] / h, self.kernel) * self.weights_objects[j]
                        break
            y.append(classes[c.argmax()])
        return np.array(y)

