from DTClassifier import DTClassifier
from sklearn.base import BaseEstimator
import numpy as np
import math


class RFClassifier(BaseEstimator):

    valid_stopping_criteria = {"none", "gain_threshold", "size_threshold", "chi_square", "depth_threshold"}

    def __init__(
            self,
            stopping_criteria: str = "none",
            gain_threshold: float = 0.0,
            size_threshold: int = 0,
            depth_threshold: int = 0,
            n_trees: int = 25
    ):
        if stopping_criteria not in self.valid_stopping_criteria:
            raise ValueError(f"{stopping_criteria} is not a valid stopping criteria")
        self.stopping_criteria = stopping_criteria
        self.gain_threshold = gain_threshold
        self.size_threshold = size_threshold
        self.depth_threshold = depth_threshold
        self.n_trees = n_trees
        self.trees = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n, m = X.shape
        self.trees = []
        for i in range(self.n_trees):
            tree = DTClassifier(self.stopping_criteria, self.gain_threshold, self.size_threshold, self.depth_threshold)
            indices = np.random.choice(n, n)
            X_train = X[indices]
            y_train = y[indices]
            features_to_ignore = np.random.choice(m, m - int(math.sqrt(m)), replace=False)
            tree.fit(X_train, y_train, ignore_features=features_to_ignore)
            self.trees.append(tree)

    def predict(self, X_test):
        predictions = np.array([tree.predict(X_test) for tree in self.trees])
        y_test = []
        for i in range(len(X_test)):
            col = predictions[:, i]
            labels, counts = np.unique(col, return_counts=True)
            y_test.append(labels[np.argmax(counts)])
        return np.array(y_test)
