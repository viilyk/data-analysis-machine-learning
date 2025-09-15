from DTClassifier import DTClassifier
from sklearn.base import BaseEstimator
import numpy as np


class BoostClassifier(BaseEstimator):

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
        self.classifiers = None
        self.labels = None
        self.n_labels = None

    def _fit_bin_classifier(self, X, y, positive_label):
        X = np.array(X)
        y = np.array(y)
        y = np.array(list(map(lambda x: 1 if x == positive_label else -1, y)))
        n, m = X.shape
        w = np.ones(n) / n
        trees = []
        alpha = []

        for i in range(self.n_trees):
            trees.append(
                DTClassifier(
                    stopping_criteria=self.stopping_criteria,
                    gain_threshold=self.gain_threshold,
                    size_threshold=self.size_threshold,
                    depth_threshold=self.depth_threshold
                )
            )
            trees[i].fit(X, y, sample_weights=w)
            prediction = trees[i].predict(X)
            error = np.sum(w * (prediction != y)) / np.sum(w)
            alpha.append(
                1 / 2 * np.log((1 - error) / error) if error != 0 else np.log((1 - 1e-8) / 1e-8)
            )

            w = w * np.exp(-1 * alpha[i] * y * prediction)
            w = w / np.sum(w)

        self.classifiers[positive_label] = (trees, alpha)

    def fit(self, X, y):
        self.classifiers = {}
        self.labels = np.unique(y)
        self.n_labels = len(self.labels)
        for i in range(self.n_labels):
            self._fit_bin_classifier(X, y, positive_label=self.labels[i])

    def predict(self, X_test):
        predictions = []
        label_map = {}

        for label, classifier in self.classifiers.items():
            trees, alpha = classifier
            prediction = alpha[0] * trees[0].predict(X_test)
            for i in range(1, self.n_trees):
                prediction += alpha[i] * trees[i].predict(X_test)
            label_map[len(predictions)] = label
            predictions.append(prediction)

        predictions = np.array(predictions)
        y_test = []

        for i in range(len(X_test)):
            col = predictions[:, i]
            idx = np.argmax(col)
            y_test.append(label_map[idx])
        return np.array(y_test)

