import numpy as np
import math
from DecisionTree import DecisionTree, Leaf
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator


class DTClassifier(BaseEstimator):

    valid_stopping_criteria = {"none", "gain_threshold", "size_threshold", "chi_square", "depth_threshold"}

    def __init__(
            self,
            stopping_criteria: str = "none",
            gain_threshold: float = 0.0,
            size_threshold: int = 0,
            depth_threshold: int = 0
    ):
        if stopping_criteria not in self.valid_stopping_criteria:
            raise ValueError(f"{stopping_criteria} is not a valid stopping criteria")
        self.stopping_criteria = stopping_criteria
        self.gain_threshold = gain_threshold
        self.size_threshold = size_threshold
        self.depth_threshold = depth_threshold
        self.X = None
        self.y = None
        self.tree = None
        self.ignore_features = None
        self._features = None
        self._feature_names = None
        self._X_train = None
        self._svg = []
        self._idx = 0

    class Split:
        def __init__(self, X, y, sample_weighs, feature, n_parts, mean=None, decision_matrix=None):
            self.X = X
            self.y = y
            self.sample_weights = sample_weighs
            self.feature = feature
            self.n_parts = n_parts
            self.mean = mean
            self.decision_matrix = decision_matrix

        def get_parts(self):
            parts = [([], [], []) for _ in range(self.n_parts)]
            for i in range(len(self.X)):
                x = self.X[i]
                if self.mean is None:
                    idx = self.decision_matrix[x[self.feature]]
                else:
                    idx = int(x[self.feature] > self.mean)
                parts[idx][0].append(x)
                parts[idx][1].append(self.y[i])
                parts[idx][2].append(self.sample_weights[i])
            parts = list(map(lambda t: (np.array(t[0]), np.array(t[1]), np.array(t[2])), parts))
            return parts

    @staticmethod
    def _entropy(labels):
        n_labels = len(labels)
        if n_labels <= 1:
            return 0
        value, counts = np.unique(labels, return_counts=True)
        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)
        if n_classes <= 1:
            return 0
        ent = 0.0
        for i in probs:
            ent -= i * math.log2(i)
        return ent

    def _information_gain(self, s, weights, splits):
        k = len(splits)
        return self._entropy(s) - sum(sum(splits[i][2]) / sum(weights) * self._entropy(splits[i][1]) for i in range(k))

    @staticmethod
    def _chi2_test(labels, parts):
        n_labels = len(labels)
        labels_map = {labels[i]: i for i in range(n_labels)}
        obs = []
        for p in parts:
            _, part_labels, _ = p
            if len(part_labels) == 0:
                continue
            count = [0 for _ in range(n_labels)]
            for y in part_labels:
                count[labels_map[y]] += 1
            obs.append(count)
        res = chi2_contingency(obs)
        return res.pvalue < 0.05

    def _build_tree(self, X, y, sample_weights, depth):
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) == 1:
            return Leaf(label=labels[0])
        if len(self._features) == 0:
            return Leaf(label=labels[np.argmax(counts)])
        if self.stopping_criteria == "size_threshold" and len(X) < self.size_threshold:
            return Leaf(label=labels[np.argmax(counts)])
        if depth == self.depth_threshold:
            return Leaf(label=labels[np.argmax(counts)])

        best_partition = (None, None)
        best_gain = -np.inf
        found_feature = False

        for feature in self._features:
            if self.ignore_features is not None and feature in self.ignore_features:
                continue
            col = X[:, feature]

            if isinstance(col[0], (float, int)):
                mean = col.mean()
                split = DTClassifier.Split(X, y, sample_weights, feature, n_parts=2, mean=mean)
                parts = split.get_parts()
            else:
                values = np.unique(self._X_train[:, feature])
                n_values = len(values)
                map_to_enum = {values[i]: i for i in range(n_values)}
                split = DTClassifier.Split(X, y, sample_weights, feature, n_parts=n_values, decision_matrix=map_to_enum)
                parts = split.get_parts()

            if self.stopping_criteria == "chi_square" and not self._chi2_test(labels, parts):
                continue

            found_feature = True
            gain = self._information_gain(y, sample_weights, parts)
            if gain > best_gain:
                best_gain = gain
                best_partition = (feature, split)

        if not found_feature:
            return Leaf(label=labels[np.argmax(counts)])

        feature, split = best_partition
        parts = split.get_parts()

        if any(len(parts[i][0]) == 0 for i in range(split.n_parts)):
            return Leaf(label=labels[np.argmax(counts)])
        if self.stopping_criteria == "gain_threshold" and best_gain < self.gain_threshold:
            return Leaf(label=labels[np.argmax(counts)])

        children = [self._build_tree(*parts[i], depth=depth+1) for i in range(split.n_parts)]
        return DecisionTree(feature=feature, children=children, mean=split.mean, decision_matrix=split.decision_matrix)

    def fit(self, X, y, sample_weights=None, ignore_features=None):
        if hasattr(X, 'columns'):
            self._feature_names = list(X.columns)
        X = np.array(X)
        y = np.array(y)
        self._X_train = X
        self._features = range(X.shape[1])
        self.ignore_features = ignore_features
        if sample_weights is None:
            sample_weights = np.ones(len(X))
        self.tree = self._build_tree(X, y, sample_weights, depth=0)

    def predict(self, X_test):
        y_test = []
        X_test = np.array(X_test)
        for x in X_test:
            cur = self.tree
            while isinstance(cur, DecisionTree):
                cur = cur.make_decision(x)
            y_test.append(cur.label)
        return np.array(y_test)



