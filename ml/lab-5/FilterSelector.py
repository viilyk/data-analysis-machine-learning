from BaseSelector import BaseSelector
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix
import numpy as np


class FilterSelector(BaseSelector):

    def __init__(self, n_features: int = 2):
        super().__init__(None, n_features)

    def fit_transform(self, X, y):
        n, m = X.shape
        X = X.toarray()

        labels = np.unique(y)
        n_labels = len(labels)
        label_map = {labels[i]: i for i in range(n_labels)}
        y = np.array(list(map(lambda label: label_map[label], y)))

        self.feature_importance = [abs(pearsonr(X[:, i], y).statistic) for i in range(m)]
        sorted_features = np.argsort(self.feature_importance)
        remove_features = sorted_features[:m - self.n_features]
        self.leave_features = sorted_features[m - self.n_features:]
        X = np.delete(X, remove_features, axis=1)
        return csr_matrix(X)
