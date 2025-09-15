import numpy as np
from scipy.sparse import csr_matrix
from BaseSelector import BaseSelector


class EmbeddedSelector(BaseSelector):
    def __init__(self, model, n_features: int = 2):
        super().__init__(model, n_features)

    def fit_transform(self, X, y):
        n, m = X.shape
        self.model.fit(X, y)
        self.feature_importance = self.model.feature_importances_
        sorted_features = np.argsort(self.feature_importance)
        remove_features = sorted_features[:m - self.n_features]
        self.leave_features = sorted_features[m - self.n_features:]
        X = X.toarray()
        X = np.delete(X, remove_features, axis=1)
        return csr_matrix(X)
