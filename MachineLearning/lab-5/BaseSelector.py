import numpy as np


class BaseSelector:

    def __init__(self, model, n_features: int = 2):
        self.model = model
        self.n_features = n_features
        self.feature_importance = None
        self.leave_features = None

    def get_words(self, vocabulary, return_importance=False):
        inv_vocabulary = {v: k for k, v in vocabulary.items()}
        words = []
        importance = []
        for i in reversed(self.leave_features):
            words.append(inv_vocabulary[i])
            if return_importance:
                importance.append(self.feature_importance[i])

        if return_importance:
            return np.array(words), np.array(importance)
        else:
            return np.array(words)
