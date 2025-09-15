class DecisionTree:

    def __init__(self, feature, children, mean=None, decision_matrix=None):
        self.mean = mean
        self.decision_matrix = decision_matrix
        self.feature = feature
        self.children = children

    def make_decision(self, x):
        if self.mean is None:
            if x[self.feature] in self.decision_matrix:
                idx = self.decision_matrix[x[self.feature]]
            else:
                idx = 0
        else:
            idx = int(x[self.feature] > self.mean)
        return self.children[idx]


class Leaf:

    def __init__(self, label):
        self.label = label
