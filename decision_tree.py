from algorithm import Algorithm
from sklearn.tree import DecisionTreeClassifier
from numpy.typing import ArrayLike

class DecisionTree(Algorithm):
    def __init__(self, training_x, training_y, testing_x, parameter_type, max_depth=5) -> None:
        self.training_x = training_x
        self.training_y = training_y
        self.testing_x = testing_x

        self.network = DecisionTreeClassifier(max_depth=max_depth)

    def train(self) -> None:
        self.network.fit(self.training_x, self.training_y)

    def predict(self) -> ArrayLike:
        return self.network.predict(self.testing_x)
    