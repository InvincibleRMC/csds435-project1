from numpy.typing import ArrayLike
from sklearn.neural_network import MLPClassifier

from algorithm import Algorithm

class NeuralNetwork(Algorithm):

    def __init__(self, training_x, training_y, testing_x) -> None:
        self.training_x = training_x
        self.training_y = training_y
        self.testing_x = testing_x

        self.network = MLPClassifier(max_iter=5000)

    def train(self) -> None:
        self.network.fit(self.training_x, self.training_y)

    def predict(self) -> ArrayLike:
        return self.network.predict(self.testing_x)