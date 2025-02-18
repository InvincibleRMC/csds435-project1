from algorithm import Algorithm
from sklearn.naive_bayes import GaussianNB
from numpy.typing import ArrayLike

class NaiveBayes(Algorithm):
    def __init__(self, training_x, training_y, testing_x) -> None:
        self.training_x = training_x
        self.training_y = training_y
        self.testing_x = testing_x

        self.network = GaussianNB()

    def train(self) -> None:
        self.network.fit(self.training_x, self.training_y)

    def predict(self) -> ArrayLike:
        return self.network.predict(self.testing_x)
    