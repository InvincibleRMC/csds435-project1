from algorithm import Algorithm
from sklearn.neighbors import KNeighborsClassifier
from numpy.typing import ArrayLike

class NearestNeighbor(Algorithm):
    def __init__(self, training_x, training_y, testing_x, parameter_type, n_neighbors=5) -> None:
        self.training_x = training_x
        self.training_y = training_y
        self.testing_x = testing_x

        self.network = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self) -> None:
        self.network.fit(self.training_x, self.training_y)

    def predict(self) -> ArrayLike:
        return self.network.predict(self.testing_x)