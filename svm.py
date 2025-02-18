from algorithm import Algorithm
from sklearn.svm import SVC
from numpy.typing import ArrayLike

class SVM(Algorithm):
    def __init__(self, training_x, training_y, testing_x, kernel='rbf', C=1.0, gamma='scale') -> None:
        self.training_x = training_x
        self.training_y = training_y
        self.testing_x = testing_x

        self.network = SVC(kernel=kernel, C=C, gamma=gamma)

    def train(self) -> None:
        self.network.fit(self.training_x, self.training_y)

    def predict(self) -> ArrayLike:
        return self.network.predict(self.testing_x)