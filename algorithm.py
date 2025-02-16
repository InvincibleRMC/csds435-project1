from numpy.typing import ArrayLike

class Algorithm:

    def __init__(self, training_x, training_y, testing_x) -> None:
        raise NotImplementedError

    def train(self) -> None:
        raise NotImplementedError
    
    def predict(self) -> ArrayLike:
        raise NotImplementedError