from enum import Enum
from numpy.typing import ArrayLike

class Parameter(Enum):
    DEFAULT = 0
    D1_PARAMETERS = 1
    D2_PARAMETERS = 2
    TRAINING = 3


class Algorithm:
    def __init__(
        self, training_x: ArrayLike, training_y: ArrayLike, testing_x: ArrayLike,
        parmater_type: Parameter
    ) -> None:
        raise NotImplementedError

    def train(self) -> None:
        raise NotImplementedError

    def predict(self) -> ArrayLike:
        raise NotImplementedError
