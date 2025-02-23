from numpy.typing import ArrayLike
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from algorithm import Algorithm, Parameter

MLP_CLASSIFIER_PARAMETER_SPACE = {
    "hidden_layer_sizes": [(50, 50, 50), (50, 100, 50), (100,)],
    "activation": ["tanh", "relu"],
    "solver": ["sgd", "adam"],
    "alpha": [0.0001, 0.05],
    "learning_rate": ["constant", "adaptive"],
}

DEFAULT = {
    "activation": ["relu"],
    "alpha": [0.0001],
    "hidden_layer_sizes": [(100,)],
    "learning_rate": ["constant"],
    "solver": ["adam"],
}

BEST_PARAMETER_SPACE_D1 = {
    "activation": ["tanh"],
    "alpha": [0.0001],
    "hidden_layer_sizes": [(100,)],
    "learning_rate": ["adaptive"],
    "solver": ["sgd"],
}

BEST_PARAMETER_SPACE_D2 = {
    "activation": ["tanh"],
    "alpha": [0.0001],
    "hidden_layer_sizes": [(50, 100, 50)],
    "learning_rate": ["constant"],
    "solver": ["sgd"],
}


class NeuralNetwork(Algorithm):
    def __init__(
        self,
        training_x: ArrayLike,
        training_y: ArrayLike,
        testing_x: ArrayLike,
        parameter_type: Parameter,
    ) -> None:
        self.training_x = training_x
        self.training_y = training_y
        self.testing_x = testing_x

        match parameter_type:
            case Parameter.DEFAULT:
                self.parameters = DEFAULT
            case Parameter.D1_PARAMETERS:
                self.parameters = BEST_PARAMETER_SPACE_D1
            case Parameter.D2_PARAMETERS:
                self.parameters = BEST_PARAMETER_SPACE_D2
            case Parameter.TRAINING:
                self.parameters = MLP_CLASSIFIER_PARAMETER_SPACE

        self.network = MLPClassifier(max_iter=5000)

    def train(self) -> None:
        self.clf = GridSearchCV(self.network, self.parameters, n_jobs=-1)
        self.clf.fit(self.training_x, self.training_y)

    def predict(self) -> ArrayLike:
        return self.clf.predict(self.testing_x)
