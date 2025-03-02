import os
from argparse import ArgumentParser

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from algorithm import Algorithm, Parameter
from decision_tree import DecisionTree
from neural_network import NeuralNetwork
from nearest_neighbor import NearestNeighbor
from naive_bayes import NaiveBayes
from svm import SVM
import pandas as pd


def main() -> None:
    np.random.seed(123)

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--algorithm",
        choices=[
            "Nearest_Neighbor",
            "Decision_Tree",
            "Naive_Bayes",
            "SVM",
            "Neural_Network",
        ],
        default="Nearest_Neighbor",
    )
    parser.add_argument(
        "--hyperparameter-type",
        choices=["Training", "Default", "D1", "D2"],
        default="Default",
    )

    args = parser.parse_args()

    if args.dataset == 1:
        filename = "project1_dataset1.txt"
    else:
        filename = "project1_dataset2_numeric.txt"

    match args.algorithm:
        case "Nearest_Neighbor":
            algorithm_type: type[Algorithm] = NearestNeighbor
        case "Decision_Tree":
            algorithm_type = DecisionTree
        case "Naive_Bayes":
            algorithm_type = NaiveBayes
        case "SVM":
            algorithm_type = SVM
        case "Neural_Network":
            algorithm_type = NeuralNetwork
        case _:
            raise Exception()

    parameter_type: Parameter    
    
    match args.hyperparameter_type:
        case "Training":
            parameter_type = Parameter.TRAINING
        case "Default":
            parameter_type = Parameter.DEFAULT
        case "D1":
            parameter_type = Parameter.D1_PARAMETERS
        case "D2":
            parameter_type = Parameter.D2_PARAMETERS

    f = open(os.path.join("data", filename))

    data = np.loadtxt(f)

    K = 10

    splits = KFold(n_splits=K)
    scalar = StandardScaler()
    fold_acc = []
    algorithm: Algorithm

    for fold_num, (train_index, test_index) in enumerate(splits.split(data)):
        training_data = data[train_index]
        testing_data = data[test_index]

        end_index = len(training_data[0]) - 1

        training_x = training_data[:, 0:end_index]
        training_y = training_data[:, end_index]

        testing_x = testing_data[:, 0:end_index]
        testing_y = testing_data[:, end_index]

        training_x = scalar.fit_transform(training_x)
        testing_x = scalar.transform(testing_x)

        algorithm = algorithm_type(training_x, training_y, testing_x, parameter_type)
        algorithm.train()

        predictions = algorithm.predict()
        fold_acc.append(
            {
                "fold_num": fold_num,
                "Accuracy": accuracy_score(testing_y, predictions),
                "Precision": precision_score(testing_y, predictions),
                "Recall": recall_score(testing_y, predictions),
                "F1 Measure": f1_score(testing_y, predictions),
            }
        )
        # print(f"Fold Number {fold_num}")
        # print(f'Accuracy = {accuracy_score(testing_y, predictions)}')
        # print(f'Precision = {precision_score(testing_y, predictions)}')
        # print(f'Recall = {recall_score(testing_y, predictions)}')
        # print(f'F-1 Measure = {f1_score(testing_y, predictions)}')
    fold_acc = pd.DataFrame(fold_acc)
    mean_values = fold_acc.mean()
    # 将平均值作为新的一行添加
    fold_acc.loc['Mean'] = mean_values
    print(
        f"-----------------model: {args.algorithm}, dataset: {args.dataset}-----------------"
    )
    print(fold_acc)


main()
