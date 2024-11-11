import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from itertools import product

class SupportVectorMachine:
    def __init__(self, kernel='linear', C=1.0):
        self.model = SVC(kernel=kernel, C=C)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

def cross_validate(model, X, y, n_folds=5):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_size = len(X) // n_folds
    accuracies = []

    for i in range(n_folds):
        test_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))
        
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    return mean_accuracy, accuracies

def custom_grid_search(X, y, param_grid, n_folds=5):
    best_score = -np.inf
    best_model = None
    best_params = {}

    for model_name, config in param_grid.items():
        model_class = config["model"]
        param_combinations = list(product(*config["params"].values()))
        param_names = list(config["params"].keys())

        for combination in param_combinations:
            params = dict(zip(param_names, combination))
            model_instance = model_class(**params)
            score, _ = cross_validate(model_instance, X, y, n_folds=n_folds)

            if score > best_score:
                best_score = score
                best_model = model_class
                best_params = params

    return best_model, best_params, best_score
