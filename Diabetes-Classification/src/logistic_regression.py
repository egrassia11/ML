from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score

class LogisticRegressionModel:
    def __init__(self, param_grid=None):
        """
        Initialize Logistic Regression model with a parameter grid for tuning
        """
        if param_grid is None:
            self.param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        else:
            self.param_grid = param_grid
        self.best_model = None
        self.best_params = {}
        self.best_f1 = 0
        self.baseline_model = None

    def train_baseline(self, X_train, y_train):
        """
        Train the baseline Logistic Regression model
        """
        self.baseline_model = LogisticRegression(max_iter=1000, random_state=42)
        self.baseline_model.fit(X_train, y_train)

    def evaluate_baseline(self, X, y):
        """
        Evaluate the baseline model
        """
        y_pred = self.baseline_model.predict(X)
        f1 = f1_score(y, y_pred)
        return f1

    def hyperparameter_tuning(self, X_train, y_train, X_dev, y_dev):
        """
        Perform hyperparameter tuning
        """
        for params in ParameterGrid(self.param_grid):
            model = LogisticRegression(max_iter=1000, random_state=42, **params)
            model.fit(X_train, y_train)
            pred_dev = model.predict(X_dev)
            f1 = f1_score(y_dev, pred_dev)
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_params = params
                self.best_model = model

    def evaluate(self, X, y):
        """
        Evaluate the best model on the dataset
        """
        y_pred = self.best_model.predict(X)
        f1 = f1_score(y, y_pred)
        return f1
