import numpy as np

class LogisticRegressionModel:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.cost_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y, epsilon=1e-5):
        m = len(y)
        h = self.sigmoid(np.dot(X, self.weights))
        cost = (-1 / m) * (np.dot(y, np.log(h + epsilon)) + np.dot((1 - y), np.log(1 - h + epsilon)))
        return cost

    def gradient_descent(self, X, y):
        m = len(y)
        self.cost_history = []

        for _ in range(self.iterations):
            h = self.sigmoid(np.dot(X, self.weights))
            gradient = np.dot(X.T, (h - y)) / m
            self.weights -= self.learning_rate * gradient
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)

    def train(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Add intercept (bias)
        self.weights = np.zeros(X.shape[1])  # Initialize weights
        self.gradient_descent(X, y)  # Train the model

    def predict(self, X, threshold=0.5):
        X = np.insert(X, 0, 1, axis=1)
        probabilities = self.sigmoid(np.dot(X, self.weights))
        return [1 if p >= threshold else 0 for p in probabilities]

    def accuracy_score(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
