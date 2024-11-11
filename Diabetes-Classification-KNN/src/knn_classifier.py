import numpy as np
from sklearn.metrics import f1_score

class KNNClassifier:
    def __init__(self, k=5):
        """
        Initialize the KNN classifier
        """
        self.k = k

    def fit(self, X, y):
        """
        Store training data
        """
        self.X_train = X.values
        self.y_train = y.values

    def predict(self, X):
        """
        Predict the class labels
        """
        X = X.values
        predictions = []
        for x_test in X:
            distances = np.sqrt(np.sum((self.X_train - x_test) ** 2, axis=1))
            nn_indices = np.argsort(distances)[:self.k]
            nn_classes = self.y_train[nn_indices]
            counts = np.bincount(nn_classes)
            prediction = np.argmax(counts)
            predictions.append(prediction)
        return np.array(predictions)

class KNNModel:
    def __init__(self, k_range=range(1,11)):
        """
        Initialize KNN model for tuning
        """
        self.k_range = k_range
        self.best_k = None
        self.best_f1 = 0
        self.best_model = None

    def hyperparameter_tuning(self, X_train, y_train, X_dev, y_dev):
        """
        Tune hyperparameters
        """
        for k in self.k_range:
            knn = KNNClassifier(k=k)
            knn.fit(X_train, y_train)
            y_pred_dev = knn.predict(X_dev)
            f1 = f1_score(y_dev, y_pred_dev)
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_k = k
                self.best_model = knn

    def evaluate(self, X, y):
        """
        Evaluate the best KNN model on the dataset.
        """
        if self.best_model is None:
            raise Exception("Model has not been trained yet.")
        y_pred = self.best_model.predict(X)
        f1 = f1_score(y, y_pred)
        return f1
