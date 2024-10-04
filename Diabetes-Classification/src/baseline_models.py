from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

class BaselineModels:
    def __init__(self, strategies=['stratified', 'most_frequent']):
        """
        Initialize baseline models
        """
        self.strategies = strategies
        self.models = {}

    def train(self, X_train, y_train):
        """
        Train DummyClassifiers
        """
        for strategy in self.strategies:
            dummy_clf = DummyClassifier(strategy=strategy, random_state=42)
            dummy_clf.fit(X_train, y_train)
            self.models[strategy] = dummy_clf

    def evaluate(self, X, y):
        """
        Evaluate the trained models
        """
        results = {}
        for strategy, model in self.models.items():
            y_pred = model.predict(X)
            f1 = f1_score(y, y_pred)
            results[strategy] = f1
        return results
