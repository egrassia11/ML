from data_loader import load_and_split_data
from baseline_models import BaselineModels
from logistic_regression import LogisticRegressionModel
from knn_classifier import KNNModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def main():
    # Load and split data
    file_path = './data/diabetes.csv'
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_and_split_data(file_path)

    # Logistic Regression Model with default hyperparameters
    lr_model = LogisticRegressionModel()
    lr_model.train_baseline(X_train, y_train)
    baseline_f1_dev = lr_model.evaluate_baseline(X_dev, y_dev)
    print(f"Baseline logistic regression F1-score on development set: {round(baseline_f1_dev, 2)}")

    # Hyperparameter tuning for logistic regression
    lr_model.hyperparameter_tuning(X_train, y_train, X_dev, y_dev)
    improved_f1_dev = lr_model.best_f1
    print(f"Improved logistic regression F1-score on development set: {round(improved_f1_dev, 2)}\n")

    # Hyperparameter tuning for KNN
    knn_model = KNNModel(k_range=range(1, 21))
    knn_model.hyperparameter_tuning(X_train, y_train, X_dev, y_dev)
    print("KNN Classifier results:")
    print(f"Best k: {knn_model.best_k}")
    print(f"Best F1-score on development set: {round(knn_model.best_f1, 2)}\n")

    # Baseline DummyClassifier Models
    baseline_models = BaselineModels(strategies=['stratified', 'most_frequent'])
    baseline_models.train(X_train, y_train)
    dev_results = baseline_models.evaluate(X_dev, y_dev)

    print("DummyClassifier on the Development Set:")

    for strategy, f1 in dev_results.items():
        print(f"strategy='{strategy}' F1-score: {round(f1, 2)}")

    # Test set results
    best_model_f1_test = lr_model.evaluate(X_test, y_test)
    print("\nTest Set Results:")
    print(f"Improved logistic regression F1-score on test set: {round(best_model_f1_test, 2)}")

    best_knn_f1_test = knn_model.evaluate(X_test, y_test)
    print(f"Best KNN F1-score on test set: {round(best_knn_f1_test, 2)}")

    print("\nEvaluating DummyClassifiers on Test Set:")
    test_results = baseline_models.evaluate(X_test, y_test)
    for strategy, f1 in test_results.items():
        print(f"strategy='{strategy}' F1-score on test set: {round(f1, 2)}")

if __name__ == "__main__":
    main()
