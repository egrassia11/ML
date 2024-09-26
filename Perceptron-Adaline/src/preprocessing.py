import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_and_save_titanic_data(file_path='./data/titanic_data.csv', output_train='./data/train.csv', output_test='./data/test.csv', output_test_labels='./data/test_labels.csv'):
    data = pd.read_csv(file_path)

    # Dropping irrelevant features
    data_cleaned = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # Converting non-numeric 'Sex' and 'Embarked' columns to numeric
    data_cleaned['Sex'] = data_cleaned['Sex'].map({'male': 0, 'female': 1})
    data_cleaned['Embarked'] = data_cleaned['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Filling missing values in 'Age' and 'Embarked' with their median values
    data_cleaned['Age'].fillna(data_cleaned['Age'].median(), inplace=True)
    data_cleaned['Embarked'].fillna(data_cleaned['Embarked'].median(), inplace=True)

    X = data_cleaned.drop('Survived', axis=1)
    y = data_cleaned['Survived']

    # Splitting the data into 70% training and 30% testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    train_data = pd.concat([X_train, y_train], axis=1)
    
    train_data.to_csv(output_train, index=False)

    X_test.to_csv(output_test, index=False)
    y_test.to_csv(output_test_labels, index=False)

    print(f"Training data saved to {output_train}")
    print(f"Testing data (without labels) saved to {output_test}")
    print(f"Test labels saved to {output_test_labels}")

preprocess_and_save_titanic_data()
