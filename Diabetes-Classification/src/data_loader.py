import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def load_and_split_data(file_path):
    diabetes_df = pd.read_csv(file_path)
    
    # Separate majority and minority classes
    majority_class = diabetes_df[diabetes_df.Outcome == 0]
    minority_class = diabetes_df[diabetes_df.Outcome == 1]
    
    # Downsample majority class to match minority class size
    majority_downsampled = resample(
        majority_class,
        replace=False,
        n_samples=len(minority_class),
        random_state=42
    )
    
    balanced_df = pd.concat([majority_downsampled, minority_class])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split the data into 70% training, 15% development, and 15% test sets
    train_df, temp_df = train_test_split(balanced_df, test_size=0.30, random_state=42)
    dev_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    
    # Separate features and labels
    X_train = train_df.drop('Outcome', axis=1)
    y_train = train_df['Outcome']
    X_dev = dev_df.drop('Outcome', axis=1)
    y_dev = dev_df['Outcome']
    X_test = test_df.drop('Outcome', axis=1)
    y_test = test_df['Outcome']
    
    return X_train, y_train, X_dev, y_dev, X_test, y_test
