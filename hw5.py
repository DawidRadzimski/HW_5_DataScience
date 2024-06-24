import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Function to load data from directory
def load_data(directory):
    data = []
    labels = []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(label_dir, file)
                    df = pd.read_csv(file_path, sep=',')
                    data.append(df)
                    labels.extend([label] * len(df))
    return pd.concat(data, ignore_index=True), pd.Series(labels)

# Load the dataset
data_dir = 'data'
data_df, labels_df = load_data(data_dir)

# Verify lengths
print(f"Length of data: {len(data_df)}")
print(f"Length of labels: {len(labels_df)}")

# Data preprocessing
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_df)

# Verify shape after scaling
print(f"Shape of scaled data: {data_scaled.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels_df, test_size=0.2, random_state=42)

# Verify split lengths
print(f"Length of X_train: {len(X_train)}")
print(f"Length of X_test: {len(X_test)}")
print(f"Length of y_train: {len(y_train)}")
print(f"Length of y_test: {len(y_test)}")

# Function to cross-validate a model
def cross_validate_model(model, X, y, cv=3):
    print(f"Starting cross-validation for {model.__class__.__name__}")
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"Cross-validation scores for {model.__class__.__name__}: {scores}")
    print(f"Mean cross-validation score for {model.__class__.__name__}: {np.mean(scores)}")

# Perform cross-validation for SVM
print("SVM Cross-Validation")
svm = SVC()
cross_validate_model(svm, data_scaled, labels_df)

# Perform cross-validation for Random Forest
print("Random Forest Cross-Validation")
rf = RandomForestClassifier()
cross_validate_model(rf, data_scaled, labels_df)

# Train and evaluate SVM model
print("Training SVM model")
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM Model")
print(classification_report(y_test, y_pred_svm))
print("Accuracy:", accuracy_score(y_test, y_pred_svm))

# Train and evaluate Random Forest model
print("Training Random Forest model")
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Model")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
