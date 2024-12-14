import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('/content/ckd_dataset.csv')
print(df.head())

print(df.info())

# Check for missing values
print(df.isnull().sum())

# Display basic statistics of the data
print(df.describe())

df_numeric = df.select_dtypes(include='number')
df[df_numeric.columns] = df_numeric.fillna(df_numeric.mean())

# Handle non-numeric columns (e.g., filling with the mode)
for col in df.select_dtypes(exclude='number').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print(df.isnull().sum())
# Replace 'class' with the actual target column name
X = df.drop('class', axis=1)  # Features
y = df['class']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check the shapes of the datasets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print(X_train.dtypes)

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Check the shape to ensure both sets have the same columns
print(X_train.shape, X_test.shape)

print(df.columns)
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('/content/ckd_dataset.csv')

# Identify categorical columns
categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class']

# Label encode binary or ordinal categorical columns (those with two values)
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

print(df.head())

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)  # Start with k=5
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Test different values of k
accuracies = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_k = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred_k))

# Plot accuracy vs. k
plt.plot(range(1, 21), accuracies, marker='o')
plt.title('Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.show()

import joblib

# Save the model to a file
joblib.dump(knn, 'knn_ckd_model.pkl')

# Load the model back later
# knn = joblib.load('knn_ckd_model.pkl')

import pickle

# Specify the path to your .pkl file
file_path = '/content/knn_ckd_model.pkl'
# Open and load the .pkl file
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Display the contents of the .pkl file
print(data)