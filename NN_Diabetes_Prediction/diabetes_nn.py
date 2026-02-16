"""
Diabetes Prediction using Scikit-Learn Neural Network
=====================================================
This script builds a Multi-Layer Perceptron (Neural Network) to predict diabetes.
It performs the following steps:
1. Load dataset (Pima Indians Diabetes)
2. Preprocess data (Normalize features, Split X/y)
3. Split into Training and Testing sets
4. Build and Train a Neural Network (MLPClassifier)
5. Evaluate Model Accuracy
6. Save the Training Dataset (normalized) to a JSON file

NOTE: This script does NOT use any camera or image processing libraries.
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("DIABETES PREDICTION PROJECT (Neural Network)")
print("=" * 60)

# ------------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------------
print("\n[1/7] Loading Dataset...")
# Define column names based on dataset description
columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
file_path = 'pima-indians-diabetes.csv'

try:
    df = pd.read_csv(file_path, header=None, names=columns)
    print("✓ Dataset loaded successfully.")
    print(f"  Shape: {df.shape}")
except FileNotFoundError:
    print(f"❌ Error: File '{file_path}' not found.")
    exit()

# ------------------------------------------------------------------
# 2. PREPROCESS DATA
# ------------------------------------------------------------------
print("\n[2/7] Preprocessing Data...")

# Separate Features (X) and Target (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# ------------------------------------------------------------------
# 3. SPLIT DATASET
# ------------------------------------------------------------------
print("\n[3/7] Splitting Dataset...")
# Split into training (80%) and testing (20%) sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print(f"✓ Training Set: {X_train.shape[0]} samples")
print(f"✓ Testing Set:  {X_test.shape[0]} samples")

# ------------------------------------------------------------------
# 4. BUILD NEURAL NETWORK MODEL
# ------------------------------------------------------------------
print("\n[4/7] Building Neural Network...")

# MLPClassifier is a Neural Network
model = MLPClassifier(
    hidden_layer_sizes=(12, 8),  # Two hidden layers with 12 and 8 neurons
    activation='relu',           # ReLU activation
    solver='adam',               # Optimizer
    max_iter=500,                # Epochs
    random_state=42,
    verbose=False
)

print("✓ Model Architecture: Input(8) -> Hidden(12) -> Hidden(8) -> Output(1)")

# ------------------------------------------------------------------
# 5. TRAIN MODEL
# ------------------------------------------------------------------
print("\n[5/7] Training Model...")
model.fit(X_train, y_train)
print("✓ Training Completed.")

# ------------------------------------------------------------------
# 6. EVALUATE MODEL
# ------------------------------------------------------------------
print("\n[6/7] Evaluating Model...")
# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✓ Model Accuracy on Test Set: {accuracy*100:.2f}%")

# ------------------------------------------------------------------
# 7. SAVE TRAINING DATA TO JSON
# ------------------------------------------------------------------
print("\n[7/7] Saving Training Data to JSON...")

training_data = {
    "metadata": {
        "description": "Training dataset for Diabetes Prediction",
        "feature_names": columns[:-1],
        "target_name": columns[-1],
        "total_samples": len(X_train),
        "normalized": True
    },
    "data": []
}

for i in range(len(X_train)):
    record = {
        "features": X_train[i].tolist(),
        "target": int(y_train[i])
    }
    training_data["data"].append(record)

json_filename = 'diabetes_training_data.json'
with open(json_filename, 'w') as json_file:
    json.dump(training_data, json_file, indent=4)

print(f"✓ Training data saved to '{json_filename}'")
print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)
