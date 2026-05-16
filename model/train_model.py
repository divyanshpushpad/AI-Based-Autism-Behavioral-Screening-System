import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('dataset/autism_dataset.csv')

# Select ONLY required columns
selected_features = [
    'A1_Score',
    'A2_Score',
    'A3_Score',
    'A4_Score',
    'A5_Score',
    'A6_Score',
    'A7_Score',
    'A8_Score',
    'A9_Score',
    'A10_Score',
    'age',
    'Class/ASD'
]

data = data[selected_features]

print(data['Class/ASD'].value_counts())
print(data['Class/ASD'].unique())

# Features and target
X = data.drop('Class/ASD', axis=1)
y = data['Class/ASD']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:")
print(accuracy * 100)

# Save model
joblib.dump(model, 'model/autism_model.pkl')

print("Model saved successfully.")