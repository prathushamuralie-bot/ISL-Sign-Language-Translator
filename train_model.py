"""
Sign Language Recognition - Model Training Script
====================================================
Idhu script, dataset/sign_data.csv la irukura landmark data ah padichu,
oru ML model (Random Forest) train pannum, apparam adha save pannum.

Epadi use pannuradhu:
1. Data collection mudinja apparam, idha run pannunga
2. Model train aagi, "model/sign_model.pkl" nu save aagum
3. Accuracy score screen la kaanum
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

CSV_FILE = "dataset/sign_data_two_hands.csv"
MODEL_SAVE_PATH = "model/sign_model.pkl"

os.makedirs("model", exist_ok=True)

# Data load pannurom
print("Dataset load pannurom...")
df = pd.read_csv(CSV_FILE)
print(f"Total samples: {len(df)}")
print(f"Signs: {df['label'].unique()}")
print(f"Samples per sign:\n{df['label'].value_counts()}")

# Features (landmarks) and labels (sign names) separate pannurom
X = df.drop('label', axis=1)
y = df['label']

# Train and test data split pannurom (80% train, 20% test)
# NOTE: samples romba kammi na (10-15 per sign), test_size konjam adjust pannalam
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Random Forest model train pannurom
print("\nModel training start aaguthu...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test data la evaluate pannurom
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"{'='*50}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# Model save pannurom
with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"\nModel save aagiduchu: {MODEL_SAVE_PATH}")
print("Ippo 'live_test.py' run panni, real-time la test pannalam!")
