"""
Sign Language Recognition - ALPHABET Model Training Script
================================================================
Idhu script, A-Z alphabet dataset ah padichu, oru separate model train pannum
(spelling mode ku, word model vera, idhu vera).
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

CSV_FILE = "dataset/alphabet_data.csv"
MODEL_SAVE_PATH = "model/alphabet_model.pkl"

os.makedirs("model", exist_ok=True)

print("Alphabet dataset load pannurom...")
df = pd.read_csv(CSV_FILE)
print(f"Total samples: {len(df)}")
print(f"Letters: {sorted(df['label'].unique())}")
print(f"Samples per letter:\n{df['label'].value_counts().sort_index()}")

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

print("\nModel training start aaguthu...")
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"Alphabet Model Accuracy: {accuracy * 100:.2f}%")
print(f"{'='*50}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, zero_division=0))

with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"\nModel save aagiduchu: {MODEL_SAVE_PATH}")
