import json
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sensor import SensorArray

# 1. Load sample dataset
with open("samples.json", "r") as f:
    samples = json.load(f)

sensor_array = SensorArray()

X = []
y = []

# 2. Generate 200 synthetic sniffs per aroma sample
sniffs_per_sample = 200

for label, data in samples.items():
    smiles_list = data["smiles"]
    for _ in range(sniffs_per_sample):
        # Apply random noise and humidity variance per sniff pass
        noise = np.random.uniform(0.02, 0.08)
        humidity = np.random.uniform(0.8, 1.2)
        signal = sensor_array.sniff(smiles_list, noise_level=noise, humidity_factor=humidity)
        X.append(signal)
        y.append(label)

X = np.array(X)
y = np.array(y)

# 3. Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluate Performance
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Electronic Nose Training Complete!")
print(f"Total Synthetic Sniffs: {len(X)}")
print(f"Test Accuracy: {acc * 100:.2f}%")

# 6. Save trained nose model
with open("nose_model.pkl", "wb") as f:
    pickle.dump(clf, f)
print("Saved trained model to 'nose_model.pkl'.")