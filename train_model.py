# ==============================
# FILE: train_model.py
# ==============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# load dataset
data = pd.read_csv("asl_data.csv", header=None)

# fitur dan label
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# split train test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# model AI
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

# train
model.fit(X_train, y_train)

# test accuracy
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)

# save model
with open("asl_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model berhasil disimpan")
