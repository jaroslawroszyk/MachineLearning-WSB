import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import pandas as pd

data = fetch_kddcup99()
X, y = data.data, data.target
X = pd.DataFrame(X, columns=data.feature_names)
X = X.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
y = np.array([1 if label == b'normal.' else 0 for label in y])
X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

accuracy = mlp.score(X_test, y_test)
print(f'Dokładność MLP: {accuracy:.2f}')

print(f'Wagi: {mlp.coefs_}')
print(f'Obciążenie: {mlp.intercepts_}')

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42))
])

pipeline.fit(X_train, y_train)
pipeline_accuracy = pipeline.score(X_test, y_test)
print(f'Dokładność MLP z standaryzacją: {pipeline_accuracy:.2f}')
