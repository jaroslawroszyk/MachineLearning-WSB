import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
import pandas as pd

data = fetch_kddcup99()
X, y = data.data, data.target
X = pd.DataFrame(X, columns=data.feature_names)
X = X.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
y = np.array([1 if label == b'normal.' else 0 for label in y])
X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

perceptron = Perceptron()
perceptron.fit(X_train, y_train)

accuracy = perceptron.score(X_test, y_test)
print(f'Dokładność perceptronu: {accuracy:.2f}')

print(f'Wagi: {perceptron.coef_}')
print(f'Obciążenie: {perceptron.intercept_}')

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('perceptron', Perceptron())
])

pipeline.fit(X_train, y_train)
pipeline_accuracy = pipeline.score(X_test, y_test)
print(f'Dokładność perceptronu z standaryzacją: {pipeline_accuracy:.2f}')
