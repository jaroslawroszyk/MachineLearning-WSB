from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

wine_data = load_wine(as_frame=True, return_X_y=True)
X,y = wine_data

y = (y == 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

perceptron = Perceptron()
perceptron.fit(X_train, y_train)


accuracy = perceptron.score(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')

print(f'Wagi: {perceptron.coef_}')
print(f'Obciążenie: {perceptron.intercept_}')

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('perceptron', Perceptron())
])

pipeline.fit(X_train, y_train)
pipeline_accuracy = pipeline.score(X_test, y_test)
print(f'Dokładność perceptronu z standaryzacją: {pipeline_accuracy:.2f}')

# G)
feature1, feature2 = 0, 1 
X_selected = X.iloc[:, [feature1, feature2]]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

perceptron_2d = Perceptron()
perceptron_2d.fit(X_train_scaled, y_train)

x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = perceptron_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, edgecolors='k')
plt.xlabel(f'Cecha {feature1}')
plt.ylabel(f'Cecha {feature2}')
plt.title('Granica decyzyjna perceptronu')
plt.show()
