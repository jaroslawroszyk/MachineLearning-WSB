import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_kddcup99, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd

data_liver = fetch_openml(data_id=8, as_frame=True, return_X_y=True)
X_liver, y_liver = data_liver

X_train_liver, X_test_liver, y_train_liver, y_test_liver = train_test_split(X_liver, y_liver, test_size=0.2, random_state=42)

mlp_regressor = MLPRegressor(hidden_layer_sizes=(10,), max_iter=500, random_state=42)
mlp_regressor.fit(X_train_liver, y_train_liver)

y_pred_mlp = mlp_regressor.predict(X_test_liver)
mae_mlp = mean_absolute_error(y_test_liver, y_pred_mlp)
print(f'MAE MLPRegressor: {mae_mlp:.2f}')

print(f'Wagi MLPRegressor: {mlp_regressor.coefs_}')
print(f'Obciążenie MLPRegressor: {mlp_regressor.intercepts_}')

pipeline_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(10,), max_iter=500, random_state=42))
])

pipeline_reg.fit(X_train_liver, y_train_liver)
y_pred_pipeline = pipeline_reg.predict(X_test_liver)
mae_pipeline = mean_absolute_error(y_test_liver, y_pred_pipeline)
print(f'MAE MLPRegressor z standaryzacją: {mae_pipeline:.2f}')

lin_reg = LinearRegression()
lin_reg.fit(X_train_liver, y_train_liver)
y_pred_lin = lin_reg.predict(X_test_liver)
mae_lin = mean_absolute_error(y_test_liver, y_pred_lin)
print(f'MAE LinearRegression: {mae_lin:.2f}')
