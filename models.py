"""
This script preprocesses, trains and evaluates a decision tree regressor and a random forest regressor
"""
# Import modules
import os
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

## Specify and run models

# Linear regression
linear = LinearRegression()
start = time.process_time()
linear.fit(X_train, y_train)
end = time.process_time()
linear_runtime = end - start
linear_predictions = linear.predict(X_test)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))

# Poisson regression
poisson = PoissonRegressor()
start = time.process_time()
poisson.fit(X_train, y_train)
end = time.process_time()
poisson_runtime = end - start
poisson_predictions = poisson.predict(X_test)
poisson_rmse = np.sqrt(mean_squared_error(y_test, poisson_predictions))

# Decision tree regressor
tree = DecisionTreeRegressor(random_state = 11)
start = time.process_time()
tree.fit(X_train, y_train)
end = time.process_time()
tree_runtime = end - start
tree_predictions = tree.predict(X_test)
tree_rmse = np.sqrt(mean_squared_error(y_test, tree_predictions))

# Random forest regressor
forest = RandomForestRegressor(random_state = 11)
start = time.process_time()
forest.fit(X_train, y_train)
end = time.process_time()
forest_runtime = end - start
forest_predictions = forest.predict(X_test)
forest_rmse = np.sqrt(mean_squared_error(y_test, forest_predictions))

# Linear SV regressor
support = LinearSVR(random_state = 11)
start = time.process_time()
support.fit(X_train, y_train)
end = time.process_time()
support_runtime = end - start
support_predictions = support.predict(X_test)
support_rmse = np.sqrt(mean_squared_error(y_test, support_predictions))

# Multi-layer perceptron regressor
perceptron = MLPRegressor(hidden_layer_sizes = (1), random_state = 11)
start = time.process_time()
perceptron.fit(X_train, y_train)
end = time.process_time()
perceptron_runtime = end - start
perceptron_predictions = perceptron.predict(X_test)
perceptron_rmse = np.sqrt(mean_squared_error(y_test, perceptron_predictions))

# Ensemble
ensemble = VotingRegressor([('linear', linear), ('forest', forest), ('support', support), ('perceptron', perceptron)])
start = time.process_time()
ensemble.fit(X_train, y_train)
end = time.process_time()
ensemble_runtime = end - start
ensemble_predictions = ensemble.predict(X_test)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_predictions))
