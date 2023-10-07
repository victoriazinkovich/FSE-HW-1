import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --- Hyberbola --- 
# Actual Hyperbola
np.random.seed(0)
X = np.sort(10 * np.random.rand(100, 1) + 1, axis=0) 
y = hyperbola(X.ravel())

# Model Fitting
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X, y)

# Estimate Accuracy
X_test = np.arange(1, 11, 0.1)[:, np.newaxis]
y_pred = regressor.predict(X_test)
y_true = hyperbola(X_test.ravel())
mse = mean_squared_error(y_true, y_pred)
print("Mean Squared Error (MSE):", mse)


# --- Polynom_3 --- 
# Actual Polynom_3
X = np.sort(10 * np.random.rand(100, 1) + 1, axis=0) 
y = polynom_3(X.ravel())

# Model Fitting
p_regressor = RandomForestRegressor(n_estimators=100, random_state=0)
p_regressor.fit(X, y)

# Estimate Accuracy
X_test = np.arange(1, 11, 0.1)[:, np.newaxis]
y_pred = p_regressor.predict(X_test)
y_true = polynom_3(X_test.ravel())
mse = mean_squared_error(y_true, y_pred)
print("Mean Squared Error (MSE):", mse)
