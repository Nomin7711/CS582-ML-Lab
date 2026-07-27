from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# 1. Load the Data
data = np.loadtxt(Path(__file__).with_name('PNOz.dat'))
ozone_data = data[:, 2]

# 2. Format the Inputs (Sliding Window)
def create_time_series_dataset(series, k, tau):
    X, y = [], []
    window_span = k * tau
    
    for i in range(window_span, len(series)):
        inputs = series[i - window_span : i : tau]
        X.append(inputs)
        y.append(series[i])
        
    return np.array(X), np.array(y)

k = 3
tau = 2
X, y = create_time_series_dataset(ozone_data, k, tau)

# 3. Split the Data (Chronological split for time-series)
train_size = len(X) - 800
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. Normalize the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the MLP (Regression problem -> Linear output)
mlp = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', 
                   solver='adam', max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)

# 6. Evaluate the MLP
test_score = mlp.score(X_test_scaled, y_test)
print(f"Test R^2 Score: {test_score:.4f}")