
# Task 3 - Test 2-hidden-layer MLP on Pima Indian dataset

import numpy as np
import mlp2

# Load Pima Indians Diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
try:
    data = np.genfromtxt(url, delimiter=",")
except Exception:
    data = np.genfromtxt("pima-indians-diabetes.csv", delimiter=",")

X = data[:, :-1]
y = data[:, -1:].astype(int)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {np.sum(y == 0)} non-diabetic, {np.sum(y == 1)} diabetic")

# Normalise features to [0, 1]
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

# Shuffle
np.random.seed(42)
idx = np.random.permutation(len(X))
X = X[idx]
y = y[idx]

# Split: 60% train, 20% validation, 20% test
n = len(X)
train_end = int(0.6 * n)
valid_end = int(0.8 * n)

train = X[:train_end]
traintargets = y[:train_end]
valid = X[train_end:valid_end]
validtargets = y[train_end:valid_end]
test = X[valid_end:]
testtargets = y[valid_end:]

print(f"\nTrain: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

# Train 2-hidden-layer MLP
# nhidden1=8, nhidden2=4 — tapering architecture for 8-input binary classification
net = mlp2.mlp2(train, traintargets, nhidden1=8, nhidden2=4, outtype='logistic')
net.earlystopping(train, traintargets, valid, validtargets, eta=0.1)

# Evaluate
print("\n=== Test Results ===")
net.confmat(test, testtargets)
