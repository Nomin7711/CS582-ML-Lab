
# Task 4 - Test recurrent MLP on Palmerston North Ozone data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mlp_rnn

PNoz = np.loadtxt('PNOz.dat')

# Normalise ozone column
PNoz[:, 2] = PNoz[:, 2] - PNoz[:, 2].mean()
PNoz[:, 2] = PNoz[:, 2] / PNoz[:, 2].max()

# Assemble input vectors (same as Task 1 for fair comparison)
t = 2
k = 3

lastPoint = np.shape(PNoz)[0] - t*(k+1)
inputs = np.zeros((lastPoint, k))
targets = np.zeros((lastPoint, 1))
for i in range(lastPoint):
    inputs[i, :] = PNoz[i:i+t*k:t, 2]
    targets[i] = PNoz[i+t*(k+1), 2]

# Split — keep test set in sequential order (important for recurrent network)
test = inputs[-400:]
testtargets = targets[-400:]
train = inputs[:-400:2]
traintargets = targets[:-400:2]
valid = inputs[1:-400:2]
validtargets = targets[1:-400:2]

print(f"Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

# Train recurrent MLP
np.random.seed(42)
net = mlp_rnn.mlp_rnn(train, traintargets, nhidden=3, outtype='linear')
net.earlystopping(train, traintargets, valid, validtargets, eta=0.25)

# Predict on test set
testout = net.predict(test)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(test)), testout, '.', label='RNN Predictions')
plt.plot(np.arange(len(test)), testtargets, 'x', label='Targets')
plt.legend()
plt.title('Recurrent MLP Predictions vs Targets (PNOz)')
plt.xlabel('Test Sample')
plt.ylabel('Normalised Ozone')
plt.tight_layout()
plt.savefig('fig_rnn_pnoz.png', dpi=150)
print("Saved fig_rnn_pnoz.png")

test_error = 0.5*np.sum((testtargets - testout)**2)
print(f"Test error: {test_error}")
