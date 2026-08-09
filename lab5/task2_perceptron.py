# Lab 5 Task 2 - Perceptron on the Task 1 data, compared against the SVM line

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

np.random.seed(42)

class1 = np.array([[1., 1.], [1., 2.], [2., 1.]])
class2 = np.array([[0., 0.], [1., 0.], [0., 1.]])

X = np.concatenate((class1, class2))
y = np.concatenate((np.ones(len(class1)), np.zeros(len(class2))))


class pcn:
    """Perceptron (Marsland, Chapter 3)"""

    def __init__(self, inputs, targets):
        self.nIn = np.shape(inputs)[1]
        self.nOut = 1
        self.nData = np.shape(inputs)[0]
        self.weights = np.random.rand(self.nIn + 1, self.nOut) * 0.1 - 0.05

    def pcntrain(self, inputs, targets, eta, nIterations):
        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)
        for n in range(nIterations):
            self.activations = self.pcnfwd(inputs)
            self.weights -= eta * np.dot(np.transpose(inputs),
                                         self.activations - targets)
        return self.weights

    def pcnfwd(self, inputs):
        activations = np.dot(inputs, self.weights)
        return np.where(activations > 0, 1, 0)


targets = y.reshape(len(y), 1)
inputs = np.concatenate((X, -np.ones((len(X), 1))), axis=1)

# The perceptron stops as soon as it separates the data, so the line it lands on
# depends on the random initial weights. Train it several times to show this.
weights = []
margins = []
for seed in range(5):
    np.random.seed(seed)
    p = pcn(X, targets)
    p.pcntrain(X, targets, 0.25, 20)
    wp = p.weights[:, 0]

    errors = np.sum(p.pcnfwd(inputs)[:, 0] != y)
    # Perceptron margin: twice the distance from the boundary to the closest point
    dist = np.abs(np.dot(X, wp[:2]) - wp[2]) / np.linalg.norm(wp[:2])
    m = 2 * dist.min()

    print("Run %d: %.4f*x1 + %.4f*x2 - %.4f = 0   errors = %d   M = %.4f"
          % (seed, wp[0], wp[1], wp[2], errors, m))
    weights.append(wp)
    margins.append(m)

print("Perceptron margins range from %.4f to %.4f" % (min(margins), max(margins)))

ysvm = np.where(y > 0, 1., -1.)
svm = SVC(kernel='linear', C=1e6)
svm.fit(X, ysvm)
ws = svm.coef_[0]
bs = svm.intercept_[0]
svm_margin = 2.0 / np.linalg.norm(ws)
print("SVM line: %.4f*x1 + %.4f*x2 + %.4f = 0" % (ws[0], ws[1], bs))
print("SVM margin M = %.4f" % svm_margin)

x1 = np.linspace(-0.5, 2.5, 100)
svm_line = -(ws[0] * x1 + bs) / ws[1]
svm_upper = -(ws[0] * x1 + bs - 1) / ws[1]
svm_lower = -(ws[0] * x1 + bs + 1) / ws[1]

plt.figure(figsize=(6, 6))
plt.plot(class1[:, 0], class1[:, 1], 'bo', markersize=9, label='Class 1 (+1)')
plt.plot(class2[:, 0], class2[:, 1], 'rs', markersize=9, label='Class 2 (0/-1)')
for i in range(len(weights)):
    wp = weights[i]
    plt.plot(x1, (wp[2] - wp[0] * x1) / wp[1], 'g-', linewidth=1.5, alpha=0.7,
             label='Perceptron runs (M = %.3f..%.3f)' % (min(margins), max(margins))
             if i == 0 else None)
plt.plot(x1, svm_line, 'k-', linewidth=2, label='SVM (M = %.3f)' % svm_margin)
plt.plot(x1, svm_upper, 'k--', linewidth=1)
plt.plot(x1, svm_lower, 'k--', linewidth=1)
plt.xlim(-0.5, 2.5)
plt.ylim(-0.5, 2.5)
plt.gca().set_aspect('equal')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Task 2 - Perceptron vs SVM Decision Boundary')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.savefig('task2_result.png', dpi=150)
print("Saved task2_result.png")
