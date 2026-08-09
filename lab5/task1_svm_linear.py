# Lab 5 Task 1 - Optimal separating line for the two classes in (8.32) / (8.33)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

class1 = np.array([[1., 1.], [1., 2.], [2., 1.]])
class2 = np.array([[0., 0.], [1., 0.], [0., 1.]])

X = np.concatenate((class1, class2))
y = np.concatenate((np.ones(len(class1)), -np.ones(len(class2))))

# Large C -> hard margin, the data is linearly separable
svm = SVC(kernel='linear', C=1e6)
svm.fit(X, y)

w = svm.coef_[0]
b = svm.intercept_[0]
margin = 2.0 / np.linalg.norm(w)

print("w =", w)
print("b =", b)
print("Decision boundary: %.3f*x1 + %.3f*x2 + %.3f = 0" % (w[0], w[1], b))
print("Support vectors:")
print(svm.support_vectors_)
print("Margin M = 2/||w|| = %.4f" % margin)
print("Half margin (boundary to nearest point) = %.4f" % (margin / 2))

x1 = np.linspace(-0.5, 2.5, 100)
boundary = -(w[0] * x1 + b) / w[1]
upper = -(w[0] * x1 + b - 1) / w[1]
lower = -(w[0] * x1 + b + 1) / w[1]

plt.figure(figsize=(6, 6))
plt.plot(class1[:, 0], class1[:, 1], 'bo', markersize=9, label='Class 1 (+1)')
plt.plot(class2[:, 0], class2[:, 1], 'rs', markersize=9, label='Class 2 (-1)')
plt.plot(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], 'ko',
         markersize=16, markerfacecolor='none', label='Support vectors')
plt.plot(x1, boundary, 'k-', label='Optimal separating line')
plt.plot(x1, upper, 'k--', linewidth=1)
plt.plot(x1, lower, 'k--', linewidth=1)
plt.xlim(-0.5, 2.5)
plt.ylim(-0.5, 2.5)
plt.gca().set_aspect('equal')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Task 1 - Maximum Margin Classifier (M = %.4f)' % margin)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.savefig('task1_result.png', dpi=150)
print("Saved task1_result.png")
