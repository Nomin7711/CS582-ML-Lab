# Lab 5 Task 3 - Make non-linear data linear with the transformation Phi_1
#
# Phi_1(x1,x2) = (4 - x2 + |x1-x2|, 4 - x1 + |x1-x2|)   if sqrt(x1^2 + x2^2) > 2
#              = (x1, x2)                                otherwise

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

positive = np.array([[2., 2.], [2., -2.], [-2., -2.], [-2., 2.]])
negative = np.array([[1., 1.], [1., -1.], [-1., -1.], [-1., 1.]])


def phi1(points):
    out = np.zeros(np.shape(points))
    for i in range(np.shape(points)[0]):
        x1, x2 = points[i]
        if np.sqrt(x1 ** 2 + x2 ** 2) > 2:
            out[i] = [4 - x2 + abs(x1 - x2), 4 - x1 + abs(x1 - x2)]
        else:
            out[i] = [x1, x2]
    return out


tpos = phi1(positive)
tneg = phi1(negative)

print("Transformed positive points:")
print(tpos)
print("Transformed negative points:")
print(tneg)

X = np.concatenate((tpos, tneg))
y = np.concatenate((np.ones(len(tpos)), -np.ones(len(tneg))))

svm = SVC(kernel='linear', C=1e6)
svm.fit(X, y)
w = svm.coef_[0]
b = svm.intercept_[0]
margin = 2.0 / np.linalg.norm(w)
print("Separating line in transformed space: %.3f*z1 + %.3f*z2 + %.3f = 0" % (w[0], w[1], b))
print("Support vectors:")
print(svm.support_vectors_)
print("Margin M = %.4f" % margin)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].plot(positive[:, 0], positive[:, 1], 'bo', markersize=9, label='Positive')
ax[0].plot(negative[:, 0], negative[:, 1], 'rs', markersize=9, label='Negative')
ax[0].set_xlim(-4, 4)
ax[0].set_ylim(-4, 4)
ax[0].set_aspect('equal')
ax[0].set_xlabel('$x_1$')
ax[0].set_ylabel('$x_2$')
ax[0].set_title('Original data (not linearly separable)')
ax[0].legend(loc='upper right')
ax[0].grid(True, alpha=0.3)

z1 = np.linspace(-2, 12, 100)
boundary = -(w[0] * z1 + b) / w[1]
upper = -(w[0] * z1 + b - 1) / w[1]
lower = -(w[0] * z1 + b + 1) / w[1]

ax[1].plot(tpos[:, 0], tpos[:, 1], 'bo', markersize=9, label='Positive')
ax[1].plot(tneg[:, 0], tneg[:, 1], 'rs', markersize=9, label='Negative')
ax[1].plot(z1, boundary, 'k-', linewidth=2, label='Separating line')
ax[1].plot(z1, upper, 'k--', linewidth=1)
ax[1].plot(z1, lower, 'k--', linewidth=1)
ax[1].set_xlim(-2, 12)
ax[1].set_ylim(-2, 12)
ax[1].set_aspect('equal')
ax[1].set_xlabel('$z_1$')
ax[1].set_ylabel('$z_2$')
ax[1].set_title('After $\\Phi_1$ (linearly separable, M = %.3f)' % margin)
ax[1].legend(loc='upper right')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task3_result.png', dpi=150)
print("Saved task3_result.png")
