# Lab 5 Task 4 - Two concentric circles separated with a polynomial kernel
#
# 4a: map the 2D points into 3D with phi(x) = (x1^2, sqrt(2)*x1*x2, x2^2),
#     which is the feature map of the polynomial kernel K(x,z) = (x.z)^2
# 4b: run SVM on the transformed data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.svm import SVC

n = 10
theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

r_inner = 1.0
r_outer = 3.0
inner = np.array([r_inner * np.cos(theta), r_inner * np.sin(theta)]).T
outer = np.array([r_outer * np.cos(theta), r_outer * np.sin(theta)]).T

X = np.concatenate((inner, outer))
y = np.concatenate((-np.ones(n), np.ones(n)))


def poly_map(points):
    """Feature map of K(x,z) = (x.z)^2"""
    x1 = points[:, 0]
    x2 = points[:, 1]
    return np.array([x1 ** 2, np.sqrt(2) * x1 * x2, x2 ** 2]).T


Z = poly_map(X)
print("First inner point %s -> %s" % (X[0], Z[0]))
print("First outer point %s -> %s" % (X[n], Z[n]))

# Verify the feature map reproduces the kernel
K_direct = np.dot(X, X.T) ** 2
K_mapped = np.dot(Z, Z.T)
print("max |K(x,z) - phi(x).phi(z)| = %.2e" % np.abs(K_direct - K_mapped).max())

# 4b - SVM in the 3D feature space
svm = SVC(kernel='linear', C=1e6)
svm.fit(Z, y)
w = svm.coef_[0]
b = svm.intercept_[0]
margin = 2.0 / np.linalg.norm(w)
print("Separating plane: %.3f*z1 + %.3f*z2 + %.3f*z3 + %.3f = 0" % (w[0], w[1], w[2], b))
print("Number of support vectors:", len(svm.support_))
print("Margin M = %.4f" % margin)
print("Training accuracy: %.2f" % svm.score(Z, y))

fig = plt.figure(figsize=(13, 6))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(inner[:, 0], inner[:, 1], 'rs', markersize=9, label='Inner circle (-1)')
ax1.plot(outer[:, 0], outer[:, 1], 'bo', markersize=9, label='Outer circle (+1)')
ax1.set_xlim(-4, 4)
ax1.set_ylim(-4, 4)
ax1.set_aspect('equal')
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('Original 2D data (no straight line works)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(Z[:n, 0], Z[:n, 1], Z[:n, 2], c='r', marker='s', s=60,
            label='Inner circle (-1)')
ax2.scatter(Z[n:, 0], Z[n:, 1], Z[n:, 2], c='b', marker='o', s=60,
            label='Outer circle (+1)')

# Separating plane found by the SVM
g1, g2 = np.meshgrid(np.linspace(0, 9, 10), np.linspace(-9, 9, 10))
g3 = -(w[0] * g1 + w[1] * g2 + b) / w[2]
ax2.plot_surface(g1, g2, g3, alpha=0.25, color='gray')

ax2.set_xlim(0, 9)
ax2.set_ylim(-9, 9)
ax2.set_zlim(0, 9)
ax2.view_init(elev=18, azim=-72)
ax2.set_xlabel('$x_1^2$')
ax2.set_ylabel('$\\sqrt{2}x_1x_2$')
ax2.set_zlabel('$x_2^2$')
ax2.set_title('After polynomial kernel map (M = %.3f)' % margin)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig('task4_result.png', dpi=150)
print("Saved task4_result.png")
