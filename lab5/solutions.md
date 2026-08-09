# Lab 5 – Support Vector Machines

Run everything from inside `lab5/`:

```bash
../.venv/bin/python task1_svm_linear.py
../.venv/bin/python task2_perceptron.py
../.venv/bin/python task3_transform.py
../.venv/bin/python task4_kernel_trick.py
```

---

## Task 1 – Optimal separating line, support vectors, margin

Data from (8.32) / (8.33):

| Class | Points |
|---|---|
| Class 1 (+1) | (1,1), (1,2), (2,1) |
| Class 2 (−1) | (0,0), (1,0), (0,1) |

**Code:** `task1_svm_linear.py` → `task1_result.png`

### Answer

**Optimal separating line**

$$2x_1 + 2x_2 - 3 = 0 \quad\Longleftrightarrow\quad x_1 + x_2 = 1.5$$

so $\mathbf{w} = (2, 2)$, $b = -3$.

**Support vectors** — the three points that sit exactly on the margin lines:

| Point | Class | $\mathbf{w}\cdot\mathbf{x} + b$ |
|---|---|---|
| (1,1) | +1 | +1 |
| (1,0) | −1 | −1 |
| (0,1) | −1 | −1 |

The other three points, (1,2), (2,1) and (0,0), give $+3$, $+3$ and $-3$. They are further away than the margin, so they play no part — you could delete them and get the identical line.

**Margin**

$$M = \frac{2}{\|\mathbf{w}\|} = \frac{2}{2\sqrt{2}} = \frac{1}{\sqrt{2}} \approx 0.7071$$

The distance from the boundary to the nearest point is half of that, $\approx 0.3536$.

### Why this is the answer (by hand)

The closest the two classes ever get is between (1,1) and the segment joining (1,0) and (0,1). The nearest point on that segment is its midpoint (0.5, 0.5), and

$$\|(1,1) - (0.5,0.5)\| = \tfrac{1}{\sqrt{2}} \approx 0.7071$$

The maximum-margin boundary is the perpendicular bisector of that shortest connection: it passes through the midpoint (0.75, 0.75) with direction $\mathbf{w} \propto (1,1)$, which gives $x_1 + x_2 = 1.5$. This matches the SVM output exactly.

---

## Task 2 – Perceptron vs SVM on the same data

**Code:** `task2_perceptron.py` → `task2_result.png`

The perceptron was trained five times with different random initial weights (η = 0.25, 20 iterations).

| Run | Line found | Errors | Margin M |
|---|---|---|---|
| 0 | 0.5049·x₁ + 0.2715·x₂ − 0.7603 = 0 | 0 | 0.0563 |
| 1 | 0.4917·x₁ + 0.5220·x₂ − 0.7000 = 0 | 0 | 0.4964 |
| 2 | 0.4936·x₁ + 0.4526·x₂ − 0.5050 = 0 | 0 | 0.0339 |
| 3 | 0.5051·x₁ + 0.5208·x₂ − 0.7291 = 0 | 0 | 0.5742 |
| 4 | 0.0467·x₁ + 0.0047·x₂ − 0.0473 = 0 | 0 | 0.0241 |
| **SVM** | **2x₁ + 2x₂ − 3 = 0** | **0** | **0.7071** |

### Explaining the differences

**The lines are different, and the perceptron's line is not even unique.** Every run gives 100% training accuracy, but a different boundary each time. The SVM gives the same line every run, no matter how it is initialised.

**Why:** the two algorithms optimise different things.

- The **perceptron** only minimises misclassifications. Its update rule is driven by the error term $(y - t)$, so once every point is on the correct side the error is zero, the weights stop changing, and training halts. Any of the infinitely many separating lines is an acceptable stopping point — which one you land on depends on the random initial weights, the learning rate, and the order of the data.
- The **SVM** minimises $\tfrac{1}{2}\|\mathbf{w}\|^2$ subject to $y_i(\mathbf{w}\cdot\mathbf{x}_i + b) \ge 1$. Zero training error is only the *constraint*; the thing actually being maximised is the margin. That is a convex quadratic program with a single global optimum, so the answer is unique.

**The M's:** every perceptron margin (0.024 to 0.574) is smaller than the SVM's 0.7071, and by definition none can ever exceed it — 0.7071 is the maximum achievable. Some runs land almost on top of a data point (run 4, M = 0.024).

**Why that matters:** a small margin means poor generalisation. Run 4's boundary passes within 0.012 of a training point, so the tiniest amount of noise in a new test point flips its label. The SVM deliberately sits as far from both classes as geometry allows, which is why it is the more robust classifier even though both score identically on the training set.

---

## Task 3 – Making non-linear data linear with Φ₁

Data:

- **Positive:** (2,2), (2,−2), (−2,−2), (−2,2)
- **Negative:** (1,1), (1,−1), (−1,−1), (−1,1)

Transformation:

$$\Phi_1\begin{pmatrix}x_1\\x_2\end{pmatrix} = \begin{cases}\begin{pmatrix}4 - x_2 + |x_1 - x_2|\\ 4 - x_1 + |x_1 - x_2|\end{pmatrix} & \text{if } \sqrt{x_1^2 + x_2^2} > 2\\[2ex] \begin{pmatrix}x_1\\x_2\end{pmatrix} & \text{otherwise}\end{cases}$$

**Code:** `task3_transform.py` → `task3_result.png`

### Transformed data

The positives all have radius $\sqrt{8} \approx 2.83 > 2$, so they get the first branch. The negatives have radius $\sqrt{2} \approx 1.41 \le 2$, so they are left alone.

| Original (positive) | $\lvert x_1 - x_2\rvert$ | Transformed |
|---|---|---|
| (2, 2) | 0 | **(2, 2)** |
| (2, −2) | 4 | **(10, 6)** |
| (−2, −2) | 0 | **(6, 6)** |
| (−2, 2) | 4 | **(6, 10)** |

| Original (negative) | Transformed |
|---|---|
| (1, 1) | **(1, 1)** |
| (1, −1) | **(1, −1)** |
| (−1, −1) | **(−1, −1)** |
| (−1, 1) | **(−1, 1)** |

### Classification with a straight line

The transform folds all four positives into the upper-right region while the negatives stay in the box $[-1,1]^2$. A straight line now separates them:

$$z_1 + z_2 - 3 = 0 \quad\Longleftrightarrow\quad z_1 + z_2 = 3$$

- Support vectors: **(2,2)** from the positives and **(1,1)** from the negatives
- Margin: $M = \sqrt{2} \approx 1.4142$

Sanity check on $z_1 + z_2$: positives give 4, 16, 12, 16 (all > 3); negatives give 2, 0, −2, 0 (all < 3).

**The point of the exercise:** in the original space the positives sit at the corners of a big square and the negatives at the corners of a small one — no straight line can separate a ring from its centre. After Φ₁ the problem becomes trivially linear. This is the whole idea behind SVM kernels: change the space, not the classifier.

---

## Task 4 – Two circles and the Kernel Trick

**Code:** `task4_kernel_trick.py` → `task4_result.png`

### 4a – Data and transformation

10 points on each circle, using $x = (r\cos\theta,\ r\sin\theta)$ with θ evenly spaced over $[0, 2\pi)$:

- **Inner circle:** $r = 1$, label −1
- **Outer circle:** $r = 3$, label +1

Polynomial kernel $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x} \cdot \mathbf{z})^2$, whose feature map into 3D is

$$\phi(x_1, x_2) = (x_1^2,\ \sqrt{2}\,x_1x_2,\ x_2^2)$$

Verified numerically in the script: $\max |K(\mathbf{x},\mathbf{z}) - \phi(\mathbf{x})\cdot\phi(\mathbf{z})| = 2.8 \times 10^{-14}$, i.e. equal to floating-point precision. That equality *is* the kernel trick — you can compute the dot product in 3D without ever building the 3D vectors.

Examples: $(1, 0) \mapsto (1, 0, 0)$ and $(3, 0) \mapsto (9, 0, 0)$.

### Why this separates them

In the new space the first and third coordinates always sum to the squared radius:

$$z_1 + z_3 = x_1^2 + x_2^2 = r^2$$

So every inner point lands on the plane $z_1 + z_3 = 1$ and every outer point on $z_1 + z_3 = 9$. Two parallel planes — a flat plane between them separates the classes perfectly. In 2D the classes are concentric rings; in 3D they are two separated sheets.

### 4b – SVM on the transformed data (optional part)

Running a linear SVM in the 3D feature space:

| | |
|---|---|
| Separating plane | $0.25z_1 + 0z_2 + 0.25z_3 - 1.25 = 0$ |
| Support vectors | 8 |
| Margin | $M = 5.6569 = 4\sqrt{2}$ |
| Training accuracy | 1.00 |

The $z_2$ coefficient is 0, exactly as predicted — the $\sqrt{2}x_1x_2$ direction carries no information about which circle a point is on, so the SVM ignores it. Dividing through by 0.25 gives

$$z_1 + z_3 = 5 \quad\Longleftrightarrow\quad x_1^2 + x_2^2 = 5$$

Mapped back to the original 2D space, the plane becomes a **circle of radius $\sqrt{5} \approx 2.236$** sitting between the two rings — sensibly, midway in squared radius between $r^2 = 1$ and $r^2 = 9$. A linear boundary in the feature space is a non-linear (circular) boundary in the input space, which is exactly what was wanted.
