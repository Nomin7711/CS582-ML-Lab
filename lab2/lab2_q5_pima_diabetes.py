import numpy as np
import matplotlib.pyplot as plt

# Load the Pima Indians Diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
try:
    data = np.genfromtxt(url, delimiter=",")
except Exception:
    data = np.genfromtxt("pima-indians-diabetes.csv", delimiter=",")

feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigree", "Age"]

X = data[:, :-1]
y = data[:, -1].astype(int)

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.sum(y == 0)} non-diabetic, {np.sum(y == 1)} diabetic")

# Plot feature distributions
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, (ax, name) in enumerate(zip(axes.flat, feature_names)):
    ax.hist(X[y == 0, i], bins=20, alpha=0.6, label="Non-diabetic", color="steelblue")
    ax.hist(X[y == 1, i], bins=20, alpha=0.6, label="Diabetic", color="coral")
    ax.set_title(name)
    ax.legend(fontsize=8)
fig.suptitle("Feature Distributions by Class", fontsize=14)
plt.tight_layout()
plt.show()

# Normalize and split
X_min, X_max = X.min(axis=0), X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min + 1e-8)

np.random.seed(42)
idx = np.random.permutation(len(X_norm))
split = int(0.8 * len(X_norm))
X_train, X_test = X_norm[idx[:split]], X_norm[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")


def perceptron_train(X, targets, mode="sequential", learning_rate=0.25, max_epochs=200):
    n, f = X.shape
    Xb = np.concatenate([X, -np.ones((n, 1))], axis=1)
    w = np.random.uniform(-0.1, 0.1, f + 1)
    hist = []
    for _ in range(max_epochs):
        if mode == "batch":
            a = np.where(Xb @ w > 0, 1, 0)
            e = targets - a
            ne = np.sum(np.abs(e))
            hist.append(ne)
            if ne == 0:
                break
            w += learning_rate * (e @ Xb)
        else:
            ne = 0
            for i in range(n):
                a = 1 if Xb[i] @ w > 0 else 0
                err = targets[i] - a
                if err != 0:
                    ne += 1
                    w += learning_rate * err * Xb[i]
            hist.append(ne)
            if ne == 0:
                break
    return w, hist


def perceptron_predict(X, w):
    Xb = np.concatenate([X, -np.ones((X.shape[0], 1))], axis=1)
    return np.where(Xb @ w > 0, 1, 0)


# Train both versions
np.random.seed(42)
w_batch, err_batch = perceptron_train(X_train, y_train, mode="batch")
np.random.seed(42)
w_seq, err_seq = perceptron_train(X_train, y_train, mode="sequential")

# Evaluate
pred_batch = perceptron_predict(X_test, w_batch)
pred_seq = perceptron_predict(X_test, w_seq)
acc_batch = np.mean(pred_batch == y_test) * 100
acc_seq = np.mean(pred_seq == y_test) * 100

print(f"\n=== Results ===")
print(f"Batch:      {len(err_batch)} epochs, {err_batch[-1]} final errors, {acc_batch:.1f}% test accuracy")
print(f"Sequential: {len(err_seq)} epochs, {err_seq[-1]} final errors, {acc_seq:.1f}% test accuracy")

# Plot convergence and confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(err_batch, label="Batch", color="steelblue", linewidth=1.5)
axes[0].plot(err_seq, label="Sequential", color="coral", linewidth=1.5)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Misclassified samples")
axes[0].set_title("Training Error Convergence")
axes[0].legend()
axes[0].grid(True, alpha=0.3)


def plot_confusion(ax, y_true, y_pred, title):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    cm = np.array([[tn, fp], [fn, tp]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-diabetic", "Diabetic"])
    ax.set_yticklabels(["Non-diabetic", "Diabetic"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)


plot_confusion(axes[1], y_test, pred_batch, f"Batch (Acc: {acc_batch:.1f}%)")
plot_confusion(axes[2], y_test, pred_seq, f"Sequential (Acc: {acc_seq:.1f}%)")
plt.tight_layout()
plt.show()
