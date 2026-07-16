import numpy as np
import matplotlib.pyplot as plt


def perceptron_batch(X, targets, learning_rate=0.25, max_epochs=100):
    n, f = X.shape
    Xb = np.concatenate([X, -np.ones((n, 1))], axis=1)
    w = np.random.uniform(-0.5, 0.5, f + 1)
    hist = []
    for _ in range(max_epochs):
        a = np.where(Xb @ w > 0, 1, 0)
        e = targets - a
        ne = np.sum(np.abs(e))
        hist.append(ne)
        if ne == 0:
            break
        w += learning_rate * (e @ Xb)
    return w, hist


def perceptron_sequential(X, targets, learning_rate=0.25, max_epochs=100):
    n, f = X.shape
    Xb = np.concatenate([X, -np.ones((n, 1))], axis=1)
    w = np.random.uniform(-0.5, 0.5, f + 1)
    hist = []
    for _ in range(max_epochs):
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


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 0, 0, 1])

# Test on AND gate
np.random.seed(42)
w_batch, err_batch = perceptron_batch(X, targets)
np.random.seed(42)
w_seq, err_seq = perceptron_sequential(X, targets)

Xb = np.concatenate([X, -np.ones((4, 1))], axis=1)
print("=== AND Gate Results ===")
print(f"Batch:      converged in {len(err_batch)} epochs, outputs: {np.where(Xb @ w_batch > 0, 1, 0)}")
print(f"Sequential: converged in {len(err_seq)} epochs, outputs: {np.where(Xb @ w_seq > 0, 1, 0)}")
print(f"Expected:   {targets}")

# Compare across all logic gates
gates = {
    "AND": np.array([0, 0, 0, 1]),
    "OR": np.array([0, 1, 1, 1]),
    "NAND": np.array([1, 1, 1, 0]),
    "NOR": np.array([1, 0, 0, 0]),
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, t) in zip(axes.flat, gates.items()):
    bl, sl = [], []
    for seed in range(50):
        np.random.seed(seed)
        _, eb = perceptron_batch(X, t)
        np.random.seed(seed)
        _, es = perceptron_sequential(X, t)
        bl.append(len(eb))
        sl.append(len(es))
    ax.bar([0, 1], [np.mean(bl), np.mean(sl)],
           yerr=[np.std(bl), np.std(sl)],
           color=["steelblue", "coral"], capsize=5, width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Batch", "Sequential"])
    ax.set_ylabel("Epochs to converge")
    ax.set_title(f"{name} Gate")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Batch vs Sequential — Average Epochs to Converge (50 runs)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
