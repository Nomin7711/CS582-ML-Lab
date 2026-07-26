# Part 1 - Why the Unit Hypersphere's Volume Shrinks Past ~5 Dimensions

## Answer

Adding a dimension does two opposite things at once: volume goes up and volume goes down. What that means is that it gives us a new direction to spread into, but at the same time a new dimension also squeezes every existing direction.

Why exactly around 5? The book's formula explains it in one line. Every time we add 2 dimensions, we multiply by:

$$v_n = \frac{2\pi}{n}v_{n-2}$$

That multiplier is greater than 1 while $n < 2\pi \approx 6.282$, and smaller than 1 afterward. So the volume increases until \(n\) passes \(2\pi\), then decreases forever. The peak occurs at **n = 5**.


<img width="824" height="697" alt="Screenshot 2026-07-26 at 14 58 47" src="https://github.com/user-attachments/assets/f32f32c6-1359-4a5d-be81-52bcb331010d" />

<img width="393" height="306" alt="Screenshot 2026-07-26 at 15 28 55" src="https://github.com/user-attachments/assets/1ebbf894-24a4-47de-8a76-3b1b36ed7a19" />

This graph shows that when the number of dimensions is above about 20, the volume is effectively zero. It was computed using the recursive formula:

$$v_n = \frac{2\pi}{n}v_{n-2}$$

As soon as $n > 2\pi$, the volume starts to shrink.

This phenomenon is called **the curse of dimensionality**, and it also applies to machine learning algorithms. As the number of input dimensions increases, much more training data is required for the model to generalize well.

---

# Part 2

## Question

**Short summary of the paper**

## Answer

The author compares two well-known dimensionality reduction methods:

- **PCA (Principal Component Analysis)** rotates the data to find the directions with the greatest variance and keeps only those. It does **not** use class labels, so it is an **unsupervised** method.
- **LDA (Linear Discriminant Analysis)** finds the directions that best separate the classes, making it a **supervised** method. However, it can only produce at most **(number of classes − 1)** dimensions.

Each method is evaluated using four classifiers:

- Decision Tree
- Naive Bayes
- Random Forest
- SVM

The evaluation pipeline is:

1. Normalize the data.
2. Convert categorical values to numeric.
3. Apply PCA or LDA.
4. Train the classifier.
5. Compare:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Sensitivity
   - Specificity

The experiments use three datasets of different sizes:

### Cardiotocography (CTG)

- 2,126 rows
- 36 attributes
- PCA reduced the data to **26 dimensions** while retaining **95% variance**
- LDA reduced it to **1 dimension**

### Diabetic Retinopathy (DR)

- 1,151 rows
- 20 attributes

### Intrusion Detection System (IDS)

- 125,973 rows
- 43 attributes
- Expanded to about **3,024 features** after one-hot encoding

### Results

The effectiveness of dimensionality reduction depends heavily on dataset size.

- On the **small DR dataset**, both PCA and LDA reduced performance because there was little unnecessary information to remove.
- On the **medium CTG dataset**, PCA maintained accuracy at around **98%**, while LDA performed slightly worse and reduced Naive Bayes accuracy to **85.6%**.
- On the **large IDS dataset**, classifiers using PCA actually outperformed those without dimensionality reduction.

Decision Trees and Random Forests were affected very little, which makes sense because tree-based models naturally perform their own feature selection.

Overall, **PCA outperformed LDA in almost every case**.

### Conclusion

Dimensionality reduction is **not automatically beneficial**.

Its usefulness increases as datasets become larger and more high-dimensional.

The authors recommend:

- **PCA + Random Forest** or **PCA + SVM** for high-dimensional datasets.
- **No dimensionality reduction** for small datasets.

---

# Part 3

From the paper, here are several important practices that can be applied.

1. **Always train a baseline model without dimensionality reduction.**

   This provides a reference point. Without a baseline, it is impossible to determine whether PCA improved performance or simply discarded useful information.

2. **Scale features before applying PCA.**

   PCA finds directions with the largest variance. If one feature has much larger numerical values than others, it can dominate the principal components simply because of its units. The paper uses **min-max normalization** to prevent this.

3. **Fit PCA only on the training data.**

   After fitting PCA on the training set, apply the same transformation to the test set. Fitting PCA on the entire dataset causes **data leakage** because information from the test data influences the training process.

4. **Choose the number of components using explained variance.**

   Rather than guessing the number of components, retain enough principal components to explain a desired percentage of the variance. The paper uses **95% explained variance**, which is a reasonable default.

5. **Understand the trade-off with interpretability.**

   Principal components are combinations of the original features, making it difficult to identify which original variables influenced a prediction. If interpretability is important, **feature selection** may be preferable to **feature extraction**.

6. **Different algorithms benefit differently.**

   Tree-based models changed very little in the experiments because they naturally perform feature selection. Distance-based and margin-based methods such as **SVM** benefit more because they are more affected by the distance concentration problem associated with the **curse of dimensionality**.

> **Note:** Running PCA before an MLP is essentially using a fixed linear first layer. An **autoencoder** is the nonlinear equivalent because it learns the feature compression automatically instead of relying on a fixed linear projection.
