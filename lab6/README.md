# Lab 6 — Comparison of Logistic Regression and K Nearest Neighbors (KNN) Classifiers

---

## Aim

To implement Logistic Regression and K Nearest Neighbors (KNN) classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset and compare their performance using standard classification evaluation metrics.

---

## Objectives

- To preprocess the dataset for classification.
- To implement Logistic Regression and KNN classifiers using Scikit-Learn.
- To evaluate both models using standard performance metrics.
- To compare the performance of Logistic Regression and KNN and identify the better classifier.

---

## Dataset

| Property | Details |
|---|---|
| **Name** | Breast Cancer Wisconsin (Diagnostic) |
| **Source** | UCI Machine Learning Repository |
| **Link** | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| **File** | `wdbc.data` (extracted from `breast+cancer+wisconsin+diagnostic.zip`) |
| **Samples** | 569 |
| **Features** | 30 numeric features (mean, SE, worst × 10 measurements) |
| **Target** | `diagnosis` — M (Malignant) or B (Benign) |

---

## Tasks Performed

1. Download and load the Breast Cancer Wisconsin (Diagnostic) dataset.
2. Perform exploratory data analysis and data preprocessing.
3. Check for missing values and prepare the dataset for classification.
4. Split the dataset into training and testing sets.
5. Train a Logistic Regression classifier.
6. Train a K Nearest Neighbors (KNN) classifier.
7. Evaluate both classifiers using Accuracy, Precision, Recall, F1 Score, Confusion Matrix.
8. Compare the performance of the two classifiers using a comparison table.
9. Interpret the results and identify the better-performing classifier.

---

## Files

```
lab6/
├── breast+cancer+wisconsin+diagnostic.zip   ← original downloaded zip
├── wdbc.data                                ← extracted dataset
├── wdbc.names                               ← column descriptions
├── lab6_2547119.ipynb                       ← main notebook
├── README.md                                ← this file
│
├── chart1_class_distribution.png
├── chart2_feature_distributions.png
├── chart3_correlation_heatmap.png
├── chart4_knn_elbow.png
├── chart5_confusion_matrices.png
├── chart6_metric_comparison.png
├── chart7_radar_chart.png
├── chart8_feature_importance.png
│
├── sl1_knn_metrics_vs_k.png
├── sl2_scaling_effect.png
├── sl3_lr_regularization.png
└── sl4_knn_distance_metrics.png
```

---

## Step-by-Step Flow (What We Saw → What We Decided)

This is the full observation → reasoning → decision chain followed in the notebook.

---

### Step 1 — Load the Data

**What we did:** Loaded `wdbc.data` using pandas. The file has no header, so we manually assigned column names using the feature list from `wdbc.names`.

Column structure:
- Column 0 → `id` (patient identifier)
- Column 1 → `diagnosis` (M or B)
- Columns 2–31 → 30 numeric features (mean, SE, worst of 10 tumour measurements)

---

### Step 2 — Basic EDA

**What we saw:**
- 569 samples, 32 columns
- **No missing values** — clean dataset
- Class split: **357 Benign (62.7%)** vs **212 Malignant (37.3%)**
- All 30 features are continuous floats — no categorical encoding needed for features

**Decision made:**
- No imputation needed since there are zero missing values.
- The slight class imbalance (63/37) is not extreme — no need for SMOTE or oversampling.
- Will use `stratify=y` in the train-test split to preserve this ratio in both splits.

---

### Step 3 — Statistical Summary

**What we saw:**
- `mean_area` ranges from ~143 to ~2501 — very large values.
- `mean_smoothness` ranges from ~0.05 to ~0.16 — tiny values.
- Some features like `mean_concavity` have a min of 0 — some tumours have no concavity at all.
- Huge scale differences exist across features.

**Decision made:**
- **Must apply feature scaling (StandardScaler)** before training.
- Without scaling, KNN's distance calculation would be dominated by features with large values (like area), completely ignoring features with small values (like smoothness).
- Logistic Regression also converges faster on scaled data.

---

### Step 4 — Chart 1: Class Distribution

**What we saw:**
- Bar chart and pie chart confirmed the 63% Benign / 37% Malignant split visually.
- The imbalance is real but not severe.

**Decision made:**
- Use `stratify=y` in `train_test_split` so both training and testing sets have the same class ratio.
- If we don't stratify, a random split might put most Malignant cases only in training or only in testing — giving misleading evaluation results.

---

### Step 5 — Chart 2: Feature Distributions by Diagnosis

**What we saw:**
- Features like **radius, perimeter, area, concavity, concave_points** → clear separation between M (pink) and B (teal).
- Malignant tumours are generally larger, more irregular, and more concave.
- Features like **smoothness, symmetry, fractal_dimension** → heavily overlapping distributions. Hard to separate by these alone.
- Malignant distributions are right-skewed — a few very large tumours exist.

**Decision made:**
- Keep all 30 features — even the overlapping ones carry some signal.
- The visible separability in key features tells us both LR and KNN should work reasonably well here.
- No feature engineering is needed for this experiment.

---

### Step 6 — Chart 3: Correlation Heatmap

**What we saw:**
- `mean_radius`, `mean_perimeter`, `mean_area` are correlated at ~0.98–0.99 with each other. They are essentially saying the same thing three different ways.
- `mean_concavity` and `mean_concave_points` are correlated at ~0.92.
- `mean_texture` and `mean_fractal_dimension` are relatively independent from others.

**Decision made:**
- Did not drop highly correlated features — doing so without a formal feature selection study could remove useful information.
- High multicollinearity can affect the interpretability of LR coefficients (weight gets split between correlated features) but doesn't hurt accuracy significantly.
- KNN is not affected by multicollinearity at all since it works on distances.

---

### Step 7 — Preprocessing

**What we did and why:**

| Action | Reason |
|---|---|
| Dropped `id` column | Patient ID has no relationship to cancer diagnosis. Keeping it would teach the model nonsense patterns from random numbers. |
| Label Encoded `diagnosis` | ML models need numbers. `M → 1`, `B → 0`. |
| 80/20 Train-Test Split with `stratify=y` | 80% training gives enough data. stratify preserves the 63/37 class ratio in both sets. |
| Applied `StandardScaler` | Brings all features to mean=0, std=1. Critical for KNN (distance-based). Also helps LR converge faster. |
| `fit_transform` on train, `transform` on test | Scaler is fit only on training data to avoid data leakage. Test data is transformed using train statistics only. |

---

### Step 8 — Logistic Regression Training

**Why LR works well here:**
- From Chart 2, we saw that several features show clean separation between Malignant and Benign.
- This suggests the data may be **nearly linearly separable** in the 30-dimensional feature space.
- Logistic Regression works by finding a hyperplane that separates the two classes — exactly suited for this kind of data.

---

### Step 9 — Chart 4: KNN Elbow Plot (Finding Best K)

**What we saw:**
- K=1 gives high accuracy but is suspicious — it memorises the nearest single point and can overfit on noise.
- Accuracy peaks at around **K=5** and then slowly decreases.
- After K=10, accuracy dips — too many neighbours means irrelevant points get included in the vote.

**Decision made:**
- Use **K=5** for the final KNN model.
- K=5 gives the best test accuracy without the risks of K=1 overfitting.

---

### Step 10 — Chart 5: Confusion Matrices

**What the boxes mean:**

```
                  Predicted
                  Benign    Malignant
Actual  Benign  [ True Neg  | False Pos ]
        Malig.  [ False Neg | True Pos  ]
```

- **False Negative** (bottom-left) = Predicted Benign but actually Malignant → **most dangerous error in cancer screening**
- **False Positive** (top-right) = Predicted Malignant but actually Benign → less dangerous, causes unnecessary follow-up

**What we saw:**
- Logistic Regression has **fewer false negatives** than KNN.
- Both models have similarly low false positive rates.

**Decision made:**
- For medical/cancer data, Logistic Regression is the safer model — it misses fewer real cancer cases.

---

### Step 11 — Charts 6 & 7: Metric Comparison (Bar + Radar)

**What we saw:**

| Metric | Logistic Regression | KNN (K=5) |
|---|---|---|
| Accuracy | **0.9649** | 0.9561 |
| Precision | **0.9750** | 0.9744 |
| Recall | **0.9286** | 0.9048 |
| F1 Score | **0.9512** | 0.9383 |

- LR wins on all four metrics.
- The radar chart (Chart 7) shows the purple area (LR) is consistently wider than amber (KNN).
- The gap is not massive — both are competitive models here.

**Why LR beats KNN here:**
- The nearly linear separability favours LR.
- With 30 features, KNN suffers from the **curse of dimensionality** — in high dimensions, all points tend to become roughly equidistant from each other, making the "nearest neighbor" idea less meaningful.
- LR is also faster at prediction time (just a dot product) vs KNN (computes distances to all training points for every new sample).

---

### Step 12 — Chart 8: Feature Importance (LR Coefficients)

**What we saw:**
- Top features by LR coefficient magnitude: `worst_concave_points`, `worst_radius`, `worst_texture`, `mean_concave_points`.
- These are almost exactly the features we visually identified as having clean class separation in Chart 2.
- `fractal_dimension` features have very small coefficients — matching our EDA observation that they had overlapping distributions.

**Key insight:**
The model is confirming what our eyes told us during EDA. This is how real data science works — the model validates your exploration, and your exploration helps you trust the model.

---

## Final Conclusion

Both classifiers perform well (>95% accuracy). **Logistic Regression** is the better classifier for this dataset because:

1. Higher scores on Accuracy, Recall, and F1
2. Fewer false negatives — safer for cancer detection
3. Data appears nearly linearly separable — LR's strength
4. More interpretable — coefficients directly show feature importance
5. Faster at inference time — just a matrix multiplication

KNN is still a strong alternative. Its main limitations here: sensitivity to feature scale (solved by scaling), and the curse of dimensionality with 30 features.

---

## Self-Learning Section

Four extra experiments that test "what if" questions that come up naturally after doing the main lab.

---

### SL-1: Is the best K for Accuracy also best for Recall?

**Question:** We chose K=5 based on accuracy. But for cancer detection, Recall matters more. Does K=5 also maximise Recall?

**What we found:** Not always. The K that maximises accuracy and the K that maximises recall can be different. In real medical applications, you'd tune K specifically for the metric that matters most (Recall in this case).

**Takeaway:** Always know which metric matters for your problem before tuning hyperparameters.

---

### SL-2: What happens if we skip scaling?

**Question:** We said scaling is essential for KNN. Let's actually prove it by running KNN on raw unscaled data and comparing.

**What we found:** KNN on unscaled data performs noticeably worse — especially on Recall. This directly proves why scaling was necessary.

**Takeaway:** For distance-based models like KNN, feature scaling is not optional — it's required.

---

### SL-3: How sensitive is Logistic Regression to the C parameter?

**What C controls:** C is the inverse of regularization strength.
- Low C (e.g. 0.001) → Strong regularization → Model is too simple → Underfitting
- High C (e.g. 1000) → Weak regularization → Model memorises training data → Overfitting
- Default C=1 → Balanced starting point

**What we found:** Very low C clearly hurts performance. Default C=1 is a good baseline. Scores plateau after C=10.

**Takeaway:** Default C=1 is safe. Tune it if you need that extra bit of performance.

---

### SL-4: Does the choice of distance metric in KNN matter?

Tested three distance metrics using 5-fold cross-validation:

| Metric | Formula | Use Case |
|---|---|---|
| **Euclidean (p=2)** | Straight-line distance | Default, works well in most cases |
| **Manhattan (p=1)** | Sum of absolute differences | More robust to outliers |
| **Chebyshev** | Maximum coordinate difference | Works differently in high dimensions |

**What we found:** Euclidean and Manhattan usually give similar results. Chebyshev can differ. Worth experimenting with based on your data.

**Takeaway:** KNN is not just one fixed algorithm — the distance metric is a meaningful hyperparameter to explore.

---

## Expected Learning Outcomes

After completing this lab, you should be able to:

- Implement Logistic Regression and KNN using Scikit-Learn.
- Apply preprocessing techniques (encoding, scaling, splitting) to a real-world dataset.
- Evaluate classification models using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
- Understand why each preprocessing decision was made based on data observations.
- Compare multiple classifiers and recommend the best one with reasoning.
- Understand the curse of dimensionality and its effect on distance-based models.
- Tune hyperparameters (K for KNN, C for LR) and understand their impact.

---

*Lab 6 | Machine Learning | Breast Cancer Wisconsin Dataset*
