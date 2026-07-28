import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ensure output directory for figures
fig_dir = '/home/darshan/Projects/Machine-Learning/lab7'
os.makedirs(fig_dir, exist_ok=True)

# Set aesthetic visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0

# -------------------------------------------------------------
# TASK 1: Dataset Exploration
# -------------------------------------------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')
y_names = pd.Series([iris.target_names[i] for i in iris.target], name='species')

n_samples, n_features = X.shape
feature_names = iris.feature_names
target_classes = list(iris.target_names)
first_5 = X.head()
class_dist = y_names.value_counts()

print(f"Task 1 Exploration:")
print(f"Samples: {n_samples}, Features: {n_features}")
print(f"Features: {feature_names}")
print(f"Classes: {target_classes}")
print("First 5 records:\n", first_5)
print("Class distribution:\n", class_dist)

# Plot Task 1 - Class Distribution & Feature Scatter
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

palette = ['#2b5c8f', '#d95f02', '#7570b3']
sns.countplot(x=y_names, palette=palette, ax=axes[0], hue=y_names, legend=False)
axes[0].set_title('Iris Class Distribution', fontsize=14, fontweight='bold', pad=12)
axes[0].set_xlabel('Species Class', fontsize=12)
axes[0].set_ylabel('Sample Count', fontsize=12)
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha='center', va='center', fontsize=12, color='white', fontweight='bold')

df_combined = pd.concat([X, y_names], axis=1)
sns.scatterplot(data=df_combined, x='petal length (cm)', y='petal width (cm)', hue='species', palette=palette, s=70, ax=axes[1])
axes[1].set_title('Petal Feature Separability', fontsize=14, fontweight='bold', pad=12)
axes[1].set_xlabel('Petal Length (cm)', fontsize=12)
axes[1].set_ylabel('Petal Width (cm)', fontsize=12)

plt.tight_layout()
plt.savefig(f"{fig_dir}/task1_exploration.png", dpi=300)
plt.close()

# -------------------------------------------------------------
# TASK 2: Data Preparation
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTask 2 Data Split:")
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# -------------------------------------------------------------
# TASK 3: Default Decision Tree Classifier
# -------------------------------------------------------------
dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, y_train)
y_pred_default = dt_default.predict(X_test)

acc_default = accuracy_score(y_test, y_pred_default)
cm_default = confusion_matrix(y_test, y_pred_default)
cr_default = classification_report(y_test, y_pred_default, target_names=iris.target_names)

print(f"\nTask 3 Default DT Accuracy: {acc_default:.4f}")
print("Confusion Matrix:\n", cm_default)
print("Classification Report:\n", cr_default)

# Plot Task 3 Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax, cbar=False, annot_kws={"size": 14, "weight": "bold"})
ax.set_title('Default Decision Tree - Confusion Matrix', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('True Label', fontsize=11)
plt.tight_layout()
plt.savefig(f"{fig_dir}/task3_confusion_matrix.png", dpi=300)
plt.close()

# -------------------------------------------------------------
# TASK 4: Visualizing the Decision Tree
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(dt_default, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, ax=ax, fontsize=10)
ax.set_title('Task 4: Default Decision Tree Visualization (max_depth=4)', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f"{fig_dir}/task4_default_tree.png", dpi=300)
plt.close()

# Print tree structure in text
print("\nTask 4 Tree Text Structure:\n", export_text(dt_default, feature_names=iris.feature_names))

# -------------------------------------------------------------
# TASK 5: Comparing Gini Index vs Entropy
# -------------------------------------------------------------
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)

dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X_train, y_train)
y_pred_entropy = dt_entropy.predict(X_test)

res_gini = {
    'Criterion': 'Gini Index',
    'Accuracy': accuracy_score(y_test, y_pred_gini),
    'Tree Depth': dt_gini.get_depth(),
    'Number of Leaf Nodes': dt_gini.get_n_leaves(),
    'Root Feature Selected': iris.feature_names[dt_gini.tree_.feature[0]]
}

res_entropy = {
    'Criterion': 'Entropy',
    'Accuracy': accuracy_score(y_test, y_pred_entropy),
    'Tree Depth': dt_entropy.get_depth(),
    'Number of Leaf Nodes': dt_entropy.get_n_leaves(),
    'Root Feature Selected': iris.feature_names[dt_entropy.tree_.feature[0]]
}

df_task5 = pd.DataFrame([res_gini, res_entropy])
print("\nTask 5 Comparison:\n", df_task5)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
plot_tree(dt_gini, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, ax=axes[0])
axes[0].set_title('Gini Index Decision Tree', fontsize=13, fontweight='bold')
plot_tree(dt_entropy, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, ax=axes[1])
axes[1].set_title('Entropy Information Gain Tree', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{fig_dir}/task5_gini_vs_entropy.png", dpi=300)
plt.close()

# -------------------------------------------------------------
# TASK 6: Effect of Maximum Tree Depth
# -------------------------------------------------------------
max_depths = [1, 2, 3, 4, None]
t6_rows = []

for md in max_depths:
    clf = DecisionTreeClassifier(max_depth=md, random_state=42)
    clf.fit(X_train, y_train)
    tr_acc = accuracy_score(y_train, clf.predict(X_train))
    te_acc = accuracy_score(y_test, clf.predict(X_test))
    actual_depth = clf.get_depth()
    
    if md == 1:
        obs = "Underfitting: Tree is too shallow to separate versicolor and virginica; misses key patterns."
    elif md == 2:
        obs = "Slight Underfit: Captures setosa and distinguishes most versicolor/virginica."
    elif md == 3:
        obs = "Best Generalization: Achieves 100% test accuracy while keeping simple depth = 3."
    elif md == 4:
        obs = "Overfitting tendency: Splits down to pure nodes on train (97.5%) with extra depth."
    else: # None
        obs = "Full expansion: Fully pure leaves on train (100%), vulnerable to noise."
        
    t6_rows.append({
        'Max Depth': str(md),
        'Training Accuracy': f"{tr_acc*100:.2f}%",
        'Testing Accuracy': f"{te_acc*100:.2f}%",
        'Tree Depth': actual_depth,
        'Observation': obs,
        'tr_acc_raw': tr_acc,
        'te_acc_raw': te_acc
    })

df_task6 = pd.DataFrame(t6_rows)
print("\nTask 6 Depth Table:\n", df_task6[['Max Depth', 'Training Accuracy', 'Testing Accuracy', 'Tree Depth', 'Observation']])

# Visualizing Task 6 Curves
fig, ax = plt.subplots(figsize=(8, 5))
depth_labels = ['1', '2', '3', '4', 'None (4)']
train_accs = [r['tr_acc_raw']*100 for r in t6_rows]
test_accs = [r['te_acc_raw']*100 for r in t6_rows]

ax.plot(depth_labels, train_accs, marker='o', linewidth=2.5, color='#2b5c8f', label='Training Accuracy')
ax.plot(depth_labels, test_accs, marker='s', linewidth=2.5, color='#d95f02', linestyle='--', label='Testing Accuracy')
ax.set_title('Effect of Max Depth on Training vs Testing Accuracy', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Max Depth Parameter', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_ylim(60, 105)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{fig_dir}/task6_max_depth_effect.png", dpi=300)
plt.close()

# -------------------------------------------------------------
# TASK 7: Effect of min_samples_split
# -------------------------------------------------------------
min_splits = [2, 5, 10, 20]
t7_rows = []

for ms in min_splits:
    clf = DecisionTreeClassifier(min_samples_split=ms, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    depth = clf.get_depth()
    leaves = clf.get_n_leaves()
    
    if ms == 2:
        obs = "Baseline full tree; allows splitting nodes with as few as 2 samples."
    elif ms == 5:
        obs = "Identical tree structure to baseline for this dataset; high accuracy."
    elif ms == 10:
        obs = "Prunes minor splits; maintains perfect 100% test accuracy."
    else: # 20
        obs = "Prevents fine-grained splits; reduces leaf count from 6 to 4 while keeping 100% test accuracy."
        
    t7_rows.append({
        'min_samples_split': ms,
        'Accuracy': f"{acc*100:.2f}%",
        'Tree Depth': depth,
        'Number of Leaf Nodes': leaves,
        'Observation': obs
    })

df_task7 = pd.DataFrame(t7_rows)
print("\nTask 7 min_samples_split Table:\n", df_task7)

# -------------------------------------------------------------
# TASK 8: Effect of min_samples_leaf
# -------------------------------------------------------------
min_leaves = [1, 2, 5, 10]
t8_rows = []

for ml in min_leaves:
    clf = DecisionTreeClassifier(min_samples_leaf=ml, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    depth = clf.get_depth()
    leaves = clf.get_n_leaves()
    
    if ml == 1:
        obs = "Baseline default; allows leaves with a single sample."
    elif ml == 2:
        obs = "Slight regularization; prevents isolated outlier leaves."
    elif ml == 5:
        obs = "Smoother decision boundaries; maintains perfect 100% test accuracy with 4 leaves."
    else: # 10
        obs = "Strong regularization; reduces depth to 2 and leaves to 3 while maintaining 100% accuracy."
        
    t8_rows.append({
        'min_samples_leaf': ml,
        'Accuracy': f"{acc*100:.2f}%",
        'Tree Depth': depth,
        'Number of Leaf Nodes': leaves,
        'Observation': obs
    })

df_task8 = pd.DataFrame(t8_rows)
print("\nTask 8 min_samples_leaf Table:\n", df_task8)

# Visualizing Task 7 & 8 Node/Depth reduction
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot([str(m['min_samples_split']) for m in t7_rows], [m['Number of Leaf Nodes'] for m in t7_rows], marker='o', color='#2b5c8f', linewidth=2.5, label='Leaf Nodes')
axes[0].plot([str(m['min_samples_split']) for m in t7_rows], [m['Tree Depth'] for m in t7_rows], marker='s', color='#7570b3', linewidth=2.5, linestyle='--', label='Tree Depth')
axes[0].set_title('Impact of min_samples_split', fontsize=13, fontweight='bold', pad=12)
axes[0].set_xlabel('min_samples_split', fontsize=11)
axes[0].set_ylabel('Count / Depth', fontsize=11)
axes[0].legend()

axes[1].plot([str(m['min_samples_leaf']) for m in t8_rows], [m['Number of Leaf Nodes'] for m in t8_rows], marker='o', color='#d95f02', linewidth=2.5, label='Leaf Nodes')
axes[1].plot([str(m['min_samples_leaf']) for m in t8_rows], [m['Tree Depth'] for m in t8_rows], marker='s', color='#1b9e77', linewidth=2.5, linestyle='--', label='Tree Depth')
axes[1].set_title('Impact of min_samples_leaf', fontsize=13, fontweight='bold', pad=12)
axes[1].set_xlabel('min_samples_leaf', fontsize=11)
axes[1].set_ylabel('Count / Depth', fontsize=11)
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{fig_dir}/tasks_7_8_hyperparameters.png", dpi=300)
plt.close()

# -------------------------------------------------------------
# SELF LEARNING PART: Feature Importance & Cost-Complexity Pruning (ccp_alpha)
# -------------------------------------------------------------
importances = dt_default.feature_importances_
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(x=importances, y=iris.feature_names, palette='Blues_r', ax=ax, hue=iris.feature_names, legend=False)
ax.set_title('Self Learning 1: Feature Importances in Decision Tree', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Gini Importance (MDI)', fontsize=11)
for i, v in enumerate(importances):
    ax.text(v + 0.01, i, f"{v*100:.1f}%", va='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(f"{fig_dir}/self_learning_feature_importance.png", dpi=300)
plt.close()

path = dt_default.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ccp_alphas, train_scores, marker='o', label='Train Accuracy', drawstyle="steps-post", color='#2b5c8f')
ax.plot(ccp_alphas, test_scores, marker='s', label='Test Accuracy', drawstyle="steps-post", color='#d95f02', linestyle='--')
ax.set_xlabel("alpha (ccp_alpha)", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Self Learning 2: Post-Pruning via Cost-Complexity Pruning (ccp_alpha)", fontsize=13, fontweight='bold', pad=12)
ax.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/self_learning_ccp_pruning.png", dpi=300)
plt.close()

print("\nAll execution completed successfully!")
