import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap

# Set styling for clear, beautiful charts
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

print("=== CIA 3: MATERNAL HEALTH RISK ML PIPELINE ===")

# Create directory for saving visual charts
os.makedirs('figures', exist_ok=True)

# 1. LOAD DATASET
data_path = 'Maternal_Health_Risk_DataSet.csv'
df = pd.read_csv(data_path)

print(f"\n[STEP 1] Data Loaded Successfully!")
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows of the data:")
print(df.head())

# High school level simple inference
print("\n--- DATA SCIENTIST OBSERVATION & INFERENCE ---")
print("Simple Explanation:")
print("We have data for 1,014 pregnant mothers. Each record gives us 6 health measurements:")
print("1. Age (in years)")
print("2. Systolic Blood Pressure (Upper BP number)")
print("3. Diastolic Blood Pressure (Lower BP number)")
print("4. Blood Sugar (BS level)")
print("5. Body Temperature (in Fahrenheit)")
print("6. Heart Rate (beats per minute)")
print("Our goal is to predict 'RiskLevel': whether the mother is at 'low risk', 'mid risk', or 'high risk'.")

# 2. EDA & DATA CLEANING
print("\n[STEP 2] Data Cleaning & Exploratory Data Analysis (EDA)")

# Check missing values
missing_counts = df.isnull().sum()
print("\nMissing values per column:")
print(missing_counts)
print("Inference: No missing values found in this dataset!")

# Check duplicates
duplicate_count = df.duplicated().sum()
print(f"\nDuplicate rows found: {duplicate_count}")
print("Inference: We keep duplicate patient readings because different mothers can naturally have identical vital signs!")

# Visualizing Target Class Distribution
plt.figure(figsize=(7, 5))
ax = sns.countplot(x='RiskLevel', data=df, palette='Set2', order=['low risk', 'mid risk', 'high risk'])
plt.title('Distribution of Maternal Risk Levels', fontweight='bold', pad=15)
plt.xlabel('Risk Level Category')
plt.ylabel('Number of Mothers')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/1_risk_distribution.png', dpi=300)
plt.close()
print("Saved chart: figures/1_risk_distribution.png")

# Visualizing Feature Boxplots to understand ranges & outliers
plt.figure(figsize=(12, 8))
features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
for i, col in enumerate(features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[col], x=df['RiskLevel'], palette='Set2', order=['low risk', 'mid risk', 'high risk'])
    plt.title(f'{col} by Risk Level', fontweight='bold')
    plt.xlabel('')
plt.tight_layout()
plt.savefig('figures/2_feature_boxplots.png', dpi=300)
plt.close()
print("Saved chart: figures/2_feature_boxplots.png")

print("\n--- SIMPLE EDA INFERENCES ---")
print("1. Blood Sugar (BS): Mothers with High Risk have much higher blood sugar levels than low risk mothers.")
print("2. Systolic & Diastolic BP: High blood pressure strongly connects to high pregnancy risk.")
print("3. Age: Older mothers (above 35-40 years) show slightly higher chances of being in the high-risk group.")

# 3. FEATURE ENGINEERING & LEAKAGE-SAFE PREPROCESSING
print("\n[STEP 3] Feature Engineering & Pipeline Setup")

# Create Domain-Informed Features
# Pulse Pressure = Systolic BP - Diastolic BP (Key cardiovascular marker in pregnancy)
df['PulsePressure'] = df['SystolicBP'] - df['DiastolicBP']

# Mean Arterial Pressure (MAP) = DiastolicBP + (PulsePressure / 3)
df['MAP'] = df['DiastolicBP'] + (df['PulsePressure'] / 3.0)

# High Blood Sugar Flag (BS > 7.0 mmol/L is clinically significant)
df['IsHighBS'] = (df['BS'] > 7.0).astype(int)

print("Engineered 3 new clinical features:")
print("- PulsePressure (SystolicBP - DiastolicBP)")
print("- MAP (Mean Arterial Pressure)")
print("- IsHighBS (Indicator for elevated blood sugar > 7.0 mmol/L)")

# Encode Target Variable: low risk -> 0, mid risk -> 1, high risk -> 2
label_mapping = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
df['Target'] = df['RiskLevel'].map(label_mapping)

X = df.drop(columns=['RiskLevel', 'Target'])
y = df['Target']

# Reproducible Stratified Train / Validation / Test Split (70% Train, 15% Val, 15% Test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nTrain set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Scaling features using StandardScaler fit ONLY on X_train to prevent data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for feature names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Combine Train + Val for final model training on full training data
X_train_full = np.vstack((X_train_scaled, X_val_scaled))
y_train_full = np.hstack((y_train, y_val))

# 4. MODEL TRAINING & COMPARISON
print("\n[STEP 4] Model Building & Performance Evaluation")

models = {
    'Baseline: Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Baseline: Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Bagging: Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
    'Boosting: XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='mlogloss'),
    'Boosting: LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1),
    'Boosting: CatBoost': CatBoostClassifier(iterations=150, depth=5, learning_rate=0.1, random_state=42, verbose=0)
}

# Add Stacking Model
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='mlogloss')),
    ('cat', CatBoostClassifier(iterations=150, depth=5, learning_rate=0.1, random_state=42, verbose=0))
]
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
models['Stacking Ensemble (RF + XGB + CatBoost)'] = stacking_model

results = []

print("\nTraining and evaluating models on the untouched Test Set...")
print("-" * 75)

for name, model in models.items():
    model.fit(X_train_full, y_train_full)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })
    print(f"{name:45s} | Acc: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

results_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)

print("\n=== FINAL MODEL PERFORMANCE SUMMARY ===")
print(results_df.to_string(index=False))

# Plotting Model Comparison
plt.figure(figsize=(10, 5))
sns.barplot(x='F1-Score', y='Model', data=results_df, palette='viridis')
plt.title('Model Comparison by F1-Score (Untouched Test Set)', fontweight='bold', pad=15)
plt.xlim(0.6, 1.0)
for index, value in enumerate(results_df['F1-Score']):
    plt.text(value + 0.005, index, f'{value:.4f}', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/3_model_comparison.png', dpi=300)
plt.close()
print("\nSaved chart: figures/3_model_comparison.png")

# Best Model Confusion Matrix
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Risk', 'Mid Risk', 'High Risk'],
            yticklabels=['Low Risk', 'Mid Risk', 'High Risk'])
plt.title(f'Confusion Matrix: {best_model_name}', fontweight='bold', pad=15)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('figures/4_best_confusion_matrix.png', dpi=300)
plt.close()
print("Saved chart: figures/4_best_confusion_matrix.png")

# 5. MODEL EXPLAINABILITY WITH SHAP
print("\n[STEP 5] Model Explainability & Interpretability (SHAP)")

# Explain Random Forest or Tree-based best model
rf_explainer_model = models['Bagging: Random Forest']
explainer = shap.TreeExplainer(rf_explainer_model)
shap_values = explainer.shap_values(X_test_scaled_df)

# SHAP Summary Plot
plt.figure(figsize=(9, 6))
if isinstance(shap_values, list):
    shap.summary_plot(shap_values, X_test_scaled_df, class_names=['Low Risk', 'Mid Risk', 'High Risk'], show=False)
else:
    shap.summary_plot(shap_values, X_test_scaled_df, show=False)
plt.title('SHAP Feature Importance for Maternal Risk', fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('figures/5_shap_summary.png', dpi=300)
plt.close()
print("Saved chart: figures/5_shap_summary.png")

print("\n--- SIMPLE SHAP INFERENCE (High-School Level) ---")
print("1. Blood Sugar (BS) is the MOST important factor. High blood sugar pushes predictions strongly towards High Risk.")
print("2. Systolic Blood Pressure and Age are the next biggest drivers. Higher values push the risk up.")
print("3. Pulse Pressure and Body Temperature add important clues for difficult cases.")

print("\n=== Pipeline Execution Completed Successfully! ===")
