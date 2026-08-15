# CIA 3: Machine Learning for Social Good Challenge
## Mission Health: Early Maternal Health Risk Prediction in Under-Resourced Clinics

---

### Executive Summary
Maternal mortality remains a critical healthcare challenge, particularly in under-resourced rural clinics. This project builds an end-to-end, reproducible, and explainable Machine Learning solution using clinical biometrics (Age, Blood Pressure, Blood Sugar, Body Temperature, Heart Rate) to predict maternal health risk levels (**Low Risk**, **Mid Risk**, **High Risk**).

---

### 1. Dataset Citation & Metadata

* **Dataset Title:** Maternal Health Risk Data Set
* **Source:** UCI Machine Learning Repository / Kaggle
* **Citation:** Ahmed, M., Kashem, M.A., Rahman, M., & Khatun, S. (2020). *Maternal Health Risk Data Set*. UCI Machine Learning Repository. [https://doi.org/10.24432/C5154X](https://archive.ics.uci.edu/dataset/863/maternal+health+risk)
* **Collection Context:** Data collected from rural hospitals, community healthcare centers, and maternal clinics in Bangladesh.
* **Dataset Structure:** 1,014 checkup records, 7 feature columns (`Age`, `SystolicBP`, `DiastolicBP`, `BS`, `BodyTemp`, `HeartRate`, `RiskLevel`).

---

### 2. Reproducibility Instructions

Follow these step-by-step instructions to run the pipeline and reproduce all results and figures:

#### Prerequisites & Environment Setup
1. **Clone/Navigate to Project Repository:**
   ```bash
   cd /home/darshan/Projects/Machine-Learning/cia3
   ```
2. **Activate Virtual Environment:**
   ```bash
   source /home/darshan/Projects/Machine-Learning/venv/bin/activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install scikit-learn xgboost lightgbm catboost shap seaborn matplotlib pandas numpy nbformat
   ```

#### Executing the Pipeline & Generating Outputs
1. **Run Python Pipeline Script:**
   ```bash
   python main_pipeline.py
   ```
   *This command executes data wrangling, feature engineering, model benchmarks, and SHAP explainability analysis. Generated visual plots will be stored in `figures/`.*

2. **Run Jupyter Notebook:**
   Launch VS Code or Jupyter Lab and open:
   ```
   CIA3_Maternal_Health_Risk_Ensemble.ipynb
   ```
   Run all cells sequentially to reproduce inline charts and outputs.

---

### 3. Q1: Real-World Impact Framing (5 Marks)

* **Mission Domain:** Mission Health
* **Social Problem:** In rural and under-resourced clinics, specialized obstetricians are scarce. Delay in identifying high-risk pregnancies leads to preventable maternal and fetal complications.
* **Intended Beneficiaries:** Expectant mothers, rural healthcare workers, midwives, and clinical triage nurses.
* **Prediction Target:** 3-Class Categorical Risk Level:
  - `0`: Low Risk (Routine care required)
  - `1`: Mid Risk (Increased monitoring needed)
  - `2`: High Risk (Immediate specialist intervention required)
* **Why Machine Learning is Suitable:** Clinical vital signs exhibit complex non-linear interactions. Machine learning models capture these subtle patterns faster and more reliably than manual paper checkups.
* **Responsible-Use Limitations:** Auxiliary decision-support tool ONLY. All model predictions must be reviewed by human healthcare professionals prior to clinical action.

---

### 4. Q2: Data Wrangling & Feature Engineering (6 Marks)

#### Data Audit & Leakage-Safe Strategy
- **Missing Values:** Audited and verified 0 missing entries across all 1,014 records.
- **Duplicates Justification:** Retained duplicate patient checkup readings because different mothers naturally present identical vitals.
- **Reproducible Split:** 70% Train, 15% Validation, 15% Untouched Test split using Stratified Sampling.
- **Leakage-Safe Scaling:** `StandardScaler` fitted strictly on Training data (`X_train`) only.

#### Domain-Informed Feature Engineering
1. **Pulse Pressure (`SystolicBP` - `DiastolicBP`):** Indicator of vascular resistance and arterial stiffness during pregnancy.
2. **Mean Arterial Pressure (`MAP`):** Calculated as `DiastolicBP + (PulsePressure / 3.0)`. Measures average arterial pressure during a cardiac cycle.
3. **High Blood Sugar Indicator (`IsHighBS`):** Binary flag for blood sugar levels exceeding `7.0 mmol/L` (gestational diabetes threshold).

---

### 5. Q3: Ensemble Architecture, Tuning, and Comparison (8 Marks)

We benchmarked 7 model architectures on the untouched test set (153 samples):

| Model Architecture | Accuracy | Precision | Recall | F1-Score | Multi-class ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Boosting: XGBoost** | **0.7712** | **0.7707** | **0.7712** | **0.7686** | **0.9073** |
| **Boosting: LightGBM** | **0.7712** | **0.7696** | **0.7712** | **0.7661** | **0.9127** |
| **Stacking Ensemble (RF + XGB + Cat)** | **0.7647** | **0.7640** | **0.7647** | **0.7625** | **0.9055** |
| **Bagging: Random Forest** | 0.7320 | 0.7326 | 0.7320 | 0.7226 | 0.8969 |
| **Boosting: CatBoost** | 0.7190 | 0.7139 | 0.7190 | 0.7110 | 0.8820 |
| **Baseline: Logistic Regression** | 0.6601 | 0.6505 | 0.6601 | 0.6499 | 0.8020 |
| **Baseline: Decision Tree** | 0.6405 | 0.6091 | 0.6405 | 0.5819 | 0.8447 |

> **Key Finding:** Ensemble methods (Gradient Boosting & Stacking) outperform the baseline Decision Tree by over **18.6% in F1-score**, demonstrating superior generalization on unseen clinical data.

---

### 6. Q4: Model Explainability & Ethics Statement (4 Marks)

#### SHAP Explainability Output
* **Global Importance:** TreeSHAP analysis reveals that **Blood Sugar (`BS`)** is the primary signal driving high maternal risk predictions, followed by **Systolic BP**, **Age**, and engineered **Pulse Pressure**.
* **Local Interpretability:** Individual patient checkups are decomposed by SHAP feature forces, allowing clinical staff to understand exactly why a specific patient was assigned a High-Risk score.

#### Ethics Statement, Governance & Deployment Limits
* **Asymmetric Risk Costs:** In clinical triage, a **False Negative** (labeling a high-risk pregnancy as low risk) can cause severe harm due to delayed treatment. The model decision boundary is calibrated to optimize high recall on High Risk cases.
* **False Positive Considerations:** A False Positive triggers supplementary diagnostic testing. While it consumes clinic resources, it carries far lower human cost than missing a high-risk patient.
* **Privacy & Bias Audit:** The dataset contains non-PII clinical vitals. Continuous performance auditing across age demographics is mandatory to prevent age-related diagnostic bias.
* **Human-in-the-Loop Governance:** The model serves strictly as an auxiliary triage queue helper and must never make autonomous medical decisions.


---

### Summary of Generated Results & Figures

All figures are automatically generated and saved in the [`figures/`](file:///home/darshan/Projects/Machine-Learning/cia3/figures/) directory:
* [`1_risk_distribution.png`](file:///home/darshan/Projects/Machine-Learning/cia3/figures/1_risk_distribution.png) - Maternal Risk Class Counts.
* [`2_feature_boxplots.png`](file:///home/darshan/Projects/Machine-Learning/cia3/figures/2_feature_boxplots.png) - Biometric Signal Ranges across Risk Levels.
* [`3_model_comparison.png`](file:///home/darshan/Projects/Machine-Learning/cia3/figures/3_model_comparison.png) - F1-Score Performance Benchmark Chart.
* [`4_best_confusion_matrix.png`](file:///home/darshan/Projects/Machine-Learning/cia3/figures/4_best_confusion_matrix.png) - Confusion Matrix of Top Model.
* [`5_shap_summary.png`](file:///home/darshan/Projects/Machine-Learning/cia3/figures/5_shap_summary.png) - Global SHAP Feature Contribution Summary.
