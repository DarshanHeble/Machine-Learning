# Lab 5: Linear Regression through Gradient Descent

## Overview
This folder contains the implementation for Lab 5. The experiment focuses on building a Linear Regression model optimized via Gradient Descent to predict student performance (final grade `G3`) based on demographic, social, and school-related features.

## Setup Instructions

1. **Install required libraries** (if you haven't already):
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. **Run the experiment script**:
   ```bash
   python linear_regression_gd.py
   ```

## What the Code Does
1. **Data Loading**: Automatically downloads the UCI Student Performance dataset.
2. **Preprocessing**: 
   - Drops `G1` and `G2` (prior grades) to make the prediction task more realistic.
   - Applies `StandardScaler` to numerical features (crucial for Gradient Descent convergence).
   - Applies `OneHotEncoder` to categorical features.
3. **Training**: Implements Gradient Descent from scratch.
4. **Hyperparameter Tuning**: Tests learning rates (0.1, 0.01, 0.001) and saves a convergence plot to `cost_convergence.png`.
5. **Evaluation**: Outputs MAE, MSE, RMSE, and R² Score.
6. **Interpretation**: Prints out Data Scientist insights and ranks feature importance based on the learned weights.
