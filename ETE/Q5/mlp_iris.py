import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

def main():
    # Ignore convergence warnings for cleaner output
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    print("--- [Code Block 1] Imports ---")
    print("Inference: Libraries imported successfully. Note the addition of StandardScaler for preprocessing.\n")

    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    print("--- [Code Block 2] Loading Data ---")
    print("Inference: The scikit-learn Iris dataset is a 'toy' dataset. It is entirely numeric and contains no missing values (NaNs). Therefore, explicit imputation or categorical encoding is not needed.\n")
    
    # Data Processing / Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("--- [Code Block 3] Data Preprocessing / Scaling ---")
    print(f"Mean of scaled data: {np.mean(X_scaled):.4f}, Std Dev: {np.std(X_scaled):.4f}")
    print("Inference: Neural Networks use gradient descent and are highly sensitive to feature scales. We used StandardScaler to normalize the features to have mean 0 and variance 1, which ensures stable and faster convergence.\n")

    # Define two architectures
    arch1 = (10,)
    arch2 = (10, 5)
    print("--- [Code Block 4] Defining Architectures ---")
    print(f"Architecture 1: {arch1}")
    print(f"Architecture 2: {arch2}")
    print("Inference: Two models defined. Arch 1 is a shallow network. Arch 2 is a deeper network.\n")
    
    # Evaluate Architecture 1
    print("--- [Code Block 5] Evaluating Architecture 1 (On Scaled Data) ---")
    mlp1 = MLPClassifier(hidden_layer_sizes=arch1, max_iter=1000, random_state=42)
    scores1 = cross_val_score(mlp1, X_scaled, y, cv=5, scoring='accuracy')
    cm1 = confusion_matrix(y, cross_val_predict(mlp1, X_scaled, y, cv=5))
    print(f"Mean Accuracy: {scores1.mean():.4f}")
    print(f"Confusion Matrix:\n{cm1}")
    print("Inference: The shallow model performs excellently on the scaled dataset, confirming that properly scaled data benefits MLP optimization.\n")

    # Evaluate Architecture 2
    print("--- [Code Block 6] Evaluating Architecture 2 (On Scaled Data) ---")
    mlp2 = MLPClassifier(hidden_layer_sizes=arch2, max_iter=1000, random_state=42)
    scores2 = cross_val_score(mlp2, X_scaled, y, cv=5, scoring='accuracy')
    cm2 = confusion_matrix(y, cross_val_predict(mlp2, X_scaled, y, cv=5))
    print(f"Mean Accuracy: {scores2.mean():.4f}")
    print(f"Confusion Matrix:\n{cm2}")
    print("Inference: The deeper model shows similar high accuracy. Extra layers aren't strictly necessary for a simple dataset like Iris.\n")

    # Model Inference (Prediction on new data)
    print("--- [Code Block 7] Actual Data Inference ---")
    final_model = MLPClassifier(hidden_layer_sizes=arch1, max_iter=1000, random_state=42)
    final_model.fit(X_scaled, y)
    
    new_raw_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    # MUST apply the same scaling to the inference data
    new_scaled_sample = scaler.transform(new_raw_sample)
    
    pred_idx = final_model.predict(new_scaled_sample)[0]
    print(f"Input (raw): {new_raw_sample[0]} => Predicted Class: {target_names[pred_idx]}")
    print("Inference: It is crucial to apply the *same* scaler transform to new inference data. The model successfully classified the normalized sample as 'setosa'.\n")

if __name__ == "__main__":
    main()
