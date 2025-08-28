"""
Utility functions for data loading, preprocessing, and model evaluation.

Author: Sameer Salman
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(path: Path, features: list):
    """Load dataset and handle missing values."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}.")
    df = pd.read_csv(path)
    df.fillna(df.median(numeric_only=True), inplace=True)
    X = df[features].copy()
    y = df['Defaulted'].copy()
    return X, y

def scale_features(X):
    """Scale features between 0 and 1 using MinMaxScaler."""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def evaluate_model(model, X_test, y_test, output_dir: Path):
    """Evaluate model and save confusion matrix and ROC curve plots."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    auc_score = roc_auc_score(y_test, y_pred)
    print(f"ROC AUC Score: {auc_score:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    roc_path = output_dir / "roc_curve.png"
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curve to {roc_path}.")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_path = output_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix image to {cm_path}.")

    return acc, auc_score, cm