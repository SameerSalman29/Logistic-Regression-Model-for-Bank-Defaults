"""
Train a Logistic Regression model to predict bank loan defaults.
Saves trained model, scaler, and evaluation plots.

Author: Sameer Salman
"""

import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from utils import load_data, scale_features, evaluate_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "loan_defaulting_credit_data.csv"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

FEATURES = ['Age', 'Annual_Income', 'Employment_Years', 'Credit_Score', 'Loan_Amount', 'Loan_Term_Months', 'Late_Payments']

def main():
    X, y = load_data(DATA, FEATURES)
    X_scaled, scaler = scale_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.6, random_state=42, stratify=y)
    
    model = LogisticRegression(class_weight='balanced', max_iter=50000, n_jobs=None, solver='lbfgs')
    model.fit(X_train, y_train)
    
    acc, auc_score, cm = evaluate_model(model, X_test, y_test, MODELS)
    
    joblib.dump(model, MODELS / "model.pkl")
    joblib.dump(scaler, MODELS / "scaler.pkl")
    print(f"Saved model to {MODELS/'model.pkl'} and scaler to {MODELS/'scaler.pkl'}.")
    balance = y.value_counts(normalize=True).to_dict()
    print("Class balance in full dataset:", balance)

if __name__ == "__main__":
    main()
