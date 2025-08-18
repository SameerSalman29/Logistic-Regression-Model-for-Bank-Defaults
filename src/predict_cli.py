from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "model.pkl"
SCALER_PATH = MODELS / "scaler.pkl"

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    raise FileNotFoundError("Model or scaler not found. Run `python src/train.py` first.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def get_int(prompt, lo=None, hi=None):
    while True:
        try:
            val = int(input(prompt))
            if lo is not None and val < lo:
                print(f"Value must be >= {lo}"); continue
            if hi is not None and val > hi:
                print(f"Value must be <= {hi}"); continue
            return val
        except ValueError:
            print("Please enter a valid integer.")

def get_float(prompt, lo=None, hi=None):
    while True:
        try:
            val = float(input(prompt))
            if lo is not None and val < lo:
                print(f"Value must be >= {lo}"); continue
            if hi is not None and val > hi:
                print(f"Value must be <= {hi}"); continue
            return val
        except ValueError:
            print("Please enter a valid number.")

print("\nEnter user details to check default risk:")
Age = get_int("Age: ", lo=18, hi=100)
Annual_Income = get_float("Annual Income (AED per YEAR): ", lo=0)
Employment_Years = get_int("Years Employed: ", lo=0)
# basic plausibility check
if (Age - Employment_Years) < 16:
    print("Warning: Employment years seem inconsistent with age.")
Credit_Score = get_int("Credit Score (300-850): ", lo=300, hi=850)
Loan_Amount = get_float("Loan Amount (AED): ", lo=0)
Loan_Term_Months = get_int("Loan Term (months, e.g., 12/24/36): ", lo=1)
Late_Payments = get_int("Number of Late Payments: ", lo=0)

features = np.array([[
    Age, Annual_Income, Employment_Years, Credit_Score, Loan_Amount, Loan_Term_Months, Late_Payments
]], dtype=float)

features_scaled = scaler.transform(features)
pred = model.predict(features_scaled)[0]
proba = model.predict_proba(features_scaled)[0][1]

if pred == 1:
    print(f"\nPrediction: LIKELY TO DEFAULT (probability={proba:.2%})")
else:
    print(f"\nPrediction: NOT LIKELY TO DEFAULT (probability={proba:.2%})")
