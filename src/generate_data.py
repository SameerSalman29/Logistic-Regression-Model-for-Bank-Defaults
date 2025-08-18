import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "loan_defaulting_credit_data.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
n_samples = 2000

# Non-defaults (Class 0): Generally low-risk
non_defaults = pd.DataFrame({
    'Age': np.random.randint(25, 65, n_samples//2),
    'Annual_Income': np.clip(np.random.normal(70_000, 10_000, n_samples//2), 30_000, 150_000).astype(int),
    'Employment_Years': np.random.randint(5, 20, n_samples//2),
    'Credit_Score': np.clip(np.random.normal(700, 50, n_samples//2), 600, 850).astype(int),  # Good credit
    'Loan_Amount': np.clip(np.random.normal(50_000, 10_000, n_samples//2), 10_000, 100_000).astype(int),
    'Loan_Term_Months': np.random.choice([12, 24, 36], n_samples//2),  # Shorter terms
    'Late_Payments': np.random.poisson(1, n_samples//2).clip(0, 2),  # 0-2 late payments
    'Defaulted': 0
})

# Defaults (Class 1): Generally high-risk
defaults = pd.DataFrame({
    'Age': np.random.randint(20, 60, n_samples//2),
    'Annual_Income': np.clip(np.random.normal(100_000, 15_000, n_samples//2), 20_000, 100_000).astype(int),
    'Employment_Years': np.random.randint(0, 10, n_samples//2),
    'Credit_Score': np.clip(np.random.normal(580, 50, n_samples//2), 300, 650).astype(int),  # Poor credit
    'Loan_Amount': np.clip(np.random.normal(80_000, 20_000, n_samples//2), 20_000, 150_000).astype(int),
    'Loan_Term_Months': np.random.choice([12, 24, 36], n_samples//2),  # Longer terms
    'Late_Payments': np.random.poisson(4, n_samples//2).clip(2, 7),  # 3-7 late payments
    'Defaulted': 1
})

# Combine and add 15% noise (swap labels for some cases)
df = pd.concat([non_defaults, defaults]).sample(frac=1, random_state=42).reset_index(drop=True)
noise_mask = np.random.rand(len(df)) < 0.15  # 15% of rows will have flipped labels
df.loc[noise_mask, 'Defaulted'] = 1 - df.loc[noise_mask, 'Defaulted']

# Save
df.to_csv(OUT, index=False)
print(f"Saved synthetic dataset to {OUT}")
