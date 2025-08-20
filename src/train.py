import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "loan_defaulting_credit_data.csv"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# Load or instruct to generate
if not DATA.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA}. Run `python src/generate_data.py` first.")

df = pd.read_csv(DATA)

features = ['Age', 'Annual_Income', 'Employment_Years', 'Credit_Score', 'Loan_Amount', 'Loan_Term_Months', 'Late_Payments']
X = df[features].copy()
y = df['Defaulted'].copy()

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.6, random_state=42, stratify=y)

# Model
model = LogisticRegression(class_weight='balanced', max_iter=50000, n_jobs=None, solver='lbfgs')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
auc_score = roc_auc_score(y_test, y_pred)
print(f"ROC AUC Score: {auc_score:.4f}")

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
roc_path = MODELS / "roc_curve.png"
plt.savefig(roc_path)
plt.close()
print(f"Saved ROC curve to {roc_path}.")


plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
cm_path = MODELS / "confusion_matrix.png"
plt.tight_layout()
plt.savefig(cm_path)
plt.close()

# Save artifacts
joblib.dump(model, MODELS / "model.pkl")
joblib.dump(scaler, MODELS / "scaler.pkl")

# Class balance
balance = y.value_counts(normalize=True).to_dict()
print("Class balance in full dataset:", balance)
print(f"Saved model to {MODELS/'model.pkl'} and scaler to {MODELS/'scaler.pkl'}.")
print(f"Saved confusion matrix image to {cm_path}.")
