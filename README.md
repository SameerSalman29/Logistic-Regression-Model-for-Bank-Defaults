# Logistic-Regression-Model-for-Bank-Defaults

Predicting bank loan defaults using logistic regression with a confusion matrix, accuracy evaluation, and bias reduction techniques.
This machine learning project predicts the likelihood of loan defaults using historical-style (synthetic) financial and personal data.
It aims to assist financial institutions in making informed lending decisions by identifying potential defaulters.

## 📦 Project Structure

```
Logistic-Regression-Model-for-Bank-Defaults/
├── data/
│   └── loan_defaulting_credit_data.csv        # synthetic dataset (generated)
├── notebooks/
│   └── loan_default_analysis.ipynb            # EDA + training walkthrough
├── src/
│   ├── generate_data.py                       # create synthetic dataset
│   ├── train.py                               # train Logistic Regression, save model + scaler
│   └── predict_cli.py                         # simple CLI to predict from user inputs
├── models/
│   ├── model.pkl                              # trained Logistic Regression
│   ├── scaler.pkl                             # MinMaxScaler used during training
│   └── confusion_matrix.png                   # confusion matrix image
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Quickstart

1) **Create a virtual environment (optional but recommended)**
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

2) **Install dependencies**
```bash
pip install -r requirements.txt
```

3) **Generate synthetic dataset (optional: skips if `data/loan_defaulting_credit_data.csv` exists)**
```bash
python src/generate_data.py
```

4) **Train the model and save artifacts**
```bash
python src/train.py
```
This will output accuracy to the console and save `models/model.pkl`, `models/scaler.pkl`, and `models/confusion_matrix.png`.

5) **Make a prediction from the CLI**
```bash
python src/predict_cli.py
```
Follow the prompts to input applicant details. The script loads the saved scaler + model, validates inputs, scales correctly, and prints the prediction.

## 📊 Features Used
- Age
- Annual_Income
- Employment_Years
- Credit_Score
- Loan_Amount
- Loan_Term_Months
- Late_Payments

## 🧪 Evaluation
- Train/test split (60/40 by default)
- Accuracy score
- Confusion matrix visualization
- Class balancing with `class_weight='balanced'` and increased `max_iter`
- ### Evaluation Metrics
- Accuracy Score
- Confusion Matrix (saved as `models/confusion_matrix.png`)
- Classification Report (precision, recall, f1-score)
- ROC Curve with AUC (saved as `models/roc_curve.png`)


## ⚖️ Bias Reduction Notes
- Synthetic data includes **15% label noise** to simulate real-world exceptions and stress test the classifier.
- Class balancing is enabled in the logistic regression.
- MinMax scaling is used to avoid unintended weighting due to different feature scales.

## 📝 License
MIT License — see [LICENSE](LICENSE).

---

> Built from Sameer Salman's project report, reconstructed into a clean, GitHub-ready codebase.
