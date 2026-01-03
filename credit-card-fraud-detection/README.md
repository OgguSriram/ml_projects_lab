# Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques on a highly imbalanced dataset. The goal is to maximize fraud detection (recall) while maintaining acceptable false-positive rates.

The project follows a complete end-to-end ML pipeline including data analysis, preprocessing, imbalance handling, model tuning, explainability, and final evaluation.

---

## 📊 Dataset
- Source: European cardholders (September 2013)
- Transactions: 284,807
- Fraud cases: 492 (0.17%)
- Features: PCA-transformed variables (V1–V28), Time, Amount
- Target: `Class` (1 = Fraud, 0 = Normal)

> Due to confidentiality, original feature meanings are not available.

---

## 🧠 Methodology
1. Exploratory Data Analysis (EDA)
2. Stratified Train-Test Split
3. Feature Scaling (Time & Amount)
4. Baseline Logistic Regression
5. Imbalance Handling (Class Weights, SMOTE)
6. Random Forest Modeling
7. Threshold Tuning using Precision–Recall Curve
8. Hyperparameter Tuning (RandomizedSearchCV)
9. Feature Importance Analysis
10. Final Evaluation on Unseen Test Data

---

## ⚙️ Models Used
- Logistic Regression (baseline & weighted)
- Random Forest
- SMOTE for class imbalance
- Threshold tuning based on business constraints

---

## 📈 Final Results (Test Set)

| Metric | Fraud Class |
|------|------------|
| Precision | 0.78 |
| Recall | 0.85 |
| F1-score | 0.81 |

- Tuned threshold: **0.44**
- Model catches **85% of frauds**
- Balanced trade-off between fraud detection and false alarms

---

## 🔍 Explainability
Random Forest feature importance revealed that a small subset of PCA components (e.g., V14, V10, V4) dominate fraud detection, indicating fraud is driven by hidden behavioral patterns rather than transaction time alone.

---

## 📂 Project Structure
