# 🚢 Titanic Survival — Optimized Threshold & Light Meta-Ensemble

This repository contains my **Titanic survival prediction solution** for the [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition.  
The model uses an **interpretable, feature-rich ensemble** combining **Gradient Boosting** and **Logistic Regression**, with a **ROC-optimized threshold** for classification.

---

## 📊 Performance Overview

| Metric | Validation |
|:-------|:------------:|
| **Accuracy** | ~0.83 |
| **ROC-AUC** | ~0.86 |

This configuration consistently performs among the stronger baselines on the leaderboard while keeping a small, explainable model footprint.

---

## 🧠 Methodology

### Core Idea
Instead of relying on one model, the pipeline **blends a nonlinear Gradient Boosting model** with a **linear Logistic Regression** model.  
This combination captures both structured (tabular) relationships and additive effects.

### Workflow Summary
1. **Data cleaning & imputation**
   - Age imputed by `(Sex, Pclass)` group median.
   - Fare filled with global median.
   - Feature for missing ages (`IsAgeMissing`).
2. **Feature engineering**
   - Extracted social **titles** (`Mr`, `Mrs`, `Miss`, etc.).
   - Derived **deck structure**, flood risk proxies, and simulated evacuation timing.
   - Computed **family and group metrics** (size, type, fare per person).
   - Interaction terms: `Sex * Title`, `Fare * Deck`, and `FamilySize * Fare`.
3. **Model training**
   - **Gradient Boosting (500 estimators)** with isotonic calibration.
   - **Logistic Regression** with L2 regularization.
   - Weighted ensemble (`0.7 * GB + 0.3 * LR`).
4. **Threshold optimization**
   - Finds best classification cutoff from **ROC curve** (`max(TPR - FPR)`).
5. **Submission**
   - Outputs a ready-to-upload `titanic_submission.csv` for Kaggle.

---

## ⚙️ Environment

| Component | Library |
|------------|----------|
| **ML Core** | scikit-learn |
| **Data** | pandas, numpy |
| **Language** | Python ≥3.9 |

---

## 🗂️ Repository Structure

