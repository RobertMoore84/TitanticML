# 🚢 Titanic Survival — Optimized Threshold & Light Meta-Ensemble

**Score:** 0.75837  
**Leaderboard Position:** #3098  

This repository contains my **Titanic survival prediction solution** for the [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition.  
The model applies a **feature-engineered pipeline** combining **Gradient Boosting** and **Logistic Regression** into a **light, tuned meta-ensemble** with calibrated probabilities and a ROC-optimized classification threshold.

---

## 📊 Performance Overview

| Metric | Public Leaderboard | Validation (local) |
|:-------|:------------------:|:-----------------:|
| **Score (Accuracy)** | **0.75837** | ~0.83 |
| **Rank** | **#3098** | — |
| **Percentile** | ~Top 25–30% globally | — |

---

## 🧠 Approach

The focus was on building a **compact and interpretable model** that captures both social and structural relationships among passengers rather than relying purely on automated feature selection.  
This project demonstrates a **probabilistic ensemble** tuned using **ROC-AUC and optimal thresholding** for robust generalisation.

### Core Methodology
1. **Data Cleaning**
   - Age imputed using `(Sex, Pclass)` group medians.
   - Fare imputed with global median.
   - Added binary indicator for missing ages.
2. **Feature Engineering**
   - Extracted **titles** from names (`Mr`, `Miss`, `Mrs`, etc.).
   - Derived **deck structure** and conceptual flood risk/evacuation features.
   - Computed **group-based features** (family size, type, fare per person).
   - Created **interaction terms**:  
     - `Sex × Title`  
     - `FarePerPerson × DeckCode`  
     - `FamilySize × FarePerPerson`
3. **Modeling**
   - **Gradient Boosting (500 trees)** — nonlinear relationships.  
   - **Logistic Regression (calibrated)** — interpretable baseline.  
   - Combined via **weighted ensemble** (`0.7 * GB + 0.3 * LR`).
4. **Threshold Optimization**
   - Cutoff derived from **ROC curve** (maximising `TPR − FPR`).
5. **Submission**
   - Outputs a ready-to-upload `titanic_submission.csv`.

---

## ⚙️ Pipeline Summary

| Step | Description |
|------|--------------|
| **Feature Creation** | Domain-informed attributes: deck, flooding risk, social structure. |
| **Model Training** | Gradient Boosting + Logistic Regression. |
| **Calibration** | Isotonic method for probability reliability. |
| **Blending** | Weighted ensemble tuned on validation ROC-AUC. |
| **Thresholding** | ROC-optimized classification cutoff. |
| **Output** | Kaggle submission file (`titanic_submission.csv`). |

---

## 🗂️ Repository Structure

Titanic-Survival-Meta-Ensemble/
│
├── titanic_survival_metaensemble_v70.py # Main training + submission script
├── titanic_submission.csv # Example Kaggle output
├── train.csv # Kaggle dataset (not included)
├── test.csv # Kaggle dataset (not included)
└── README.md # This documentation

yaml
Copy code

---

## 🧪 How to Run

1. Download the **Titanic dataset** from Kaggle and place it in your working directory:
path_to_data/
├── train.csv
└── test.csv

go
Copy code
2. Run the script:
```bash
python titanic_survival_metaensemble_v70.py
The model will:

Train both Gradient Boosting and Logistic Regression models.

Blend predictions and tune threshold.

Generate a Kaggle-ready file: titanic_submission.csv.

📈 Example Output
yaml
Copy code
Optimal Threshold: 0.384
Accuracy: 0.8325
ROC-AUC: 0.8612

✅ Submission file created: titanic_submission.csv
💡 Insights
Deck location and evacuation proxies improved separation between survival groups.

Fare per person and family structure metrics were among the most predictive engineered features.

The calibrated ensemble provided more consistent probability outputs than a single GB model.

Version v70 achieved an effective balance between interpretability, generalisation, and score stability.

📦 Tools & Libraries
Category	Library
Machine Learning	scikit-learn
Data Wrangling	pandas, numpy
Language	Python ≥3.9

🧬 Author
Rob Moore
Ecology Biologist | Data Scientist
Kaggle handle: RobMoore

🏁 Summary
Developed a reproducible meta-ensemble pipeline achieving
0.75837 accuracy on Kaggle’s Titanic Survival leaderboard (Top ~25%).
Demonstrated strong feature engineering, model calibration, and ensemble optimization for structured tabular data.
