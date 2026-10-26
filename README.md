# Telco Customer Segmentation & Churn Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aansikkaw/Customer-Segmentation-Churn-Prediction-System/blob/main/Customer_Segmentation_%26_Churn_Prediction_System.ipynb)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

A notebook that groups telecom customers into three segments with K-Means and predicts which customers are likely to cancel with Logistic Regression, compared against a Decision Tree. The two outputs are joined so each segment can receive a different retention action.

## Data

The public **Telco Customer Churn** dataset: 7,043 customers, 21 columns, 26.5% churn. The CSV (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) is **not stored in this repo**. Download it (for example from Kaggle, "Telco Customer Churn") and place it next to the notebook or in a `data/` folder.

This repo shows the method on public data. It does not contain any client data.

## Method

1. **Cleaning.** `TotalCharges` is text in the raw file because 11 rows are blank. All 11 belong to customers with tenure 0 (not yet billed), so they are set to 0.
2. **EDA.** Churn split, tenure against churn, and senior-citizen churn.
3. **Segmentation.** K-Means (k = 3) on standardised `tenure`, `MonthlyCharges` and `TotalCharges`. Segments are named from their profiles in code, because cluster numbers are arbitrary.
4. **Prediction.** One-hot encoding, an 80/20 split (`random_state=42`), scaling fitted on the training split, then Logistic Regression and a depth-5 Decision Tree.
5. **Checks.** 5-fold stratified cross-validation, an ablation without the segment feature, a threshold table, and ROC curves.

## Results

Test split (1,409 customers, 373 churners). Always predicting "no churn" scores 73.5% accuracy.

| Model | Accuracy | Churn precision | Churn recall | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.818 | 0.680 | 0.592 | 0.862 |
| Decision Tree (depth 5) | 0.798 | 0.625 | 0.598 | 0.837 |

Stratified 5-fold cross-validation (mean +/- standard deviation) gives a somewhat lower, more stable picture:

| Model | Accuracy | Precision | Recall | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.804 +/- 0.010 | 0.657 +/- 0.021 | 0.548 +/- 0.030 | 0.845 +/- 0.013 |
| Decision Tree (depth 5) | 0.792 +/- 0.012 | 0.638 +/- 0.027 | 0.500 +/- 0.051 | 0.824 +/- 0.014 |

The segment label (`Customer_Segment`) is included as a classifier feature but adds nothing: the Logistic Regression ROC-AUC is 0.862 with and without it.

### Segments

| Segment | Customers | Avg tenure | Avg monthly charge | Churn rate |
| :--- | :---: | :---: | :---: | :---: |
| High-Risk New Users | 2,689 | 13 months | $75 | 47.1% |
| VIP / Core Revenue | 2,200 | 59 months | $90 | 15.4% |
| Budget-Conscious | 2,154 | 29 months | $27 | 12.3% |

K = 3 was chosen because three groups are easy to act on, not because it scores best: silhouette is 0.48 for k = 2, 0.45 for k = 3 and 0.47 for k = 4.

### Alert threshold

Test split, Logistic Regression probabilities:

| Threshold | Customers flagged | Precision | Recall |
| :---: | :---: | :---: | :---: |
| 0.3 | 540 | 0.55 | 0.80 |
| 0.4 | 434 | 0.60 | 0.70 |
| 0.5 | 325 | 0.68 | 0.59 |
| 0.7 | 92 | 0.83 | 0.20 |

A 0.70 cut-off gives a short, accurate list but finds only about 1 in 5 churners. The right threshold depends on the cost of a retention offer compared with the value of a kept customer.

## Retention actions (hypotheses, not results)

The notebook lists a suggested action per segment (onboarding support for new high-risk users, loyalty rewards for core revenue customers, low-cost automated nurturing for budget customers). It scores the **test split only**; it does not send anything. Whether the actions reduce churn would need a randomised holdout.

## Run it

```bash
git clone https://github.com/aansikkaw/Customer-Segmentation-Churn-Prediction-System.git
cd Customer-Segmentation-Churn-Prediction-System
pip install -r requirements.txt
# place WA_Fn-UseC_-Telco-Customer-Churn.csv in this folder or in data/
jupyter notebook "Customer_Segmentation_&_Churn_Prediction_System.ipynb"
```

## Limitations and next steps

- One public dataset, and one split for the headline numbers (cross-validation above shows the spread).
- No hyperparameter tuning; the tree depth is a fixed choice.
- `tenure`, `MonthlyCharges` and `TotalCharges` are strongly related, so single coefficients are unstable and show association, not cause.
- K-Means is fitted on all customers (it never sees the churn label, but the segments are not computed from the training split only).
- Next: gradient boosting with SHAP explanations, probability calibration, survival analysis for tenure, and uplift modelling.
