# 📉 Telco Customer Churn Prediction & Retention Optimization

<div align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white" alt="Matplotlib" />
</div>

## 📌 Executive Summary
Customer churn is a critical revenue-drain for telecommunications companies. This project builds an end-to-end machine learning pipeline that not only **predicts which customers are likely to cancel their service** but also **segments them into actionable cohorts** to drive targeted business retention strategies. 

By combining **Unsupervised Learning (K-Means)** for customer profiling with **Supervised Learning (Logistic Regression)** for risk probability scoring, this system bridges the gap between raw data and automated business intervention.

---

## 🛠️ Methodology & Technical Architecture

### 1. Data Preprocessing & Exploratory Data Analysis (EDA)
* **Data Cleaning:** Handled missing numeric data within financial features (`TotalCharges`) and engineered categorical variables using one-hot encoding for algorithm compatibility.
* **Visual Analytics:** Conducted EDA utilizing `seaborn` and `matplotlib` to isolate primary churn drivers, revealing inverse correlations between tenure duration and churn probability, alongside demographic risk factors (e.g., Senior Citizen churn rates).

### 2. Unsupervised Learning: Customer Segmentation
Applied **K-Means Clustering** to continuous financial and engagement metrics (`tenure`, `MonthlyCharges`, `TotalCharges`) to discover three distinct customer archetypes:
* **Segment 0 (High-Risk New Users):** Low tenure, high monthly charges. Require immediate high-touch onboarding.
* **Segment 1 (VIP / Core Revenue):** High tenure, high monthly charges. Require exclusive loyalty rewards to protect top-line revenue.
* **Segment 2 (Bargain Hunters):** Low tenure, low financial commitment. Ideal for automated, low-cost nurturing pipelines.

### 3. Supervised Learning: Predictive Modeling
Engineered a classification pipeline comparing multiple algorithms to predict binary churn outcomes.
* **Feature Scaling:** Applied `StandardScaler` to normalize distributions for gradient-based convergence.
* **Model Selection:** Trained a baseline **Decision Tree Classifier** (80% Accuracy) and an optimized **Logistic Regression Classifier** (82% Accuracy). 
* **Evaluation:** Evaluated algorithms utilizing precision, recall, F1-scores, and generated a **Receiver Operating Characteristic (ROC) Curve** to analyze the True Positive vs. False Positive trade-off.

### 4. Actionable Business Integration
The model outputs a precise **Churn Probability Score** for every active customer. 
* Identified a highly targeted list of "At-Risk" customers (Probability > 0.70).
* Cross-referenced these predictive risk scores with the K-Means cluster tags to trigger automated, segment-specific retention playbooks (e.g., freezing upselling, deploying billing credits, or routing to senior customer success managers).

---

## 📊 Key Results & Model Performance

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **82%** | **0.68** | **0.59** | **0.63** |
| Decision Tree (Baseline) | 80% | 0.62 | 0.60 | 0.61 |

* **AUC Score:** Sustained a high Area Under the Curve (AUC), confirming the model's robust capability to distinguish between retaining and churning customers across varying classification thresholds.

---

## 🚀 How to Run the Pipeline

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_GITHUB_HANDLE/YOUR_REPO_NAME.git](https://github.com/YOUR_GITHUB_HANDLE/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
