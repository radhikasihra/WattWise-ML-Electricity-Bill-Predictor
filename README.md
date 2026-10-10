# WattWise ML Electricity Bill Predictor

**WattWise | Smart Electricity Bill Prediction Using Machine Learning**

**Predicting household electricity expenses using statistical analysis, feature engineering, and comparative machine learning.**

## Project Overview

Electricity costs vary across households based on property characteristics, occupancy, and appliance usage. This project explores those relationships and develops regression models to estimate a household's electricity bill from eight household-level features.

The study combines exploratory data analysis, statistical hypothesis testing, and a comparison of ten regression approaches to identify an effective predictive model.

## Problem Statement

Household electricity expenses can be difficult to anticipate without understanding the underlying drivers of consumption. The objective is to **predict the electricity bill (`amount_paid`)** using readily available household attributes and examine which variables show meaningful relationships with bill amounts.

## Dataset & Features

- **Dataset size:** 2,000 records, 11 original columns
- **Prediction target:** `amount_paid`
- **Model inputs:** Number of rooms (`num_rooms`), household members (`num_people`), house area (`housearea`), air conditioner ownership (`is_ac`), TV ownership (`is_tv`), flat indicator (`is_flat`), number of children (`num_children`), and urban location (`is_urban`)
- **Data quality:** No missing values or duplicate records were found in the initial data inspection

Monthly income was examined during exploratory analysis but was not included in the final predictive feature set. An `appliance_score` feature was explored separately, but was also not included among the final model inputs.

## Analytical Approach

- **Exploratory analysis:** Descriptive statistics, distributions, outlier inspection, and relationships between household features and electricity bills
- **Preprocessing:** IQR-based removal of outliers in the target, label encoding of categorical indicators, and standardization of selected input features
- **Statistical testing:** Independent two-sample t-test for AC ownership; Pearson and Spearman correlation analyses
- **Model development:** 80/20 train–test split and comparison of linear, tree-based, nearest-neighbor, support-vector, boosting, and ensemble regressors
- **Evaluation:** Mean Squared Error (MSE) and coefficient of determination (R²)

## Model Performance

Recorded test-set metrics from the notebook:

| Model | R² Score | MSE |
|---|---:|---:|
| **Decision Tree Regressor** | **0.9543** | **13,238.29** |
| XGBoost Regressor | 0.9517 | 13,985.61 |
| Random Forest Regressor | 0.9476 | 15,169.04 |
| Voting Regressor | 0.9217 | 22,651.95 |
| Lasso Regression | 0.8874 | 32,593.73 |
| Ridge Regression | 0.8874 | 32,588.93 |
| Linear Regression | 0.8874 | 32,593.85 |
| Hybrid Voting Regressor | 0.8658 | 38,838.58 |
| KNN Regressor | 0.8635 | 39,515.90 |
| Support Vector Regressor | 0.1606 | 242,977.30 |

**Best recorded model: Decision Tree Regressor**, with an **R² of 0.9543**, explaining approximately **95.43% of the variation** in the held-out test target. Its recorded MSE was **13,238.29**. These figures are from the notebook's saved outputs and have not been independently revalidated.

## Key Findings

- **Air conditioning:** Electricity bills differed significantly between AC and non-AC households in the notebook's two-sample t-test (p ≈ 2.26 × 10⁻⁷³). This is an association, not proof of causation.
- **Appliance usage:** The exploratory `appliance_score` (AC + TV indicators) showed a moderate positive correlation with bill amount (r ≈ 0.424).
- **House area:** A weak but statistically significant positive relationship with the bill was observed (Pearson r ≈ 0.089).
- **Monthly income:** The recorded correlation with bill amount was close to zero (r ≈ 0.023).
- **Model selection:** The Decision Tree produced the strongest recorded R² among the evaluated models, narrowly ahead of XGBoost and Random Forest.

## Technology Stack

**Python · Pandas · NumPy · Matplotlib · Seaborn · SciPy · Scikit-learn · XGBoost · Jupyter Notebook**

## Business Relevance

A household-level bill estimation model could support early cost estimates, budgeting discussions, and exploratory analysis of the factors associated with electricity expenses. Real-world use would require validation on new data from the relevant utility, region, and billing period.

## Project Limitations

The notebook uses a single train–test split; cross-validation and external dataset validation were not performed. Outliers were filtered before the split, and the model results should therefore be viewed as exploratory rather than production-validated. The notebook compares models without hyperparameter optimization, and reported metrics may vary across runs because some estimators lack a fixed random seed.

## Repository Contents

- [`Electricity_bill_prediction_cleaned.ipynb`](Electricity_bill_prediction_cleaned.ipynb) — Full analysis, visualizations, statistical tests, model comparison, and example prediction

---

*Portfolio project demonstrating exploratory data analysis, statistical reasoning, regression modeling, and evidence-based model comparison.*
