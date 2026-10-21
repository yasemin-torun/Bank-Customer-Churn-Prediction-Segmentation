# Bank Customer Churn Prediction & Segmentation

## Project Overview

This project develops a machine learning-based framework for customer risk analysis and segmentation in the banking sector. It combines predictive modeling, customer segmentation, and campaign recommendations to support data-driven customer retention strategies.

## Objectives

- Develop and evaluate machine learning models for customer risk prediction.
- Compare model performance using multiple evaluation metrics.
- Identify customer segments using unsupervised learning.
- Generate targeted campaign recommendations based on customer risk and segment profiles.
- Present results through an interactive dashboard.

## Technologies & Tools

- **Language:** Python
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Machine Learning:** XGBoost, Artificial Neural Networks (ANN)
- **Class Imbalance:** SMOTE
- **Clustering & Dimensionality Reduction:** K-Means, PCA
- **Visualization & Dashboard:** Matplotlib, Seaborn, Streamlit

## Methodology

1. **Data Preprocessing:** Cleaned the data, handled missing values, encoded categorical variables, and prepared features for modeling.
2. **Model Development:** Trained and evaluated machine learning models, including XGBoost and ANN.
3. **Model Evaluation:** Compared performance using accuracy, precision, recall, F1-score, and ROC-AUC.
4. **Customer Segmentation:** Applied K-Means clustering and PCA to explore customer groups and their characteristics.
5. **Campaign Recommendations:** Developed a rule-based system to suggest targeted campaigns based on customer risk and segment profiles.
6. **Interactive Dashboard:** Presented model outputs, segmentation results, and campaign recommendations using Streamlit.

## Model Performance

| Model | Accuracy | Recall | ROC-AUC |
|---|---:|---:|---:|
| XGBoost | 87.76% | 45.65% | 0.7709 |
| ANN | 81.72% | 57.09% | 0.7665 |

XGBoost achieved higher accuracy, while ANN achieved higher recall. Both models were evaluated using multiple metrics to assess their predictive performance.

## Key Skills Demonstrated

- Data cleaning and preprocessing
- Feature engineering and categorical encoding
- Machine learning model development and evaluation
- Handling imbalanced datasets with SMOTE
- Customer segmentation and exploratory analysis
- Data visualization and interactive dashboard development
- Translating analytical results into business-oriented recommendations

## Project Context

Graduation project: *Customer Retention Strategy Through Churn Prediction and Behavioral Segmentation in the Banking Sector*.
