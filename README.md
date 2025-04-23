# Autism Prediction Using Machine Learning



---

## 1. Executive Summary

This project explores the application of machine learning algorithms in predicting autism spectrum disorder (ASD) using demographic data and standardized questionnaire responses. The objective is to develop a predictive model that can support early screening interventions, particularly in low-resource settings where traditional diagnostic tools may be inaccessible. 

Three classification models were trained: Decision Tree, Random Forest, and XGBoost. Among these, the Random Forest model exhibited the highest accuracy and robustness, making it the most suitable for deployment.

This report details the problem definition, data processing steps, model development, evaluation, and recommendations for future improvement.

---

## 2. Problem Statement

Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition that affects communication, behaviour, and social interaction. Early identification and intervention are critical for improving outcomes, yet access to diagnostic services is limited in many regions. Traditional diagnostic methods are time-consuming, expensive, and require clinical expertise, making early screening challenging.

This project aims to address this gap by developing an automated machine learning-based classification model that can predict the likelihood of ASD using questionnaire responses and demographic data.

---

## 3. Target Users

- Healthcare providers
- School counsellors
- Parents and guardians
- Policymakers in public health
- Developers of health screening software

---

## 4. Main Objective

To develop a machine learning-based tool capable of predicting autism in individuals based on questionnaire responses and demographic features.

### 4.1 Specific Objectives

- Apply data pre-processing techniques.
- Train, evaluate and compare Decision Tree, Random Forest, and XGBoost models.
- Select the best-performing model and interpret its predictive capability.
- Export and save the final model for future use.

---

## 5. Proposed Solution

Develop a machine learning-based predictive model trained on a structured dataset. The solution includes data pre-processing, model selection, training, cross-validation, evaluation, and saving the best model.

---

## 6. Data Collection and Pre-processing

### 6.1 Data Source

The dataset was obtained from Kaggle, containing behavioural and demographic variables collected via standardized autism screening questionnaires.

### 6.2 Data Collection

The dataset was collected from anonymized self-reported surveys or clinical records targeting early autism detection. Minor class imbalance was observed.

### 6.3 Initial Structure and Importation

Using Python's `pandas`, the CSV was loaded and inspected using `.info()`, `.head()`, `.shape()`, and `.describe()`.

#### Outputs from EDA:

- Shape: (800, 22)
- Integer columns: 12
- Float columns: 2
- Object columns: 8
- All columns non-null
- Class/ASD mean ~0.20 (imbalance noted)

### 6.4 Insights from EDA

- **Outliers:** Present in age and result but retained.
- **Class Imbalance:** Addressed with SMOTE.
- **Low Correlation Between Features:** No redundancy.
- **Categorical Features:** Label encoded.
- **Visualizations:** Class distribution bar plot, heatmap.

### 6.5 Data Cleaning Steps

- Dropped irrelevant features (e.g. ID, age_desc)
- Converted age to integer
- Handled minimal missing values

### 6.6 Feature Engineering

- Label encoding for categorical features
- Standardized target labels
- Numerical features kept on natural scale

### 6.7 Addressing Class Imbalance

Used SMOTE to synthetically generate minority class samples, resulting in a balanced dataset.

---

## 7. Model Development

### 7.1 Model Selection

- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

### 7.2 Training Strategy

- Train-test split: 80:20
- SMOTE applied to training set
- Same data used for all models

### 7.3 Validation Approach

- 5-fold cross-validation

#### Cross-Validation Results

- Decision Tree: 0.86
- Random Forest: 0.91
- XGBoost: 0.90

### 7.4 Evaluation Metrics

- Accuracy
- F1-Score
- Confusion Matrix

### 7.5 Model Comparison

| Model           | Accuracy | CV Accuracy | Precision | Recall   | F1-Score |
|----------------|----------|-------------|-----------|----------|----------|
| Decision Tree  | 0.86     | 0.86        | Moderate  | Moderate | Moderate |
| Random Forest  | 0.91     | 0.92        | High      | High     | High     |
| XGBoost        | 0.90     | 0.90        | High      | Balanced | High     |

### 7.6 Evaluation Summary

- Decision Tree: High false positives.
- Random Forest: Best performance with tuned hyperparameters.
- XGBoost: Close performance, good balance.

---

## 8. Recommendations

1. Deploy Random Forest via a web app (e.g. Streamlit).
2. Retrain with diverse data, add SHAP/LIME for interpretability.
3. Long-term: Integrate with national systems, explore multi-modal enhancements, open-source the tool.

---

## 9. Conclusion

Random Forest was selected due to its high predictive accuracy and balanced performance. It offers potential for supporting early ASD screening in primary care settings.
