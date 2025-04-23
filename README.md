[![](images/image1.png){style="width: 720.00px; height: 960.00px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);"}]{style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 720.00px; height: 960.00px;"}

------------------------------------------------------------------------

[![](images/image2.png){style="width: 705.67px; height: 1157.47px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);"}]{style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 705.67px; height: 1157.47px;"}

[AUTISM PREDICTION USING MACHINE LEARNING]{.c23}

1.  [Executive Summary]{.c1}

[This project explores the application of machine learning algorithms in
predicting autism spectrum disorder (ASD) using demographic data and
standardized questionnaire responses. The objective is to develop a
predictive model that can support early screening interventions,
particularly in low-resource settings where traditional diagnostic tools
may be inaccessible. The data underwent extensive pre-processing and was
used to train three classification models: Decision Tree, Random Forest,
and XGBoost. Among these, the Random Forest model exhibited the highest
accuracy and robustness, making it the most suitable for deployment.
This report details the problem definition, data processing steps, model
development, evaluation, and recommendations for future
improvement.]{.c3 .c2}

[2. Problem Statement]{.c1}

[Autism Spectrum Disorder (ASD) is a complex neurodevelopmental
condition that affects communication, behaviour, and social interaction.
Early identification and intervention are critical for improving
outcomes, yet access to diagnostic services is limited in many regions.
Traditional diagnostic methods are time-consuming, expensive, and
require clinical expertise, making early screening challenging,
particularly in low-resource settings.]{.c3 .c2}

[This project aims to address this gap by developing an automated
machine learning-based classification model that can predict the
likelihood of ASD in individuals using questionnaire responses and
demographic data. By leveraging data-driven techniques, the model seeks
to provide an accurate, scalable, and accessible tool for preliminary
autism screening to support timely clinical follow-up and
intervention.]{.c3 .c2}

[3. Target Users]{.c1}

-   [Healthcare providers]{.c3 .c2}
-   [School counsellors]{.c3 .c2}
-   [Parents and guardians]{.c3 .c2}
-   [Policymakers in public health]{.c3 .c2}
-   [Developers of health screening software]{.c3 .c2}

[4. Main Objective]{.c1}

[To develop a machine learning-based tool capable of predicting autism
in individuals based on questionnaire responses and demographic
features.]{.c3 .c2}

[4.1 Specific Objectives]{.c1}

-   [To apply data pre-processing techniques to clean, transform, and
    balance the dataset for effective training.]{.c3 .c2}
-   [To train, evaluate and compare multiple machine learning models
    (Decision Tree, Random Forest, and XGBoost) for their effectiveness
    in classifying ASD.]{.c3 .c2}
-   [To select the best-performing model and provide performance
    metrics, including confusion matrices, to interpret its predictive
    capability.]{.c3 .c2}
-   [To export and save the final model for future use in clinical or
    screening applications.]{.c3 .c2}

[5. Proposed Solution]{.c1}

[Develop a machine learning-based predictive model trained on a
structured dataset of questionnaire responses and demographic data. The
solution includes applying pre-processing techniques, model selection,
training, cross-validation, and evaluation, followed by saving the best
model for real-world deployment in screening environments.]{.c3 .c2}

[6. Data Collection and Pre-processing]{.c1}

[6.1 Data Source]{.c1}

[The dataset used for autism prediction was obtained from a structured
CSV file from Kaggle. It contains behavioural and demographic variables
collected through standardized autism screening questionnaires. These
include binary responses (Yes/No or 0/1) to ten questions, alongside
metadata such as age, gender, ethnicity, and family history of ASD.]{.c3
.c2}

[6.2 Data Collection]{.c1}

[The dataset was collected from anonymized self-reported surveys or
clinical records targeting early autism detection. The data represents a
balanced mix of ASD-positive and ASD-negative individuals but required
further rebalancing due to slight class imbalance.]{.c3 .c2}

[6.3 Initial Structure and Importation]{.c1}

[Using Python's pandas, the CSV file was loaded into a DataFrame for
analysis. Exploratory commands such as .info (),. head(), .shape(), and
.describe() were used to assess column types, detect missing values, and
understand distributions.]{.c3 .c2}

[6.3.1 Outputs from EDA:]{.c1}

-   [Shape:]{.c4}[ (800, 22)]{.c3 .c2}
-   [Integer columns:]{.c4}[ 12 (e.g., A1_Score to A10_Score, ID,
    Class/ASD)]{.c3 .c2}
-   [Float columns:]{.c4}[ 2 (age, result)]{.c3 .c2}
-   [Object columns:]{.c4}[ 8 (e.g., gender, ethnicity, jaundice,
    autism, country_of_res, used_app_before, age_desc, relation)]{.c3
    .c2}
-   [.info():]{.c4}[ All columns are non-null with appropriate data
    types]{.c3 .c2}
-   [.describe():]{.c4}[ Provided statistical summaries for numeric
    columns like age and result, with Class/ASD mean \~0.20 indicating
    class imbalance]{.c3 .c2}
-   [.head():]{.c4}[ Displayed the first five rows to confirm data
    format]{.c3 .c2}

[6.4 Insights from Exploratory Data Analysis (EDA)]{.c1}

-   [Outliers Detected:]{.c4}[ Numerical features such as age and result
    exhibited a few outliers, particularly at the higher ends of the
    distribution. These were retained as they appeared plausible and did
    not significantly skew model performance.]{.c3 .c2}
-   [Class Imbalance Observed:]{.c4}[ The target variable (Class/ASD) is
    imbalanced, with fewer positive autism cases compared to negative
    ones. This warranted the use of SMOTE during pre-processing. Some
    categorical features, such as used_app_before and family history of
    autism, also showed skewed distributions.]{.c3 .c2}
-   [Low Correlation Between Features:]{.c4}[ Correlation analysis
    revealed that there were no highly correlated numerical features
    that would necessitate dropping due to redundancy.]{.c3 .c2}
-   [Categorical Feature Transformation:]{.c4}[ Label encoding was
    applied to all categorical fields (e.g., gender, ethnicity,
    jaundice, relation) to convert them into a format suitable for model
    training.]{.c3 .c2}
-   [Visualization Tools Used:]{.c4}

```{=html}
<!-- -->
```
-   [Bar plot showing class distribution reveals imbalance.]{.c3 .c2}
-   [Heatmap shows correlation between features.]{.c3 .c2}

[6.5 Data Cleaning Steps]{.c1}

-   [Dropped irrelevant features: ID and age_desc were excluded from
    analysis as they had no predictive value.]{.c3 .c2}
-   [Converted age to integer to ensure consistency.]{.c3 .c2}
-   [Handled any missing values by replacing incomplete records (no
    imputation was necessary due to minimal loss).]{.c3 .c2}

[6.6 Feature Engineering]{.c1}

-   [Label Encoding:]{.c4}[ Applied to categorical fields like gender,
    ethnicity, and family history, converting them into machine-readable
    numeric form.]{.c3 .c2}
-   [Target Transformation:]{.c4}[ The Class/ASD label was standardized
    to binary format for classification purposes.]{.c3 .c2}
-   [Normalization:]{.c4}[ Numerical features like age were kept on a
    natural scale given their limited range, avoiding excessive
    transformations.]{.c3 .c2}

[6.7 Addressing Class Imbalance]{.c1}

[The dataset initially had a slight skew toward the non-ASD class. To
improve model learning, SMOTE (Synthetic Minority Oversampling
Technique) from the imblearn library was used to synthetically generate
minority class samples, resulting in a balanced dataset ready for
supervised learning.]{.c3 .c2}

[]{.c3 .c2}

[]{.c3 .c2}

[7. Model Development ]{.c1}

[7.1 Model Selection and Hyperparameter Tuning]{.c1}

[To identify the most effective algorithm for autism prediction, three
supervised learning classifiers were evaluated:]{.c2 .c18}

-   [Decision Tree Classifier:]{.c4}[ A basic yet interpretable model
    that splits data based on feature thresholds.]{.c3 .c2}
-   [Random Forest Classifier:]{.c4}[ An ensemble of decision trees that
    improves performance through bagging and reduces overfitting.]{.c3
    .c2}
-   [XGBoost Classifier:]{.c4}[ A gradient boosting model known for
    superior accuracy and robustness in handling imbalanced data.]{.c3
    .c2}

[7.2 Training Strategy]{.c1}

-   [The pre-processed dataset was split into training and testing sets
    using an 80:20 ratio.]{.c3 .c2}
-   [SMOTE was applied to the training set to address class imbalance by
    synthetically oversampling the minority class (ASD positive).]{.c3
    .c2}
-   [Each model was trained on the same balanced training data to ensure
    a fair comparison.]{.c3 .c2}

[7.3 Validation Approach]{.c1}

-   [Cross-validation (CV) was performed using 5-fold CV to minimize
    bias and variance in performance estimation.]{.c3 .c2}
-   [CV ensured the model generalized well to unseen data and prevented
    overfitting.]{.c3 .c2}

[7.3.1 Cross-Validation Results]{.c1}

-   [Decision Tree:]{.c4}[ Cross-validation accuracy = 0.86]{.c3 .c2}
-   [Random Forest:]{.c4}[ Cross-validation accuracy = 0.91]{.c3 .c2}
-   [XGBoost:]{.c4}[ Cross-validation accuracy = 0.90]{.c3 .c2}

[7.4 Evaluation Metrics]{.c1}

-   [Accuracy:]{.c4}[ Percentage of correctly predicted instances.]{.c3
    .c2}
-   [F1-Score:]{.c4}[ Harmonic mean of precision and recall,
    particularly important due to class imbalance.]{.c3 .c2}
-   [Confusion Matrix:]{.c4}[ Provided a breakdown of true positives,
    false positives, true negatives, and false negatives.]{.c3 .c2}

[7.5 Model Comparison and Evaluation]{.c1}

[Three supervised learning models (Decision Tree, Random Forest, and
XGBoost) were trained and evaluated using 5-fold cross-validation and
standard classification metrics.]{.c3 .c2}

  -------------------------- ----------------- -------------------- --------------------- --------------------- ---------------------
  [Model]{.c1}               [Accuracy]{.c1}   [CV Accuracy]{.c1}   [Precision]{.c1}      [Recall]{.c1}         [F1-Score]{.c1}
  [Decision Tree]{.c3 .c2}   [0.86]{.c3 .c2}   [0.86]{.c3 .c2}      [Moderate]{.c3 .c2}   [Moderate]{.c3 .c2}   [Moderate]{.c3 .c2}
  [Random Forest]{.c3 .c2}   [0.91]{.c3 .c2}   [0.92]{.c3 .c2}      [High]{.c3 .c2}       [High]{.c3 .c2}       [High]{.c3 .c2}
  [XGBoost]{.c3 .c2}         [0.90]{.c3 .c2}   [0.90]{.c3 .c2}      [High]{.c3 .c2}       [Balanced]{.c3 .c2}   [High]{.c3 .c2}
  -------------------------- ----------------- -------------------- --------------------- --------------------- ---------------------

[7.6 Evaluation Summary]{.c1}

-   [Decision Tree:]{.c4}[ Performed adequately but exhibited higher
    false positives and lower generalization.]{.c3 .c2}
-   [Random Forest:]{.c4}[ Achieved the highest overall performance,
    especially with tuned parameters (bootstrap=False, max_depth=20,
    n_estimators=50). It showed superior accuracy and balance across all
    metrics.]{.c3 .c2}
-   [XGBoost:]{.c4}[ Performed closely to Random Forest, with strong
    balance and robustness, making it a viable alternative where
    interpretability is needed.]{.c2 .c3}

------------------------------------------------------------------------

[]{.c3 .c2}

[8. Recommendations]{.c1}

[Given the success of the model, several recommendations were made.
]{.c2}

1.  [In the short term, it is recommended to deploy the Random Forest
    model via a user-friendly web application using frameworks like
    Streamlit or Flask. This would allow pilot testing in clinics or
    schools. ]{.c14 .c2}
2.  [In the medium term, the model should be retrained with more diverse
    and real-world data to improve generalizability. Incorporating
    interpretability tools such as SHAP or LIME can enhance transparency
    and trust in the model's predictions. ]{.c14 .c2}
3.  [Long-term recommendations include integrating the tool into
    national health information systems, in Kenya where there is limited
    access to pediatric specialists. There is also potential to enhance
    the model by incorporating audio-visual data, such as speech
    patterns or facial expressions, to create a multi-modal diagnostic
    aid. Finally, publishing the tool as open-source software can
    accelerate its adoption and customization by developers and
    researchers worldwide]{.c14 .c2}
4.  []{.c14 .c2}

[9.  Conclusion]{.c1}

[Based on cross-validation performance and confusion matrix evaluation,
the Random Forest model was selected as the best model for deployment
due to its high predictive accuracy and balanced classification
performance. This model, when exported and deployed, can serve as an
assistive tool for early screening of ASD cases, especially in community
or primary healthcare settings.]{.c3 .c2}

[]{.c3 .c2}

<div>

[]{.c14 .c2}

[]{.c14 .c19}

</div>
