# Project Proposal: 
## Predicting Undiagnosed Mental Illness Using Random Forest on MIMIC-IV

**Team Health Heroes**

## 1. Introduction

Mental illness is often underdiagnosed, especially in populations affected by socioeconomic disparities, cultural stigma, and healthcare accessibility barriers. Many individuals experiencing mental health conditions seek medical care for somatic complaints (e.g., chronic pain, gastrointestinal distress, fatigue), yet their underlying mental illness remains undiagnosed.

This project aims to develop a Random Forest (RF) machine learning model using the MIMIC-IV electronic medical record (EMR) dataset to:
1. Predict undiagnosed mental illness based on clinical conditions, healthcare utilization, and social determinants of health (SDOH).
2. Analyze the role of SDOH in the diagnostic gap, identifying potential biases and disparities.

By focusing on high precision, we aim to minimize false positives, ensuring that flagged patients are truly at risk of undiagnosed mental illness.

## 2. Research Questions

We have three primary research questions:

1. Can we accurately predict cases of suspected undiagnosed mental illness using structured EMR data from MIMIC-IV?
2. Which clinical and SDOH factors contribute most to the risk of being undiagnosed?
3. Are there systematic disparities (e.g., by race, insurance type, language barriers) in the likelihood of being undiagnosed?

In order to answer these questions, we need access to protected health information.

## 3. Data Source: MIMIC-IV Dataset

MIMIC-IV is a publicly available, de-identified EMR dataset containing information on patients admitted to Beth Israel Deaconess Medical Center. The dataset includes:

- **Demographics:** Age, sex, race, insurance status.
- **Clinical Diagnoses:** ICD-10-coded conditions, including both somatic and psychiatric diagnoses.
- **Medications:** Prescription records, which can indicate psychiatric treatment.
- **Healthcare Utilization:** Frequency of hospitalizations, emergency department (ED) visits, and readmissions.

While MIMIC-IV does not provide direct social network data, SDOH proxies such as insurance type, race/ethnicity, and language preference can help analyze disparities in mental health diagnosis.

## 4. Methodology

### 4.1. Label Definition: Identifying “Undiagnosed Mental Illness”

We define the target variable (undiagnosed_suspected) as follows:
- **Positive Case (1) – Suspected Undiagnosed:**
    - Patients with repeated ED visits, polypharmacy, or chronic somatic complaints (e.g., IBS, fibromyalgia, migraines) but no recorded mental health diagnosis (e.g., depression, anxiety).
    - Patients prescribed sedatives, painkillers, or GI medications without an associated psychiatric diagnosis.
- **Negative Case (0) – No Suspicion of Undiagnosed Mental Illness:**
    - Patients with neither a mental health diagnosis nor strong indicators of undiagnosed conditions.

Since this is an imperfect proxy, we will validate it through clinical expert review and sensitivity analysis.

### 4.2. Feature Engineering

**A. Clinical Features**
- Comorbid Conditions: Cardiovascular, respiratory, metabolic, pain-related conditions.
- Medication Usage: Antidepressants, anxiolytics, analgesics, sedatives.
- Healthcare Utilization Patterns:
    - Number of ED visits, readmissions.
    - Length of stay, ICU admissions.

**B. Social Determinants of Health (SDOH) Proxies**
- Race/Ethnicity: To explore disparities in diagnosis rates.
- Insurance Type: Proxy for socioeconomic status.
- Language Preference: Proxy for potential cultural barriers.
- Admission Type: Emergency vs. elective (indicating healthcare access patterns).
- Discharge Disposition: Home vs. nursing facility (indicating social support).

**C. Derived Features**
- Somatic Symptom Burden: Count of pain-related, GI, and fatigue-related diagnoses.
- Polypharmacy Flag: Patients prescribed multiple drugs across different symptom categories.
- Frequent Utilizer Flag: Patients in the top percentile of ED visits and readmissions.

### 4.3. Model Development: Random Forest Classifier

We will use Random Forest (RF) due to its:
- Suitability for tabular EMR data
- Robustness to missing values and nonlinearity
- High interpretability through SHAP values and feature importance

**Training and Evaluation Plan**
- **Target Variable:** undiagnosed_suspected (Binary classification: 1 = suspected undiagnosed, 0 = not suspected).
- **Cross-Validation:** Stratified 5-fold cross-validation to handle class imbalance.
- **Evaluation Metrics:**
    - Primary Metric: Precision (minimizing false positives)
    - Secondary Metrics: Recall, F1-score, AUROC
- **Threshold Tuning:** Adjusting RF probability thresholds to optimize precision.

**Hyperparameter Tuning:**
- Grid Search to optimize:
    - n_estimators (number of trees)
    - max_depth (tree depth)
    - min_samples_split (minimum data points in a node)

## 5. Model Interpretation & Disparity Analysis

To explain why certain patients are flagged as undiagnosed, we will use:
- **Feature Importance Analysis:**
    - Gini Importance (built-in RF feature importance).
    - Permutation Importance to assess which variables impact predictions the most.
- **SHAP Analysis (Shapley Additive Explanations):**
    - Identify which SDOH factors contribute most to being undiagnosed.
    - Compare SHAP distributions across race/ethnicity and insurance groups to detect systemic disparities.
- **Subgroup Disparity Analysis:**
    - Are racial minorities disproportionately predicted as undiagnosed?
    - Does insurance status influence prediction probability?

## 6. Validation and Sensitivity Analysis
- **Clinical Expert Validation:** Collaborate with psychiatrists and physicians to review model predictions.
- **Sensitivity Analysis:**
    - Test different definitions of undiagnosed_suspected to ensure robustness.
    - Compare against external datasets if available.

## 7. Ethical Considerations
- **Bias Mitigation:** Monitor disparities to avoid reinforcing healthcare inequities.
- **Transparency & Interpretability:** Prioritize explainable ML techniques.
- **Data Privacy:** MIMIC-IV is already de-identified, but strict handling protocols will be followed.

## 8. Expected Outcomes & Impact
- **Predictive Model for Undiagnosed Mental Illness:** A high-precision classifier capable of identifying at-risk patients.
- **Insight into SDOH Contributions:** Understanding how race, insurance, and socioeconomic factors influence diagnostic gaps.
- **Policy Implications:** Findings could inform targeted interventions to reduce underdiagnosis in vulnerable populations.

## 9. Project Timeline

| Phase   | Tasks                              | Timeline |
|---------|------------------------------------|----------|
| Phase 1 | Data Extraction & Preprocessing    | Month 1  |
| Phase 2 | Feature Engineering & Labeling     | Month 2  |
| Phase 3 | Model Training & Evaluation        | Month 3  |
| Phase 4 | Interpretation & Disparity Analysis| Month 4  |
| Phase 5 | Validation & Sensitivity Testing   | Month 5  |
| Phase 6 | Reporting & Policy Implications    | Month 6  |

## 10. Conclusion

This study will use Random Forests on MIMIC-IV to predict undiagnosed mental illness and analyze SDOH disparities. The findings will provide insights into systemic biases in mental health diagnosis, supporting interventions for equitable healthcare access.
