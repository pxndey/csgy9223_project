# csgy9223_project

## Responsible AI and the Law Project

Contributers:

- [Anushk Pandey](https://www.github.com/pxndey)  
- [Yang Sherpa](https://www.linkedin.com/in/yang-sherpa/)

Dataset Used:

[Diabetes Clinical Dataset](https://www.kaggle.com/datasets/ziya07/diabetes-clinical-dataset100k-rows)

## Methodology and Objectives

This project aims to experimentally investigate the impact of data imbalance and the effectiveness of various bias mitigation techniques (preprocessing and inprocessing) on model fairness.

### 1. Data Preparation

The initial dataset (approximately 100,000 entries) are partitioned and processed as follows:

- **Holdout/Validation Set (30,000 Entries):** A balanced subset of the original data, strictly reserved and unseen by any model during training or hyperparameter tuning. This set will be used for final, unbiased evaluation of all tested approaches.
- **Training/Testing Sets (Remaining 70,000 Entries):** This portion will be used to create three scenario-based datasets for training and initial testing:
  - **Dataset A (Baseline):** A **racially balanced** dataset designed to establish a baseline performance and fairness benchmark achievable with the inherent properties of the data.
  - **Dataset B (Caucasian-Biased):** A dataset intentionally skewed to be overrepresented by a specific demographic (e.g., Caucasian), simulating a scenario where data collection was conducted in a non-representative environment (e.g., a predominantly white neighborhood).

### 2. Model Selection

We utilize two foundational machine learning models known for their interpretability and widespread use:

- Logistic Regression (LR)
- Random Forest (RF)

However, for Adversarial Debiasing, we use a small Multi-Layer Perceptron to simulate training a classifier and an adversary

### 3. Experimental Plan

The following steps are executed sequentially on the prepared datasets:

| Step | Dataset(s) | Technique | Objective |
| :--- | :--- | :--- | :--- |
| **3.1** | A, B | LR, RF Training | Establish initial performance and fairness metrics for the baseline and biased datasets. |
| **3.2** | B | **SMOTE** (Synthetic Minority Over-sampling Technique) | Investigate the effect of traditional **artificial upsampling** on mitigating bias in the data-scarce minority group(s) in the biased set. |
| **3.3** | B | **Adversarial Debiasing** | Evaluate the efficacy of advanced **in-processing debiasing methods** to enforce fairness constraints during model training. |

### 4. Evaluation Metrics

All models are evaluated using the Holdout/Validation Set on the following key metrics:

- **Overall Performance:** F1-Score.
- **Fairness Metrics:**
  - **False Negative Rate (FNR) Parity:** Assessing the equality of FNR across all protected attributes (races).
  - **Equalized Odds:** Requiring equal true positive rates (TPR) and false positive rates (FPR) across groups.
