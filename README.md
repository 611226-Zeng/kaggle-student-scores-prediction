# Student Score Prediction

A tabular regression project based on **Kaggle Playground Series S6E1**.  
The goal is to predict students' `exam_score` using demographic, study, sleep, and learning-related features.

---

## Project Overview

This project follows a structured workflow:

1. Build a reproducible baseline
2. Perform feature ablation
3. Tune LightGBM parameters with controlled experiments
4. Train the final model and generate submission files
5. Organize outputs for reproducibility and portfolio presentation

The final solution uses:

- **LightGBM Regressor**
- **5-fold KFold cross-validation**
- **RMSE** as the evaluation metric
- **One-hot encoding** for categorical features
- **Feature pruning** by removing `id` from model input

---

## Dataset

Competition: **Kaggle Playground Series S6E1 – Predicting Student Test Scores**

Input files:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

Target column:

- `exam_score`

Task type:

- **Regression**

---

## Method

### 1. Baseline
The initial baseline included:

- reading and inspecting the data
- splitting `X`, `y`, and `X_test`
- one-hot encoding categorical features
- aligning train/test columns
- training a default LightGBM model
- evaluating with 5-fold KFold cross-validation

### 2. Feature Engineering
Several feature ideas were tested through ablation experiments:

- `study_hours * class_attendance`
- `study_hours * sleep_hours`
- `age_bin`
- removing `id`

Key finding:
- engineered interaction features did **not** improve performance
- removing `id` produced a small but real gain

### 3. Parameter Tuning
Starting from the best feature version (**baseline without `id`**), LightGBM was tuned manually while keeping validation fixed.

Tuned parameters included:

- `n_estimators`
- `learning_rate`
- `num_leaves`
- `min_child_samples`

---

## Final Model

### Best feature version
- baseline without `id`

### Best parameters
- `n_estimators = 300`
- `learning_rate = 0.05`
- `num_leaves = 127`
- `min_child_samples = 50`
- `random_state = 42`

### Validation setup
- **5-fold KFold**
- `shuffle=True`
- `random_state=42`

### Evaluation metric
- **RMSE**

### Best CV Result
- **Mean RMSE: 8.773542**

---

## Reproducibility

To keep results reproducible, this project fixes:

- feature version
- train/test preprocessing logic
- KFold split strategy
- random seed
- LightGBM parameters

Saved reproducibility files:

- `best_params.json`
- `feature_ablation.md`
- `tuning_notes.md`

---

## What I Learned

Through this project, I practiced:

- building a full tabular ML pipeline
- separating baseline, feature engineering, and tuning stages
- using KFold cross-validation correctly
- comparing experiments with controlled variables
- analyzing feature importance
- making a project reproducible and portfolio-ready
