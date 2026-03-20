import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor


# Read data
train = pd.read_csv('/kaggle/input/competitions/playground-series-s6e1/train.csv')
test = pd.read_csv('/kaggle/input/competitions/playground-series-s6e1/test.csv')
sample_submission = pd.read_csv('/kaggle/input/competitions/playground-series-s6e1/sample_submission.csv')


# Build features and target
# Remove id from model input
target_col = 'exam_score'

X = train.drop(columns=[target_col, 'id']).copy()
y = train[target_col]
X_test = test.drop(columns=['id']).copy()


# Preprocess features
# One-hot encode categorical variables
# Align train and test columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

X_num = X[numeric_features].copy()
X_cat = pd.get_dummies(X[categorical_features], dtype=int)

X_test_num = X_test[numeric_features].copy()
X_test_cat = pd.get_dummies(X_test[categorical_features], dtype=int)

X_processed = pd.concat([X_num, X_cat], axis=1)
X_test_processed = pd.concat([X_test_num, X_test_cat], axis=1)

X_processed, X_test_processed = X_processed.align(
    X_test_processed,
    join='left',
    axis=1,
    fill_value=0
)

X_test_processed = X_test_processed[X_processed.columns]


# 5-fold KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_rmses = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_processed), start=1):
    X_train = X_processed.iloc[train_idx]
    X_valid = X_processed.iloc[valid_idx]
    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=50,
        random_state=42
    )

    model.fit(X_train, y_train)
    pred_valid = model.predict(X_valid)

    fold_rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
    fold_rmses.append(fold_rmse)

    print(f"Fold {fold} RMSE: {fold_rmse:.6f}")

mean_rmse = np.mean(fold_rmses)
std_rmse = np.std(fold_rmses)

print("\nFold RMSEs:", fold_rmses)
print(f"Mean RMSE: {mean_rmse:.6f}")
print(f"Std RMSE: {std_rmse:.6f}")


# Train final model on full training data
final_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=127,
    min_child_samples=50,
    random_state=42
)

final_model.fit(X_processed, y)


# Predict test set and create final submission
pred_test = final_model.predict(X_test_processed)

final_submission = pd.DataFrame({
    'id': test['id'],
    'exam_score': pred_test
})

final_submission.to_csv('/kaggle/working/final_submission.csv', index=False)
print("final_submission.csv saved")


# Save best params
best_params = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 50,
    "random_state": 42
}

with open('/kaggle/working/best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)

print("best_params.json saved")


# Plot feature importance
feature_importance_df = pd.DataFrame({
    'feature': X_processed.columns,
    'importance': final_model.feature_importances_
}).sort_values(by='importance', ascending=False)

top_features = feature_importance_df.head(20).sort_values(by='importance')

plt.figure(figsize=(10, 8))
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.savefig('/kaggle/working/feature_importance.png')
plt.show()

print("feature_importance.png saved")
