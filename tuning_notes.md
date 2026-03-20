# Tuning Notes

## Goal
Tune LightGBM parameters on top of the current best feature version.

## Fixed setup
- Feature version: baseline without `id`
- Validation: 5-fold KFold
- Metric: RMSE
- Random state: 42

## Baseline before tuning
- Feature version: baseline without `id`
- Mean RMSE: 8.814694

## Tuning experiments

### T1
**Params**
- `n_estimators = 300`
- `learning_rate = 0.05`
- `num_leaves = 31`
- `min_child_samples = 50`
- `random_state = 42`

**Mean RMSE**
- `8.793979`

**Result**
- Better than baseline

---

### T2
**Params**
- `n_estimators = 300`
- `learning_rate = 0.05`
- `num_leaves = 63`
- `min_child_samples = 50`
- `random_state = 42`

**Mean RMSE**
- `8.780514`

**Result**
- Better than T1

---

### T3
**Params**
- `n_estimators = 300`
- `learning_rate = 0.05`
- `num_leaves = 63`
- `min_child_samples = 20`
- `random_state = 42`

**Mean RMSE**
- `8.781691`

**Result**
- Slightly worse than T2

---

### T4
**Params**
- `n_estimators = 300`
- `learning_rate = 0.05`
- `num_leaves = 127`
- `min_child_samples = 50`
- `random_state = 42`

**Mean RMSE**
- `8.773542`

**Result**
- Current best

## Best params so far
- `n_estimators = 300`
- `learning_rate = 0.05`
- `num_leaves = 127`
- `min_child_samples = 50`
- `random_state = 42`

## Best RMSE
- `8.773542`

## Observations
1. Removing `id` improved the baseline.
2. Increasing `num_leaves` from 31 to 63 improved RMSE.
3. Increasing `num_leaves` from 63 to 127 improved RMSE again.
4. Lowering `min_child_samples` from 50 to 20 did not help.
5. The current model benefits from stronger tree capacity, but still prefers a relatively conservative leaf-size constraint.

## Conclusion
The current best version is the LightGBM model **without `id`**, using:
- `n_estimators = 300`
- `learning_rate = 0.05`
- `num_leaves = 127`
- `min_child_samples = 50`
- `random_state = 42`

Final best **Mean RMSE = 8.773542**.
