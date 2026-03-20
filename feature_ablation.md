# Feature Ablation

## V0 Baseline
- CV RMSE: `8.815348`
- Conclusion: current baseline

## V1 + `study_hours * class_attendance`
- CV RMSE: `8.821054`
- Conclusion: worse than baseline, do not keep

## V2 + `study_hours * class_attendance` + `study_hours * sleep_hours`
- CV RMSE: `8.826020`
- Conclusion: worse than baseline, do not keep

## V3 + `age_bin`
- CV RMSE: `8.815348`
- Conclusion: same as baseline, no clear gain

## V4 baseline without `id`
- CV RMSE: `8.814694`
- Change vs baseline: better
- Conclusion: remove `id` from model input
