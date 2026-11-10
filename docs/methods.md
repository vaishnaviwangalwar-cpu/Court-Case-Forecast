# Methods
**Table.** One row per case (`case_id, court_id, case_type, act, filing_date, decision_date, duration_months, event, congestion_count, filing_year, filing_month`, plus `hearing_k_date`). `event=1` if decided; else right-censored at the extraction date. Months = days / 30.4375.

**Features (at filing).** log open-case count at filing, congestion ratio (sanctioned strength if provided, else per-court 90th percentile of congestion fitted on training data), month sin/cos, quarter, case type, act, court (courts below 30 training cases pooled into OTHER).

**Models.** Baseline (24-month delay classifier + on-time / delayed duration regressors, gradient boosting); Cox PH (ridge, alpha 0.1); Random Survival Forest (stratified subsample cap for scale); DeepHit via pycox (optional). All expose `survival(X, times)`.

**Landmarks.** Cohort = cases still open at filing, first hearing, 6 and 12 months; target = remaining time; features add hearing count, time to first hearing, mean gap, time since last hearing (all computed only from hearings up to the landmark). Compared with the at-filing model's conditional forecast S(24)/S(origin) on the same cohort.

**Uncertainty.** Split-conformal on |y - p| for the 24-month delay label; calibration set = 20% of training data unseen by the model.

**Validation.** Random, temporal (train earlier years re-censored at the test start; test the last filing year) and spatial (25% of districts held out). Metrics: C-index, time-dependent AUC at 12/24/36/60 months, integrated Brier score, 24-month AUC/Brier on known labels, KM-based calibration deciles, conformal coverage and width, C-index by court size.

**Explainability.** Permutation importance and retraining ablation by feature group. SHAP is not used.
