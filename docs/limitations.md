# Limitations
1. No real-data results yet; synthetic data only validates the code.
2. Column aliases in `ingest.py` are unverified against the real files; hearing history must be joined into `hearing_k_date` columns.
3. Pandemic-affected decisions are excluded, which can bias long-duration cases (documented scope limit). The cut-off date is an assumption.
4. Temporal splits cap usable follow-up at the training window; 36-60 month metrics may be NaN.
5. Conformal intervals are marginal (not per-case or per-court) and exclude cases censored before 24 months.
6. Congestion ratio falls back to a data-driven capacity proxy when sanctioned strength is absent.
7. Judge-level features and outcomes are deliberately out of scope; the ablation row for them is empty.
8. The SimPy simulator is an uncalibrated toy.
9. DeepHit, Docker images and MLflow were not exercised in the build environment.
10. Duration forecasts can reflect historic backlogs and inequities; do not use them to allocate attention to individuals.
