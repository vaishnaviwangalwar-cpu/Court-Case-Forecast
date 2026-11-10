# Court Case Forecasting System

Forecasts **how long a court case will take** (never the outcome) using survival analysis, with calibrated
uncertainty, a FastAPI service and a Streamlit dashboard. Implements the technical plan (stack, structure,
schema, 8-week roadmap).

> **Status.** All code paths run and are tested end to end on a **synthetic** e-Courts-like dataset
> (`--synthetic`). The real Development Data Lab files were not available when this was built, so
> `src/data/ingest.py` has an *unverified* column-alias map you must adapt, and **no real-data results exist yet**.
> `docs/results.md` is generated from synthetic data and says so.

## Quick start
```bash
pip install -e ".[dev]"                  # add ,deep for DeepHit (torch+pycox), ,tracking for MLflow
python -m src.pipeline --synthetic --n 20000 --fast   # train + validate + write artifacts/bundle.joblib
python -m pytest -q
uvicorn src.api.main:app --reload        # http://localhost:8000/docs
streamlit run src/dashboard/app.py       # http://localhost:8501
docker compose -f docker/docker-compose.yml up --build   # both, after training
```
Real data: put yearly CSV/Parquet/DTA files in `data/raw/`, fix `COLUMN_ALIASES`, run `python -m src.pipeline`.
Start with one state (Maharashtra, 2012-2014) by filtering the raw files, then scale.

## What is where (maps to the roadmap)
| Week | Deliverable | Code |
|---|---|---|
| 1 | ingest, clean, censor, versioned Parquet | `src/data/ingest.py`, `clean.py`, `features.py` |
| 2 | published-baseline replication + comparison table | `src/models/baseline.py` |
| 3 | Cox PH, Random Survival Forest, C-index / TD-AUC / IBS, P(T>24) | `cox.py`, `rsf.py`, `evaluation/validate.py` |
| 4 | DeepHit, same temporal split, scale (DuckDB/Polars, stratified subsample) | `deephit.py`, `rsf.py` |
| 5 | conformal intervals, temporal + spatial + random splits | `uncertainty/conformal.py`, `validate.py` |
| 6 | landmarks, permutation importance, ablation, small-court check | `landmark.py`, `evaluation/explain.py` |
| 7 | API, dashboard (case/court/simulator), Docker, SimPy toy | `api/`, `dashboard/`, `simulation/` |
| 8 | model card, docs, tests, cleanup | `docs/`, `tests/` |

## Design decisions worth knowing
- **Censoring:** pending cases are right-censored at 25 Mar 2020; decisions on/after `PANDEMIC_START` are excluded
  (mid-March 2020 is my assumption - confirm against the PRD).
- **Leakage:** at-filing features use only filing-time information; hearing features never look past the landmark;
  the temporal split re-censors training data at the test start so no future decisions leak.
- **Honest validation:** temporal and spatial results are reported next to random-split ones; IPCW metrics are
  truncated to the training follow-up range (`eval_horizon_cap_months`).
- **Conformal interval** covers the realised 24-month yes/no outcome, so it is wide. Cases censored before 24 months
  have no label and are excluded from calibration (documented selection effect).
- **Excluded:** inferred gender/religion, judge features (not in data), outcome prediction.
- **MAPIE** is listed as an option in the plan; a custom split-conformal implementation is used instead (no extra dependency).
- **Not verified here:** DeepHit (torch/pycox absent from the build sandbox), Docker builds, MLflow (JSON fallback is used when not installed).
