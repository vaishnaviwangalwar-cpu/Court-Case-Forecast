"""Global configuration: paths, seeds, constants and published-baseline numbers."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

ROOT = Path(os.getenv("CCF_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.getenv("CCF_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
ARTIFACT_DIR = Path(os.getenv("CCF_ARTIFACT_DIR", ROOT / "artifacts"))
MLRUNS_DIR = ROOT / "mlruns"
DOCS_DIR = ROOT / "docs"

SEED = 42
DAYS_PER_MONTH = 30.4375

# Censoring / scope rules (PRD): pending cases are right-censored at the extraction date.
EXTRACTION_DATE = date(2020, 3, 25)
# Decisions on/after this date are treated as pandemic-affected and excluded (documented limitation).
# ASSUMPTION: courts began closing around mid-March 2020; adjust if the PRD says otherwise.
PANDEMIC_START = date(2020, 3, 16)

# Prediction scope
HORIZON_MONTHS = 24
EVAL_TIMES = (12, 24, 36, 60)
CONFORMAL_ALPHA = 0.10
MAX_HEARINGS = 5

# Published baseline to replicate (from the PRD)
PUBLISHED_BASELINE = {
    "auc_24m": 0.89,
    "ontime_rmse": 4.90,
    "ontime_mae": 3.33,
    "delayed_rmse": 13.79,
    "delayed_mae": 10.74,
}

# Case-type normalisation across courts (extend as you meet real labels)
CASE_TYPE_SYNONYMS = {
    "CHEQUE_BOUNCE": {"CHEQUE_BOUNCE", "CHEQUE_DISHONOUR", "NI_ACT_138", "NI_ACT", "SEC_138_NI_ACT"},
    "THEFT": {"THEFT", "THEFT_CASE"},
    "HURT": {"HURT", "SIMPLE_HURT"},
    "NDPS": {"NDPS", "NDPS_ACT"},
    "DOMESTIC_VIOLENCE": {"DOMESTIC_VIOLENCE", "DV_ACT"},
    "TRAFFIC_OFFENCE": {"TRAFFIC_OFFENCE", "MV_ACT", "TRAFFIC"},
    "FRAUD": {"FRAUD", "CHEATING"},
    "LAND_DISPUTE": {"LAND_DISPUTE", "LAND"},
}

# Columns that must never be used as features (noisy / out of scope / outcome-related)
EXCLUDED_FEATURES = ("gender", "religion", "judge_id", "disposition", "outcome")
