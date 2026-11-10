from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_bundle
from src.bundle import DISCLAIMER, ModelBundle
from src.simulation.court_sim import SimParams, simulate

router = APIRouter()


class CaseIn(BaseModel):
    court_id: str
    case_type: str
    act: str
    filing_date: dt.date
    congestion_count: int | None = Field(None, ge=0, description="Open cases in the court at filing (default: court typical)")
    sanctioned_strength: float | None = Field(None, gt=0)


class SimIn(BaseModel):
    arrivals_per_month: float = Field(60, gt=0)
    slots: int = Field(20, ge=1)
    extra_capacity_pct: float = Field(0, ge=0, le=200)
    fast_track_share: float = Field(0, ge=0, le=1)


def _one(case: CaseIn, b: ModelBundle) -> dict:
    if case.court_id not in b.meta["courts"]:
        # still answered (encoder maps to OTHER) but flagged: the forecast leans on pooled behaviour
        pass
    return b.predict([case.model_dump()])[0]


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/meta")
def meta(b: ModelBundle = Depends(get_bundle)):
    return {"model_version": b.version, **{k: b.meta[k] for k in ("courts", "case_types", "acts", "data_period", "synthetic_data")},
            "disclaimer": DISCLAIMER}


@router.post("/predict")
def predict(case: CaseIn, b: ModelBundle = Depends(get_bundle)):
    """Survival curve, P(T>24m), conformal interval and model version for one case."""
    out = _one(case, b)
    out["known_court"] = case.court_id in b.meta["courts"]
    return out


@router.post("/curve")
def curve(case: CaseIn, b: ModelBundle = Depends(get_bundle)):
    o = _one(case, b)
    return {"model_version": o["model_version"], "times_months": o["times_months"], "survival": o["survival"]}


@router.post("/interval")
def interval(case: CaseIn, b: ModelBundle = Depends(get_bundle)):
    o = _one(case, b)
    return {"model_version": o["model_version"], "p_delay_24m": o["p_delay_24m"], "interval": o["interval"]}


@router.post("/simulate")
def sim(body: SimIn, b: ModelBundle = Depends(get_bundle)):
    """Stretch goal: toy discrete-event scenario comparison (baseline vs capacity vs fast-track)."""
    base = SimParams(arrivals_per_month=body.arrivals_per_month, slots=body.slots)
    scen = SimParams(arrivals_per_month=body.arrivals_per_month, slots=body.slots,
                     extra_capacity_pct=body.extra_capacity_pct, fast_track_share=body.fast_track_share)
    return {"model_version": b.version, "baseline": simulate(base), "scenario": simulate(scen), "disclaimer": DISCLAIMER}
