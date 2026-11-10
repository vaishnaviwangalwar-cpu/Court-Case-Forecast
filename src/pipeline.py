"""End-to-end run: ingest -> clean -> baseline + survival models -> validation -> uncertainty ->
landmarks -> explainability -> deployable bundle + results report.

    python -m src.pipeline --synthetic --n 20000          # demo on synthetic data
    python -m src.pipeline                                # real data in data/raw/
"""
from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))  # silences joblib/wmic warning on Windows 11

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src import config
from src.bundle import ModelBundle
from src.data import clean, ingest
from src.data.features import FeatureEncoder, delay_label
from src.evaluation import explain, validate
from src.models.baseline import Baseline, compare_to_published
from src.models.base import SurvivalModel
from src.models.cox import CoxModel
from src.models.deephit import DeepHitModel, deep_available
from src.models.landmark import LANDMARKS, REMAIN_GRID, LandmarkModel, landmark_cohort
from src.models.rsf import RSFModel
from src.tracking import log_run
from src.uncertainty.conformal import ConformalDelayInterval

log = logging.getLogger("pipeline")


def _factories(fast: bool, with_deep: bool) -> dict:
    f = {"cox": lambda: CoxModel(),
         "rsf": (lambda: RSFModel(n_estimators=40, min_samples_leaf=40, max_train=8000)) if fast else (lambda: RSFModel())}
    if with_deep and deep_available():
        f["deephit"] = lambda: DeepHitModel(epochs=20 if fast else 60)
    elif with_deep:
        log.warning("torch/pycox not installed -> DeepHit skipped (pip install '.[deep]')")
    return f


def _prep(df: pl.DataFrame, enc: FeatureEncoder | None = None):
    pdf = df.to_pandas()
    enc = enc or FeatureEncoder().fit(pdf)
    return pdf, enc, enc.transform(pdf)


def _clean_nan(o):
    if isinstance(o, dict):
        return {str(k): _clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_nan(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def run_scheme(name: str, train: pl.DataFrame, test: pl.DataFrame, factories: dict, seed: int = config.SEED) -> dict:
    """Fit on 80% of `train`, calibrate conformal on the other 20%, evaluate on `test`."""
    fit_df, cal_df = validate.random_split(train, 0.2, seed)
    fit_pd, enc, Xfit = _prep(fit_df)
    cal_pd, _, Xcal = _prep(cal_df, enc)
    te_pd, _, Xte = _prep(test, enc)
    res: dict = {"n_train": fit_df.height, "n_calibration": cal_df.height, "n_test": test.height, "models": {}}

    base = Baseline().fit(Xfit, fit_pd["duration_months"], fit_pd["event"])
    res["baseline"] = base.evaluate(Xte, te_pd["duration_months"], te_pd["event"])
    fitted: dict[str, SurvivalModel] = {}
    for mname, make in factories.items():
        m = make().fit(Xfit, fit_pd["duration_months"], fit_pd["event"])
        fitted[mname] = m
        r = validate.evaluate_survival(m, Xfit, fit_pd["duration_months"], fit_pd["event"], Xte,
                                       te_pd["duration_months"], te_pd["event"])
        y_cal = delay_label(cal_pd["duration_months"].to_numpy(), cal_pd["event"].to_numpy())
        y_te = delay_label(te_pd["duration_months"].to_numpy(), te_pd["event"].to_numpy())
        kc, kt = y_cal >= 0, y_te >= 0
        conf = ConformalDelayInterval().calibrate(m.p_delay(Xcal)[kc], y_cal[kc])
        r["conformal"] = conf.report(m.p_delay(Xte)[kt], y_te[kt])
        r["calibration"] = validate.calibration_table(m.p_delay(Xte), te_pd["duration_months"], te_pd["event"]).round(4).to_dict("records")
        res["models"][mname] = r
        log_run(f"{name}/{mname}", {"scheme": name, "model": mname}, {**{k: v for k, v in r.items() if isinstance(v, float)},
                                                                       "coverage": r["conformal"]["empirical_coverage"]})
    res["_fitted"], res["_enc"] = fitted, enc
    return res


def _conditional_p_atfiling(model, X, origin, horizon=24.0):
    S = model.survival(X, REMAIN_GRID)
    s_h = S[:, int(horizon)]
    s_o = S[np.arange(len(X)), np.clip(np.floor(origin).astype(int), 0, len(REMAIN_GRID) - 1)]
    p = np.clip(s_h / np.maximum(s_o, 1e-6), 0, 1)
    return np.where(origin >= horizon, 1.0, p)


def run_landmarks(train: pl.DataFrame, test: pl.DataFrame, filing_model: SurvivalModel, enc: FeatureEncoder) -> list[dict]:
    lm_model = LandmarkModel(lambda: CoxModel()).fit(train)
    rows = []
    for lm in LANDMARKS:
        if lm.name not in lm_model.parts_:
            continue
        pdf, p_lm = lm_model.p_delay(test, lm)
        p_base = _conditional_p_atfiling(filing_model, enc.transform(pdf), pdf["_origin"].to_numpy())
        for tag, p in (("at-filing model", p_base), ("landmark model", p_lm)):
            m = validate.binary_delay_metrics(p, pdf["duration_months"], pdf["event"])
            rows.append({"landmark": lm.name, "model": tag, "n_cohort": len(pdf), **m})
    return rows


def md_table(df: pd.DataFrame) -> str:
    df = df.copy()
    df = df.map(lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)) if hasattr(df, "map") else df.applymap(str)
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(r) + " |" for r in df.values.tolist()) + "\n"


def run(n: int = 20000, synthetic: bool = False, fast: bool = False, with_deep: bool = True,
        out_dir: Path = config.ARTIFACT_DIR, data_dir: Path | None = None, docs_dir: Path | None = config.DOCS_DIR,
        state: str | None = "Maharashtra", years: list[int] | None = None, max_rows: int = 200_000,
        reuse_table: bool = False) -> dict:
    inter = (data_dir or config.DATA_DIR) / "intermediate"
    proc = (data_dir or config.DATA_DIR) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not reuse_table or not (proc / "modelling_table_latest.json").exists():   # always rebuild unless told otherwise
        src = ingest.ingest(synthetic=synthetic, n=n, out_dir=inter, state=state, years=years)
        clean.run(src, proc)
    manifest = json.loads((proc / "modelling_table_latest.json").read_text())
    df = clean.load_modelling_table(proc)
    log.info("modelling table v%s: %d rows, event rate %.3f", manifest["version"], df.height, df["event"].mean())
    if df.height > max_rows:  # keeps one-hot matrices in memory; congestion was already computed on the full table
        log.info("subsampling %d -> %d rows for modelling (--max-rows)", df.height, max_rows)
        df = df.sample(n=max_rows, seed=config.SEED)
    factories = _factories(fast, with_deep)

    max_year = int(df["filing_year"].max())
    schemes = {"random": validate.random_split(df, 0.25)}
    schemes["temporal"] = validate.temporal_split(df, max_year)
    sp_train, sp_test, held = validate.spatial_split(df)
    schemes["spatial"] = (sp_train, sp_test)

    results: dict = {"table_version": manifest["version"], "synthetic": bool(synthetic), "cleaning_counts": manifest["counts"],
                     "spatial_holdout_districts": held, "schemes": {}}
    runs = {}
    for name, (tr, te) in schemes.items():
        log.info("scheme %-8s train=%d test=%d", name, tr.height, te.height)
        runs[name] = run_scheme(name, tr, te, factories)
        results["schemes"][name] = {k: v for k, v in runs[name].items() if not k.startswith("_")}

    # baseline replication table (random split mirrors the published setup most closely)
    results["baseline_comparison"] = compare_to_published(results["schemes"]["random"]["baseline"]).to_dict("records")

    # landmarks + explainability on the temporal scheme (the honest one)
    t = runs["temporal"]
    train_t, test_t = schemes["temporal"]
    primary = "cox"
    results["landmarks"] = run_landmarks(train_t, test_t, t["_fitted"][primary], t["_enc"])
    te_pd, _, Xte = _prep(test_t, t["_enc"])
    fit_pd, _, Xfit = _prep(validate.random_split(train_t, 0.2)[0], t["_enc"])
    results["permutation_importance"] = explain.permutation_importance(t["_fitted"][primary], Xte, te_pd["duration_months"], te_pd["event"]).to_dict("records")
    results["ablation"] = explain.group_ablation(factories[primary], Xfit, fit_pd["duration_months"], fit_pd["event"],
                                                 Xte, te_pd["duration_months"], te_pd["event"]).to_dict("records")
    results["court_size_performance"] = validate.performance_by_court_size(t["_fitted"][primary], Xte, te_pd).to_dict("records")

    # deployable bundle: best model by temporal C-index, trained on all data minus a calibration split
    best = max(t["models"], key=lambda k: t["models"][k]["c_index"])
    fit_df, cal_df = validate.random_split(df, 0.2)
    fit_pd, enc, Xfit = _prep(fit_df)
    cal_pd, _, Xcal = _prep(cal_df, enc)
    model = factories[best]().fit(Xfit, fit_pd["duration_months"], fit_pd["event"])
    y_cal = delay_label(cal_pd["duration_months"].to_numpy(), cal_pd["event"].to_numpy())
    kc = y_cal >= 0
    conf = ConformalDelayInterval().calibrate(model.p_delay(Xcal)[kc], y_cal[kc])
    meta = {"data_period": f"{int(df['filing_year'].min())}-{max_year}", "extraction_date": str(config.EXTRACTION_DATE),
            "n_train": fit_df.height, "synthetic_data": bool(synthetic), "model": best,
            "courts": list(enc.courts_), "case_types": list(enc.case_types_), "acts": list(enc.acts_),
            "state": state if not synthetic else None, "rows_modelled": df.height}
    bundle = ModelBundle(version=f"{manifest['version']}-{best}", model=model, encoder=enc, conformal=conf, meta=meta)
    bundle.save(out_dir)
    results["deployed"] = {"version": bundle.version, "conformal_q": conf.q_, **meta}
    (out_dir / "metrics.json").write_text(json.dumps(_clean_nan(results), indent=2, default=str))
    if docs_dir:
        write_results_md(results, docs_dir / "results.md")
    log.info("deployed model %s -> %s", bundle.version, out_dir)
    return results


def write_results_md(r: dict, path: Path) -> None:
    L = ["# Results (auto-generated)\n"]
    if r["synthetic"]:
        L.append("> **These numbers come from SYNTHETIC data. They validate the pipeline, not the science. "
                 "Re-run on real e-Courts data before drawing any conclusion.**\n")
    L.append(f"Modelling table `v{r['table_version']}`; cleaning counts: `{r['cleaning_counts']}`\n")
    rows = []
    for s, sv in r["schemes"].items():
        for m, mv in sv["models"].items():
            rows.append({"split": s, "model": m, "C-index": mv["c_index"], "IBS": mv["ibs"], "AUC@24m": mv.get("auc_24m"),
                         "Brier@24m": mv.get("brier_24m"), "TD-AUC@12": mv["td_auc"].get(12), "TD-AUC@36": mv["td_auc"].get(36),
                         "TD-AUC@60": mv["td_auc"].get(60), "coverage": mv["conformal"]["empirical_coverage"],
                         "width": mv["conformal"]["mean_interval_width"]})
    L.append("## Survival models by validation scheme\n" + md_table(pd.DataFrame(rows)))
    L.append("## Baseline replication vs published\n" + md_table(pd.DataFrame(r["baseline_comparison"])))
    L.append("## Landmark models (temporal split)\n" + md_table(pd.DataFrame(r["landmarks"])))
    L.append("## Permutation importance (feature groups)\n" + md_table(pd.DataFrame(r["permutation_importance"])))
    L.append("## Ablation (judge features unavailable -> documented exclusion)\n" + md_table(pd.DataFrame(r["ablation"])))
    L.append("## C-index by court size\n" + md_table(pd.DataFrame(r["court_size_performance"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))


def main_cli() -> None:  # entry point used by pyproject [project.scripts]
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-deep", action="store_true")
    ap.add_argument("--state", default="Maharashtra")
    ap.add_argument("--years", type=int, nargs="*", help="e.g. --years 2012 2013 2014")
    ap.add_argument("--max-rows", type=int, default=200_000)
    ap.add_argument("--reuse-table", action="store_true")
    a = ap.parse_args()
    run(a.n, a.synthetic, a.fast, not a.no_deep, state=a.state, years=a.years, max_rows=a.max_rows, reuse_table=a.reuse_table)


if __name__ == "__main__":
    main_cli()
