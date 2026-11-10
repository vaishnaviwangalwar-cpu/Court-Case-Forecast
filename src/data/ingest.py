"""Download/extract/load e-Courts files and harmonise column names.

The real Development Data Lab files are NOT bundled and not downloadable from this environment.
Drop the yearly CSV/Parquet/DTA files into data/raw/ and adapt COLUMN_ALIASES to the real headers
(I could not verify them against the real files).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from src import config
from src.data.synthetic import generate_cases

log = logging.getLogger(__name__)

# raw column name (lower-case) -> canonical name. ADAPT to the actual schema.
COLUMN_ALIASES = {
    "ddl_case_id": "case_id", "cnr": "case_id",
    "ddl_court_id": "court_id", "court_no": "court_id",
    "type_name": "case_type", "case_type_name": "case_type",
    "act_name": "act", "act_section": "act",
    "date_filed": "filing_date", "filing": "filing_date",
    "date_of_decision": "decision_date", "decision": "decision_date",
}
REQUIRED = ["case_id", "court_id", "case_type", "act", "filing_date"]


def _read_one(path: Path) -> pl.DataFrame:
    if path.suffix == ".csv":
        return pl.read_csv(path, infer_schema_length=10000, ignore_errors=True)
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if path.suffix == ".dta":
        import pandas as pd
        return pl.from_pandas(pd.read_stata(path))
    raise ValueError(path)


def harmonise(df: pl.DataFrame) -> pl.DataFrame:
    df = df.rename({c: c.lower() for c in df.columns})
    df = df.rename({c: COLUMN_ALIASES[c] for c in df.columns if c in COLUMN_ALIASES and COLUMN_ALIASES[c] not in df.columns})
    if "decision_date" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Date).alias("decision_date"))
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after harmonising: {missing}. Update COLUMN_ALIASES.")
    for c in df.columns:
        if c.endswith("_date") and df[c].dtype != pl.Date:
            df = df.with_columns(pl.col(c).cast(pl.Utf8).str.to_date(strict=False).alias(c))
    return df


def load_raw(raw_dir: Path = config.RAW_DIR) -> pl.DataFrame:
    files = sorted(p for p in raw_dir.glob("*") if p.suffix in {".csv", ".parquet", ".dta"})
    if not files:
        raise FileNotFoundError(f"No raw files in {raw_dir}. Download e-Courts data or use --synthetic.")
    frames = []
    for p in files:
        df = harmonise(_read_one(p))
        log.info("loaded %s: %d rows", p.name, df.height)
        frames.append(df)
    out = pl.concat(frames, how="diagonal_relaxed")
    log.info("ingested total: %d rows from %d files", out.height, len(files))
    return out


# ---------------------------------------------------------------- Development Data Lab layout
# data/raw/courtcasesdata/{cases/cases/cases_YYYY.csv, keys/keys/*.csv, acts_sections/acts_sections.csv, ...}
DDL_FILING = ["date_of_filing", "date_filing", "filing_date"]
DDL_DECISION = ["date_of_decision", "decision_date"]
DDL_FIRST_LIST = ["date_first_list"]
NEEDED = ["ddl_case_id", "state_code", "dist_code", "court_no", "type_name"]
BAD_YEAR = 3000  # DDL uses sentinels like 5000-01-01 for "missing"


def _pick(cols: list[str], cands: list[str], what: str, required: bool = True) -> str | None:
    for c in cands:
        if c in cols:
            return c
    if required:
        raise ValueError(f"No {what} column found. Looked for {cands}. Columns present: {cols}")
    return None


def _date(col: str) -> pl.Expr:
    d = pl.col(col).cast(pl.Utf8).str.strip_chars().str.to_date("%Y-%m-%d", strict=False)
    return pl.when(d.dt.year() < BAD_YEAR).then(d).otherwise(None)


def ddl_case_files(raw_dir: Path) -> list[Path]:
    import re
    return sorted(p for p in raw_dir.rglob("cases_*.csv") if re.fullmatch(r"cases_\d{4}\.csv", p.name))


def _find_key(raw_dir: Path, name: str) -> Path | None:
    hits = list(raw_dir.rglob(name))
    return hits[0] if hits else None


def load_ddl(raw_dir: Path = config.RAW_DIR, state: str | None = "Maharashtra",
             years: list[int] | None = None) -> pl.DataFrame:
    files = ddl_case_files(raw_dir)
    if years:
        files = [f for f in files if int(f.stem.split("_")[1]) in set(years)]
    if not files:
        raise FileNotFoundError(f"No cases_YYYY.csv files for years={years} under {raw_dir}")

    codes = None
    if state:
        ck = _find_key(raw_dir, "cases_court_key.csv") or _find_key(raw_dir, "cases_state_key.csv")
        key = pl.read_csv(ck, infer_schema_length=0)
        codes = key.filter(pl.col("state_name").str.to_lowercase() == state.lower())["state_code"].unique().to_list()
        if not codes:
            raise ValueError(f"State {state!r} not in {ck.name}")
        log.info("state %s -> state_code %s", state, codes)

    frames = []
    for f in files:
        lf = pl.scan_csv(f, infer_schema_length=0, ignore_errors=True)
        cols = lf.collect_schema().names()
        missing = [c for c in NEEDED if c not in cols]
        if missing:
            raise ValueError(f"{f.name}: missing {missing}. Columns present: {cols}")
        fil = _pick(cols, DDL_FILING, "filing date")
        dec = _pick(cols, DDL_DECISION, "decision date")
        fst = _pick(cols, DDL_FIRST_LIST, "first listing date", required=False)
        if codes:
            lf = lf.filter(pl.col("state_code").is_in(codes))
        lf = lf.select(
            pl.col("ddl_case_id").alias("case_id"),
            pl.concat_str([pl.col("state_code"), pl.col("dist_code"), pl.col("court_no")], separator="_").alias("court_id"),
            pl.col("year").alias("_year"), pl.col("type_name").alias("_type_code"),
            _date(fil).alias("filing_date"), _date(dec).alias("decision_date"),
            (_date(fst) if fst else pl.lit(None, dtype=pl.Date)).alias("hearing_1_date"))
        df = lf.collect()
        log.info("%s: %d rows", f.name, df.height)
        frames.append(df)
    df = pl.concat(frames)

    # case type: code -> label via type_name_key
    tk = _find_key(raw_dir, "type_name_key.csv")
    if tk is not None:
        key = pl.read_csv(tk, infer_schema_length=0)
        label = "type_name_s" if "type_name_s" in key.columns else next(c for c in key.columns if c not in {"year", "type_name", "count"})
        on_left, on_right = (["_year", "_type_code"], ["year", "type_name"]) if "year" in key.columns else (["_type_code"], ["type_name"])
        key = key.select(*on_right, pl.col(label).alias("_label")).unique(subset=on_right)
        df = df.join(key, left_on=on_left, right_on=on_right, how="left")
        df = df.with_columns(pl.coalesce(pl.col("_label"), pl.col("_type_code")).alias("case_type")).drop("_label")
    else:
        log.warning("type_name_key.csv not found -> using raw type codes")
        df = df.with_columns(pl.col("_type_code").alias("case_type"))

    # act: first act listed per case (acts_sections has one row per act/section)
    ap = _find_key(raw_dir, "acts_sections.csv")
    if ap is not None:
        acts = (pl.scan_csv(ap, infer_schema_length=0, ignore_errors=True).select("ddl_case_id", "act")
                .join(df.select(pl.col("case_id").alias("ddl_case_id")).lazy(), on="ddl_case_id", how="semi")
                .group_by("ddl_case_id").agg(pl.col("act").drop_nulls().first()).collect())
        df = df.join(acts.rename({"ddl_case_id": "case_id"}), on="case_id", how="left")
        df = df.with_columns(("ACT_" + pl.col("act")).fill_null("UNKNOWN").alias("act"))
    else:
        df = df.with_columns(pl.lit("UNKNOWN").alias("act"))
    log.info("DDL ingest: %d rows (state=%s, years=%s)", df.height, state, years)
    return df.select("case_id", "court_id", "case_type", "act", "filing_date", "decision_date", "hearing_1_date")


def ingest(synthetic: bool = False, n: int = 20000, raw_dir: Path = config.RAW_DIR,
           out_dir: Path = config.INTERMEDIATE_DIR, state: str | None = "Maharashtra",
           years: list[int] | None = None) -> Path:
    if synthetic:
        df = generate_cases(n=n)
    elif ddl_case_files(raw_dir):
        df = load_ddl(raw_dir, state, years)
    else:
        df = load_raw(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ingested.parquet"
    df.write_parquet(path)
    log.info("wrote %s (%d rows)", path, df.height)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--state", default="Maharashtra")
    ap.add_argument("--years", type=int, nargs="*")
    a = ap.parse_args()
    ingest(a.synthetic, a.n, state=a.state, years=a.years)
