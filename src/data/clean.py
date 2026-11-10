"""Clean, label, censor; produce the versioned Parquet modelling table."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from src import config

log = logging.getLogger(__name__)


def normalise_case_type(s: str | None) -> str | None:
    if s is None:
        return None
    key = re.sub(r"[^A-Z0-9]+", "_", s.upper()).strip("_")
    for canon, variants in config.CASE_TYPE_SYNONYMS.items():
        if key in variants:
            return canon
    return key


def add_congestion(df: pl.DataFrame) -> pl.DataFrame:
    """Open cases in the same court at the moment of filing (known at filing time, so leakage-safe)."""
    courts = df["court_id"].to_numpy()
    f_all = df["filing_date"].cast(pl.Int32).cast(pl.Float64).to_numpy()
    d_all = df["decision_date"].cast(pl.Int32).cast(pl.Float64).to_numpy()  # null -> NaN
    out = np.zeros(df.height, dtype=np.int64)
    for c in np.unique(courts):
        idx = np.where(courts == c)[0]
        f = f_all[idx]
        dec = np.sort(d_all[idx][~np.isnan(d_all[idx])])
        opened = np.searchsorted(np.sort(f), f, side="right")   # filed on/before this case
        closed = np.searchsorted(dec, f, side="right")          # already decided by then
        out[idx] = np.maximum(opened - closed - 1, 0)           # minus the case itself
    return df.with_columns(pl.Series("congestion_count", out))


def clean(df: pl.DataFrame, extraction=config.EXTRACTION_DATE, pandemic_start=config.PANDEMIC_START):
    counts: dict[str, int] = {"input": df.height}

    def step(name: str, frame: pl.DataFrame) -> pl.DataFrame:
        counts[name] = frame.height
        log.info("%-34s %d rows", name, frame.height)
        return frame

    df = step("dedupe_case_id", df.unique(subset="case_id", keep="first", maintain_order=True))
    df = df.with_columns(pl.col("case_type").map_elements(normalise_case_type, return_dtype=pl.Utf8),
                         pl.col("court_id").cast(pl.Utf8).str.strip_chars(),
                         pl.col("act").cast(pl.Utf8).str.strip_chars().str.to_uppercase())
    df = step("drop_null_keys", df.drop_nulls(["case_id", "court_id", "case_type", "filing_date"]))
    df = step("drop_filed_after_extraction", df.filter(pl.col("filing_date") <= extraction))
    df = step("drop_decision_before_filing",
              df.filter(pl.col("decision_date").is_null() | (pl.col("decision_date") >= pl.col("filing_date"))))
    df = step("drop_decision_after_extraction",
              df.filter(pl.col("decision_date").is_null() | (pl.col("decision_date") <= extraction)))
    df = step("drop_pandemic_affected",
              df.filter(pl.col("decision_date").is_null() | (pl.col("decision_date") < pandemic_start)))

    hearing_cols = sorted(c for c in df.columns if re.fullmatch(r"hearing_\d+_date", c))
    end = pl.coalesce(pl.col("decision_date"), pl.lit(extraction))
    df = df.with_columns(
        *[pl.when((pl.col(h) >= pl.col("filing_date")) & (pl.col(h) <= end)).then(pl.col(h)).otherwise(None).alias(h)
          for h in hearing_cols],
        pl.col("decision_date").is_not_null().cast(pl.Int8).alias("event"),
        ((end - pl.col("filing_date")).dt.total_days() / config.DAYS_PER_MONTH).clip(lower_bound=1 / config.DAYS_PER_MONTH).alias("duration_months"),
        pl.col("filing_date").dt.year().alias("filing_year"),
        pl.col("filing_date").dt.month().alias("filing_month"),
    )
    df = add_congestion(df)
    counts["final"] = df.height
    counts["events"] = int(df["event"].sum())
    counts["censored"] = df.height - counts["events"]
    front = ["case_id", "court_id", "case_type", "act", "filing_date", "decision_date", "duration_months",
             "event", "congestion_count", "filing_year", "filing_month"]
    return df.select(front + [c for c in df.columns if c not in front]), counts


def write_versioned(df: pl.DataFrame, counts: dict, out_dir: Path = config.PROCESSED_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_tmp.parquet"
    df.write_parquet(tmp, compression="zstd")
    version = hashlib.sha256(tmp.read_bytes()).hexdigest()[:8]
    path = out_dir / f"modelling_table_v{version}.parquet"
    tmp.replace(path)
    n = duckdb.sql(f"select count(*) from read_parquet('{path}')").fetchall()[0][0]
    assert n == df.height, "row-count mismatch after writing parquet"
    manifest = {"version": version, "path": path.name, "counts": counts,
                "extraction_date": str(config.EXTRACTION_DATE), "pandemic_start": str(config.PANDEMIC_START)}
    (out_dir / "modelling_table_latest.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote %s", path)
    return path


def load_modelling_table(out_dir: Path = config.PROCESSED_DIR) -> pl.DataFrame:
    manifest = json.loads((out_dir / "modelling_table_latest.json").read_text())
    return pl.read_parquet(out_dir / manifest["path"])


def run(src: Path | None = None, out_dir: Path = config.PROCESSED_DIR) -> Path:
    src = src or config.INTERMEDIATE_DIR / "ingested.parquet"
    df, counts = clean(pl.read_parquet(src))
    return write_versioned(df, counts, out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path)
    run(ap.parse_args().src)
