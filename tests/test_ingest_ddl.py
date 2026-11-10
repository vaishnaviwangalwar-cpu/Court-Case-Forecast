"""Builds a tiny fake Development Data Lab folder (same layout/columns as the real download) and ingests it."""
import polars as pl

from src.data import clean
from src.data.ingest import ingest


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(path)


def test_ddl_layout(tmp_path):
    raw = tmp_path / "raw" / "courtcasesdata"
    n = 400
    for year in (2012, 2013):
        rows = {"ddl_case_id": [f"01-01-01-{year}{i:05d}" for i in range(n)], "year": [str(year)] * n,
                "state_code": ["01"] * (n - 50) + ["02"] * 50, "dist_code": ["01"] * n,
                "court_no": ["01", "02"] * (n // 2), "type_name": ["5", "6"] * (n // 2),
                "date_of_filing": [f"{year}-03-01"] * n,
                "date_of_decision": [f"{year}-09-01" if i % 3 else "5000-01-01" for i in range(n)],
                "date_first_list": [f"{year}-04-01"] * n, "date_last_list": ["5000-01-01"] * n, "date_next_list": [""] * n}
        _write(raw / "cases" / "cases" / f"cases_{year}.csv", rows)
    _write(raw / "keys" / "keys" / "cases_court_key.csv", {"year": ["2012", "2012"], "state_code": ["01", "02"],
           "state_name": ["Maharashtra", "Bihar"], "district_name": ["X", "Y"], "dist_code": ["01", "01"],
           "court_no": ["01", "01"], "court_name": ["a", "b"]})
    _write(raw / "keys" / "keys" / "type_name_key.csv", {"year": ["2012"] * 2 + ["2013"] * 2, "type_name": ["5", "6"] * 2,
           "type_name_s": ["crl.m.a.", "s.c.c."] * 2, "count": ["1"] * 4})
    _write(raw / "acts_sections" / "acts_sections.csv", {"ddl_case_id": ["01-01-01-201200000", "01-01-01-201200000"],
           "act": ["17353", "99"], "section": [None, None]})
    path = ingest(synthetic=False, raw_dir=tmp_path / "raw", out_dir=tmp_path / "inter", state="Maharashtra", years=[2012, 2013])
    df = pl.read_parquet(path)
    assert df.height == 2 * 350 and set(df["case_type"]) == {"crl.m.a.", "s.c.c."}
    assert df["court_id"].str.starts_with("01_01_").all()
    assert df["decision_date"].null_count() > 0 and df["hearing_1_date"].null_count() == 0   # sentinel -> null
    assert df.filter(pl.col("case_id") == "01-01-01-201200000")["act"][0] == "ACT_17353"
    out, counts = clean.clean(df)
    assert counts["final"] > 0 and "hearing_1_date" in out.columns
