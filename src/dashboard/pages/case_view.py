import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import datetime as dt  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.dashboard.components import disclaimer, get_bundle_or_stop  # noqa: E402

st.title("Case view")
disclaimer()
b = get_bundle_or_stop()
c1, c2 = st.columns(2)
court = c1.selectbox("Court", b.meta["courts"])
ctype = c1.selectbox("Case type", b.meta["case_types"])
act = c2.selectbox("Act", b.meta["acts"])
fdate = c2.date_input("Filing date", dt.date(2014, 6, 1))
out = b.predict([{"court_id": court, "case_type": ctype, "act": act, "filing_date": fdate}])[0]
m1, m2 = st.columns(2)
m1.metric("P(delay > 24 months)", f"{out['p_delay_24m']:.1%}")
iv = out["interval"]
m2.metric(f"{int((1 - iv['alpha']) * 100)}% conformal interval", f"{iv['lower']:.0%} - {iv['upper']:.0%}")
st.caption("The interval has guaranteed coverage of the realised yes/no outcome, so it is wide by design.")
st.line_chart(pd.DataFrame({"months since filing": out["times_months"], "P(case still pending)": out["survival"]}).set_index("months since filing"))
st.caption(f"Model version: {out['model_version']} | filing month {fdate.month}")
