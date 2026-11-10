import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from src.dashboard.components import disclaimer, get_bundle_or_stop  # noqa: E402

st.set_page_config(page_title="Court Case Forecasting", layout="wide")
st.title("Court Case Forecasting System")
disclaimer()
b = get_bundle_or_stop()
st.write("Forecasts how long a case is likely to stay pending, with calibrated uncertainty. "
         "Use the pages in the sidebar: **case view**, **court view**, **simulator**.")
c1, c2, c3 = st.columns(3)
c1.metric("Model version", b.version)
c2.metric("Data period", b.meta["data_period"])
c3.metric("Training cases", f"{b.meta['n_train']:,}")
