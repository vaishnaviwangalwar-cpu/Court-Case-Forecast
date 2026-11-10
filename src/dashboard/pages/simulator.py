import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.dashboard.components import disclaimer, get_bundle_or_stop  # noqa: E402
from src.simulation.court_sim import SimParams, simulate  # noqa: E402

st.title("Policy simulator (stretch goal, illustrative toy)")
disclaimer()
get_bundle_or_stop()
st.caption("A toy queueing model, not calibrated to any real court. Use it to explore direction, not magnitude.")
arr = st.slider("Case arrivals per month", 10, 200, 60)
slots = st.slider("Baseline hearing slots", 5, 60, 20)
cap = st.slider("Additional capacity (%)", 0, 100, 0, step=5)
ft = st.slider("Fast-tracked share of cases", 0.0, 1.0, 0.0, step=0.05)
base = simulate(SimParams(arrivals_per_month=arr, slots=slots))
scen = simulate(SimParams(arrivals_per_month=arr, slots=slots, extra_capacity_pct=cap, fast_track_share=ft))
c1, c2 = st.columns(2)
c1.metric("Baseline: share > 24 months", f"{base['share_over_24m']:.1%}")
c2.metric("Scenario: share > 24 months", f"{scen['share_over_24m']:.1%}", f"{scen['share_over_24m'] - base['share_over_24m']:+.1%}")
st.line_chart(pd.DataFrame({"months": base["times_months"], "baseline": base["survival"], "scenario": scen["survival"]}).set_index("months"))
st.caption(base["utilisation_note"] + " (baseline)")
