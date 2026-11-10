import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.dashboard.components import disclaimer, get_bundle_or_stop  # noqa: E402
from src.data.clean import load_modelling_table  # noqa: E402

st.title("Court view")
disclaimer()
b = get_bundle_or_stop()
try:
    _t = load_modelling_table(config.PROCESSED_DIR)
    df = _t.sample(n=min(_t.height, 300_000), seed=config.SEED).to_pandas()
except FileNotFoundError:
    st.error("Modelling table not found in data/processed/.")
    st.stop()


@st.cache_data
def summary(version: str) -> "pd.DataFrame":
    return b.court_summary(df)


s = summary(b.version)
st.write("Share of sampled cases per court that the model forecasts as long-delay (P(delay>24m) > 0.5).")
st.bar_chart(s.set_index("court_id")["share_long_delay"])
st.dataframe(s, use_container_width=True)
