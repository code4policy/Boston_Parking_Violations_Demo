from pathlib import Path
from datetime import date
import duckdb
import pandas as pd
import streamlit as st
import pydeck as pdk

APP_DIR = Path(__file__).resolve().parent

REQUIRED_PARQUETS = [
    "agg_map_grid_day_norm.parquet",
    "agg_offenders_day_norm.parquet",
    "agg_streets_day_norm.parquet",
    "agg_offender_violation_day.parquet",
    "agg_offender_hour_day.parquet",
    "tickets_2024_enriched.parquet",
]


def check_required_files(base_dir: Path = APP_DIR) -> None:
    missing = [name for name in REQUIRED_PARQUETS if not (base_dir / name).exists()]
    if not missing:
        return

    st.error(
        "Missing required data files. This app expects pre-built Parquet artifacts in the repo (or mounted storage).\n\n"
        "Missing:\n- "
        + "\n- ".join(missing)
        + "\n\nHow to fix:\n"
        "1) Run PARQUET.ipynb up through the parquet-write cells to generate these files locally, then deploy them with the app; or\n"
        "2) Store the Parquet files in a remote bucket and modify the app to download them at startup.\n"
    )
    st.stop()

@st.cache_resource
def get_con():
    check_required_files(APP_DIR)
    con = duckdb.connect()

    con.execute(f"CREATE OR REPLACE VIEW agg_map_grid_day_norm AS SELECT * FROM '{(APP_DIR / 'agg_map_grid_day_norm.parquet').as_posix()}'")
    con.execute(f"CREATE OR REPLACE VIEW agg_offenders_day_norm AS SELECT * FROM '{(APP_DIR / 'agg_offenders_day_norm.parquet').as_posix()}'")
    con.execute(f"CREATE OR REPLACE VIEW agg_streets_day_norm AS SELECT * FROM '{(APP_DIR / 'agg_streets_day_norm.parquet').as_posix()}'")
    con.execute(f"CREATE OR REPLACE VIEW agg_offender_violation_day AS SELECT * FROM '{(APP_DIR / 'agg_offender_violation_day.parquet').as_posix()}'")
    con.execute(f"CREATE OR REPLACE VIEW agg_offender_hour_day AS SELECT * FROM '{(APP_DIR / 'agg_offender_hour_day.parquet').as_posix()}'")
    con.execute(f"CREATE OR REPLACE VIEW tickets_enriched AS SELECT * FROM '{(APP_DIR / 'tickets_2024_enriched.parquet').as_posix()}'")

    return con

@st.cache_data
def get_violation_values():
    con = get_con()
    df = con.execute(
        """
        SELECT DISTINCT violation_desc_long
        FROM agg_map_grid_day_norm
        ORDER BY 1
        """
    ).fetchdf()
    return df["violation_desc_long"].tolist()

def q_map(con, date_start, date_end, violations):
    if violations:
        placeholders = ",".join(["?"] * len(violations))
        sql = f"""
        SELECT
          lat_bin,
          lon_bin,
          SUM(ticket_count) AS total_count,
          SUM(ticket_count) / SUM(day_weight) AS avg_per_day
        FROM agg_map_grid_day_norm
        WHERE ticket_date BETWEEN ? AND ?
          AND violation_desc_long IN ({placeholders})
        GROUP BY 1,2
        """
        params = [date_start, date_end] + list(violations)
    else:
        sql = """
        SELECT
          lat_bin,
          lon_bin,
          SUM(ticket_count) AS total_count,
          SUM(ticket_count) / SUM(day_weight) AS avg_per_day
        FROM agg_map_grid_day_norm
        WHERE ticket_date BETWEEN ? AND ?
        GROUP BY 1,2
        """
        params = [date_start, date_end]
    return con.execute(sql, params).fetchdf()

def q_top_offenders(con, date_start, date_end, violations, top_n, metric="total"):
    top_n = int(top_n)
    score_expr = "SUM(ticket_count)" if metric == "total" else "SUM(ticket_count) / SUM(day_weight)"

    if violations:
        placeholders = ",".join(["?"] * len(violations))
        sql = f"""
        SELECT
          deid_lpn,
          {score_expr} AS score,
          SUM(ticket_count) AS total_tickets,
          SUM(day_weight) AS active_days
        FROM agg_offenders_day_norm
        WHERE ticket_date BETWEEN ? AND ?
          AND violation_desc_long IN ({placeholders})
        GROUP BY 1
        ORDER BY score DESC
        LIMIT ?
        """
        params = [date_start, date_end] + list(violations) + [top_n]
    else:
        sql = f"""
        SELECT
          deid_lpn,
          {score_expr} AS score,
          SUM(ticket_count) AS total_tickets,
          SUM(day_weight) AS active_days
        FROM agg_offenders_day_norm
        WHERE ticket_date BETWEEN ? AND ?
        GROUP BY 1
        ORDER BY score DESC
        LIMIT ?
        """
        params = [date_start, date_end, top_n]

    return con.execute(sql, params).fetchdf()

def q_top_streets(con, date_start, date_end, violations, top_n, metric="total"):
    top_n = int(top_n)
    score_expr = "SUM(ticket_count)" if metric == "total" else "SUM(ticket_count) / SUM(day_weight)"

    if violations:
        placeholders = ",".join(["?"] * len(violations))
        sql = f"""
        SELECT
          street_key,
          {score_expr} AS score,
          SUM(ticket_count) AS total_tickets,
          SUM(day_weight) AS active_days
        FROM agg_streets_day_norm
        WHERE ticket_date BETWEEN ? AND ?
          AND violation_desc_long IN ({placeholders})
        GROUP BY 1
        ORDER BY score DESC
        LIMIT ?
        """
        params = [date_start, date_end] + list(violations) + [top_n]
    else:
        sql = f"""
        SELECT
          street_key,
          {score_expr} AS score,
          SUM(ticket_count) AS total_tickets,
          SUM(day_weight) AS active_days
        FROM agg_streets_day_norm
        WHERE ticket_date BETWEEN ? AND ?
        GROUP BY 1
        ORDER BY score DESC
        LIMIT ?
        """
        params = [date_start, date_end, top_n]

    return con.execute(sql, params).fetchdf()

def q_offender_violation_breakdown(con, deid_lpn, date_start, date_end, violations=None):
    deid_lpn = int(deid_lpn)
    violations = violations or []

    if violations:
        placeholders = ",".join(["?"] * len(violations))
        sql = f"""
        SELECT
          violation_desc_long,
          SUM(ticket_count) AS tickets
        FROM agg_offender_violation_day
        WHERE deid_lpn = ?
          AND ticket_date BETWEEN ? AND ?
          AND violation_desc_long IN ({placeholders})
        GROUP BY 1
        ORDER BY tickets DESC
        """
        params = [deid_lpn, date_start, date_end] + list(violations)
    else:
        sql = """
        SELECT
          violation_desc_long,
          SUM(ticket_count) AS tickets
        FROM agg_offender_violation_day
        WHERE deid_lpn = ?
          AND ticket_date BETWEEN ? AND ?
        GROUP BY 1
        ORDER BY tickets DESC
        """
        params = [deid_lpn, date_start, date_end]

    return con.execute(sql, params).fetchdf()

def q_offender_hour_profile(con, deid_lpn, date_start, date_end):
    deid_lpn = int(deid_lpn)
    sql = """
    SELECT
      issue_hour,
      SUM(ticket_count) AS tickets
    FROM agg_offender_hour_day
    WHERE deid_lpn = ?
      AND ticket_date BETWEEN ? AND ?
    GROUP BY 1
    ORDER BY 1
    """
    return con.execute(sql, [deid_lpn, date_start, date_end]).fetchdf()

st.set_page_config(page_title="Boston Violations Dashboard", layout="wide")
from PIL import Image

IMAGES_DIR = APP_DIR / "images"

header_img = Image.open(IMAGES_DIR / "header.png")
st.image(header_img, use_container_width=True)
st.markdown(
    "<h1 style='text-align: center;'>City of Boston Parking Violations</h1>",
    unsafe_allow_html=True,
)


con = get_con()

c1, c2, c3, c4 = st.columns([2, 2, 3, 2])

with c1:
    date_start = st.date_input("Start date", value=date(2024, 1, 1)).strftime("%Y-%m-%d")
with c2:
    date_end = st.date_input("End date", value=date(2024, 12, 31)).strftime("%Y-%m-%d")

viol_vals = get_violation_values()

with c3:
    violations = st.multiselect("Violations", options=viol_vals, default=[])
with c4:
    metric = st.selectbox("Metric", options=["total", "avg_per_day"], index=0)
    top_n = st.slider("Top N", min_value=5, max_value=100, value=20, step=5)

st.subheader("City Map of Violations")

df_map = q_map(con, date_start, date_end, violations)

value_col = "total_count" if metric == "total" else "avg_per_day"
df_map = df_map.dropna(subset=["lat_bin", "lon_bin", value_col])
# ---- Discrete 5-bin color scale based on quantiles
v = df_map[value_col].astype(float)

q20, q40, q60, q80 = v.quantile([0.2, 0.4, 0.6, 0.8]).tolist()

def color_bin(x):
    if x <= q20:
        return [173, 216, 230, 160]   # light blue
    elif x <= q40:
        return [0, 0, 139, 160]       # dark blue
    elif x <= q60:
        return [255, 215, 0, 160]     # yellow
    elif x <= q80:
        return [255, 140, 0, 160]     # orange
    else:
        return [220, 20, 60, 160]     # red

df_map["fill_color"] = v.apply(color_bin)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position="[lon_bin, lat_bin]",
    get_radius=25,
    pickable=True,
    auto_highlight=True,
    get_fill_color="fill_color",
)

view_state = pdk.ViewState(
    latitude=float(df_map["lat_bin"].mean()) if len(df_map) else 42.3601,
    longitude=float(df_map["lon_bin"].mean()) if len(df_map) else -71.0589,
    zoom=11,
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": f"{value_col}: {{{value_col}}}"},
)
st.pydeck_chart(deck)
# Legend for 5-bin color scale (quantiles)

# Updated legend to use actual quantile values
legend = [
    (f"Low (≤ {q20:.2f})", f"≤ {q20:.2f}", "rgb(173,216,230)"),  # light blue
    (f"Moderate-Low ({q20:.2f}–{q40:.2f})", f"{q20:.2f}–{q40:.2f}", "rgb(0,0,139)"),  # dark blue
    (f"Medium ({q40:.2f}–{q60:.2f})", f"{q40:.2f}–{q60:.2f}", "rgb(255,215,0)"),  # yellow
    (f"High ({q60:.2f}–{q80:.2f})", f"{q60:.2f}–{q80:.2f}", "rgb(255,140,0)"),  # orange
    (f"Very High (> {q80:.2f})", f"> {q80:.2f}", "rgb(220,20,60)"),  # red
]

legend_html = "<div style='display:flex; gap:14px; align-items:center; justify-content:center; flex-wrap:wrap; font-size:13px; margin: 0 auto;'>"
legend_html += f"<div><b>{value_col}</b></div>"
for name, rng, color in legend:
    legend_html += (
        "<div style='display:flex; align-items:center; gap:6px;'>"
        f"<span style='width:14px; height:14px; background:{color}; display:inline-block; border:1px solid #333;'></span>"
        f"<span>{name}</span>"
        "</div>"
    )
legend_html += "</div>"

st.markdown(legend_html, unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Offenders")
    df_off = q_top_offenders(con, date_start, date_end, violations, top_n, metric=metric)
    st.dataframe(df_off, use_container_width=True, height=320)

with col2:
    st.subheader("Top Streets")
    df_st = q_top_streets(con, date_start, date_end, violations, top_n, metric=metric)
    st.dataframe(df_st, use_container_width=True, height=320)

st.divider()

st.subheader("Offender Drilldown")

offender_options = df_off["deid_lpn"].tolist() if len(df_off) else []
selected = st.selectbox("Select offender", options=offender_options)

if selected is not None and selected != "":
    df_break = q_offender_violation_breakdown(con, selected, date_start, date_end, violations)
    st.write("Violation breakdown")
    st.dataframe(df_break, use_container_width=True, height=220)

    df_hour = q_offender_hour_profile(con, selected, date_start, date_end)
    st.write("Hourly profile")
    st.dataframe(df_hour, use_container_width=True, height=220)

