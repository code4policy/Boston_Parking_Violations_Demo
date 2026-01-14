from pathlib import Path
from datetime import date
import json
import hashlib
import colorsys
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
BOSTON_BOUNDARY_PATH = APP_DIR / "city_of_boston_outline_boundary_water_excluded.json"

CARTO_DARK = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"


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


@st.cache_data(show_spinner=False)
def load_geojson(path: str) -> dict:
    # utf-8-sig handles files saved with a BOM
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def maybe_load_geojson(path: Path) -> dict | None:
    if not path.exists():
        return None

    try:
        geo = load_geojson(path.as_posix())
    except Exception as e:
        st.warning(f"Could not load GeoJSON from {path.name}: {e}")
        return None

    if not isinstance(geo, dict) or "type" not in geo:
        st.warning(f"GeoJSON file {path.name} does not look valid.")
        return None

    return geo

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


def q_offender_points(
    con,
    date_start,
    date_end,
    violations,
    deid_lpns=None,
    limit=50000,
    exclude_deid_lpns=None,
):
    """Return individual ticket points for one/many offenders.

    - deid_lpns: None => all offenders
                int => one offender
                list[int] => many offenders
    """

    limit = int(limit)
    if deid_lpns is None:
        deid_lpns_list = None
    elif isinstance(deid_lpns, (list, tuple, set)):
        deid_lpns_list = [int(x) for x in deid_lpns]
    else:
        deid_lpns_list = [int(deid_lpns)]

    exclude_list = None
    if exclude_deid_lpns:
        exclude_list = [int(x) for x in exclude_deid_lpns]

    where = ["ticket_issue_date BETWEEN ? AND ?", "latitude IS NOT NULL", "longitude IS NOT NULL"]
    params = [date_start, date_end]

    if deid_lpns_list:
        placeholders = ",".join(["?"] * len(deid_lpns_list))
        where.append(f"deid_lpn IN ({placeholders})")
        params.extend(deid_lpns_list)

    if exclude_list:
        placeholders = ",".join(["?"] * len(exclude_list))
        where.append(f"deid_lpn NOT IN ({placeholders})")
        params.extend(exclude_list)

    if violations:
        placeholders = ",".join(["?"] * len(violations))
        where.append(f"violation_desc_long IN ({placeholders})")
        params.extend(list(violations))

    sql = f"""
    SELECT
      deid_lpn,
      latitude AS lat,
      longitude AS lon,
      ticket_issue_date,
      ticket_issue_time,
      violation_desc_long,
      location,
      street_name,
      street_no,
      ticket_number
    FROM tickets_enriched
    WHERE {' AND '.join(where)}
    LIMIT ?
    """
    params.append(limit)

    df = con.execute(sql, params).fetchdf()
    if df.empty:
        return df

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()
    df["feature_type"] = "Offender ticket"
    return df


def color_for_deid_lpn(deid_lpn: int, alpha: int = 180) -> list[int]:
    """Stable, visually distinct-ish color per offender id.

    Uses md5 hash -> hue; keeps saturation/value fixed for readability.
    Returns [r, g, b, a].
    """

    h = hashlib.md5(str(int(deid_lpn)).encode("utf-8")).digest()
    hue = int.from_bytes(h[:2], "big") / 65535.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.70, 0.95)
    return [int(r * 255), int(g * 255), int(b * 255), int(alpha)]

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

value_col = "total_count" if metric == "total" else "avg_per_day"

with st.expander("Map layers", expanded=True):
    st.caption("Basemap: Carto Dark (fixed)")
    offender_filter = st.selectbox(
        "Offenders",
        options=[
            "All offenders",
            "Top 5",
            "Top 10",
            "Top 20",
            "Top 30",
            "Top 50",
            "Enter specific deid_lpn",
        ],
        index=None,
        placeholder="Choose an option",
        help="Controls which offenders are plotted as individual ticket points.",
    )

    include_deid_187 = st.checkbox(
        "Include deid_lpn 187",
        value=True,
        help="Toggle this specific vehicle on/off in the offender points.",
    )

    offender_points_limit = st.slider(
        "Max offender points",
        min_value=500,
        max_value=50000,
        value=8000,
        step=500,
        help="Limits points drawn for performance.",
    )

selected_offenders = None
if offender_filter == "Enter specific deid_lpn":
    offender_text = st.text_input(
        "Enter deid_lpn",
        value="",
        placeholder="e.g. 123456",
    ).strip()
    if offender_text:
        if offender_text.isdigit():
            selected_offenders = [int(offender_text)]
        else:
            st.warning("deid_lpn must be numeric.")
elif offender_filter == "All offenders":
    selected_offenders = None
elif offender_filter in {"Top 5", "Top 10", "Top 20", "Top 30", "Top 50"}:
    top_n_sel = int(offender_filter.split()[-1])
    try:
        df_off_preview = q_top_offenders(
            con,
            date_start,
            date_end,
            violations,
            top_n=top_n_sel,
            metric=metric,
        )
        selected_offenders = df_off_preview["deid_lpn"].tolist() if len(df_off_preview) else []
    except Exception:
        selected_offenders = []

    if not selected_offenders:
        st.info("No offenders available for the current filters.")
        selected_offenders = None

if offender_filter is not None and not include_deid_187:
    # Remove 187 from any explicit selection list. For "All offenders" we exclude it in SQL.
    if selected_offenders:
        selected_offenders = [x for x in selected_offenders if int(x) != 187]
        if not selected_offenders and offender_filter != "All offenders":
            st.info("After excluding 187, there are no offenders to plot.")
boundary_geojson = maybe_load_geojson(BOSTON_BOUNDARY_PATH)

show_offender_layer = offender_filter is not None

df_map = pd.DataFrame()
if not show_offender_layer:
    df_map = q_map(con, date_start, date_end, violations)
    df_map = df_map.dropna(subset=["lat_bin", "lon_bin", value_col])

    # Align tooltip fields across layers (avoid 'undefined')
    if not df_map.empty:
        df_map["lat"] = df_map["lat_bin"]
        df_map["lon"] = df_map["lon_bin"]
        df_map["deid_lpn"] = ""
        df_map["ticket_issue_date"] = ""
        df_map["ticket_issue_time"] = ""
        df_map["location"] = ""
        df_map["ticket_number"] = ""

offender_points = pd.DataFrame()
if offender_filter == "All offenders":
    offender_points = q_offender_points(
        con,
        date_start,
        date_end,
        violations,
        None,
        limit=offender_points_limit,
        exclude_deid_lpns=([187] if not include_deid_187 else None),
    )
elif selected_offenders:
    offender_points = q_offender_points(
        con,
        date_start,
        date_end,
        violations,
        selected_offenders,
        limit=offender_points_limit,
        exclude_deid_lpns=([187] if not include_deid_187 else None),
    )

if offender_filter is not None:
    if not offender_points.empty:
        # Align metric fields across layers (avoid 'undefined')
        offender_points["total_count"] = ""
        offender_points["avg_per_day"] = ""
        offender_points[value_col] = ""
        offender_points["fill_color"] = offender_points["deid_lpn"].apply(color_for_deid_lpn)
    st.caption(f"Offender points shown: {len(offender_points):,}")

if show_offender_layer and offender_filter is not None and offender_points.empty:
    st.info("No offender ticket points found for the current filters.")

if not df_map.empty:
    df_map["feature_type"] = "Violation cell"

layers = []

if boundary_geojson is not None:
    layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            data=boundary_geojson,
            pickable=False,
            stroked=True,
            filled=True,
            extruded=False,
            wireframe=False,
            # High-contrast styling (works on dark/light basemaps)
            get_fill_color=[255, 255, 255, 10],
            get_line_color=[255, 140, 0, 220],
            line_width_min_pixels=1,
        )
    )

if offender_filter is not None and not offender_points.empty:
    offender_points = offender_points.copy()
    offender_points["radius_px"] = 5
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=offender_points,
            get_position="[lon, lat]",
            get_radius="radius_px",
            radius_units="pixels",
            pickable=True,
            auto_highlight=True,
            get_fill_color="fill_color",
            get_line_color=[255, 255, 255, 200],
            line_width_min_pixels=1,
        )
    )

q20 = q40 = q60 = q80 = 0.0
if not show_offender_layer:
    if df_map.empty:
        st.info("No violations found for the current filters.")
    else:
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

        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=df_map,
                get_position="[lon_bin, lat_bin]",
                get_radius=25,
                pickable=True,
                auto_highlight=True,
                get_fill_color="fill_color",
            )
        )

view_state = pdk.ViewState(
    latitude=(
        float(offender_points["lat"].mean())
        if (show_offender_layer and not offender_points.empty)
        else (float(df_map["lat_bin"].mean()) if len(df_map) else 42.3601)
    ),
    longitude=(
        float(offender_points["lon"].mean())
        if (show_offender_layer and not offender_points.empty)
        else (float(df_map["lon_bin"].mean()) if len(df_map) else -71.0589)
    ),
    zoom=13 if (show_offender_layer and not offender_points.empty) else 11,
)

map_style = CARTO_DARK

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    map_style=map_style,
    tooltip={
        "html": (
            (
                "<b>Offender ticket</b><br/>"
                "deid_lpn: {deid_lpn}<br/>"
                "Date: {ticket_issue_date}<br/>"
                "Time: {ticket_issue_time}<br/>"
                "Violation: {violation_desc_long}<br/>"
                "Location: {location}<br/>"
                "Ticket #: {ticket_number}"
            )
            if show_offender_layer
            else (
                "<b>Violation cell</b><br/>"
                f"<b>{value_col}</b>: {{{value_col}}}<br/>"
                "Total tickets: {total_count}<br/>"
                "Avg/day: {avg_per_day}<br/>"
                "Lat: {lat_bin}<br/>"
                "Lon: {lon_bin}"
            )
        ),
        "style": {
            "backgroundColor": "rgba(0, 0, 0, 0.80)",
            "color": "white",
            "fontSize": "12px",
        },
    },
)
st.pydeck_chart(deck)
if not show_offender_layer:
    # Legend for 5-bin color scale (quantiles)
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

