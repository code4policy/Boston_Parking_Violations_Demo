from pathlib import Path
from datetime import date
import json
import hashlib
import colorsys
import duckdb
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import altair as alt

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


def q_offender_time_series(
    con,
    date_start,
    date_end,
    violations,
    deid_lpns=None,
    exclude_deid_lpns=None,
):
    """Daily ticket counts for the selected offenders.

    - deid_lpns: None => all offenders (returns one series labeled "All offenders")
                int => one offender
                list[int] => many offenders
    """

    if deid_lpns is None:
        deid_lpns_list = None
    elif isinstance(deid_lpns, (list, tuple, set)):
        deid_lpns_list = [int(x) for x in deid_lpns]
    else:
        deid_lpns_list = [int(deid_lpns)]

    exclude_list = None
    if exclude_deid_lpns:
        exclude_list = [int(x) for x in exclude_deid_lpns]

    where = ["ticket_issue_date BETWEEN ? AND ?"]
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

    if deid_lpns_list:
        sql = f"""
        SELECT
          ticket_issue_date,
          deid_lpn,
          COUNT(*) AS tickets
        FROM tickets_enriched
        WHERE {' AND '.join(where)}
        GROUP BY 1,2
        ORDER BY 1,2
        """
    else:
        sql = f"""
        SELECT
          ticket_issue_date,
          'All offenders' AS deid_lpn,
          COUNT(*) AS tickets
        FROM tickets_enriched
        WHERE {' AND '.join(where)}
        GROUP BY 1
        ORDER BY 1
        """

    df = con.execute(sql, params).fetchdf()
    if df.empty:
        return df
    df["ticket_issue_date"] = pd.to_datetime(df["ticket_issue_date"], errors="coerce")
    df = df.dropna(subset=["ticket_issue_date"]).copy()
    return df


def q_offender_time_heatmap(
    con,
    date_start,
    date_end,
    violations,
    deid_lpns=None,
    exclude_deid_lpns=None,
):
    """Hour-of-day x day-of-week heatmap for selected offenders (aggregated together)."""

    if deid_lpns is None:
        deid_lpns_list = None
    elif isinstance(deid_lpns, (list, tuple, set)):
        deid_lpns_list = [int(x) for x in deid_lpns]
    else:
        deid_lpns_list = [int(deid_lpns)]

    exclude_list = None
    if exclude_deid_lpns:
        exclude_list = [int(x) for x in exclude_deid_lpns]

    where = ["ticket_issue_date BETWEEN ? AND ?"]
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
    WITH base AS (
      SELECT
        CAST(ticket_issue_date AS DATE) AS d,
        TRY_CAST(SUBSTR(ticket_issue_time, 1, 2) AS INTEGER) AS issue_hour
      FROM tickets_enriched
      WHERE {' AND '.join(where)}
    )
    SELECT
      STRFTIME(d, '%a') AS day_name,
      CAST(STRFTIME(d, '%w') AS INTEGER) AS day_num,
      issue_hour,
      COUNT(*) AS tickets
    FROM base
    WHERE issue_hour BETWEEN 0 AND 23
    GROUP BY 1,2,3
    ORDER BY 2,3
    """

    return con.execute(sql, params).fetchdf()


def q_first_time_offender_points(
        con,
        date_start,
        date_end,
        violations,
        limit=50000,
        exclude_deid_lpns=None,
):
        """Return ticket points for offenders whose first-ever ticket date falls within the selected window.

        Definition used here:
        - "First-time offender" means: MIN(ticket_issue_date) for that deid_lpn is between date_start and date_end.
        - Points returned are still limited to tickets within the selected date window.
        """

        limit = int(limit)

        exclude_list = None
        if exclude_deid_lpns:
                exclude_list = [int(x) for x in exclude_deid_lpns]

        where = [
                "t.ticket_issue_date BETWEEN ? AND ?",
                "t.latitude IS NOT NULL",
                "t.longitude IS NOT NULL",
        ]
        params = [date_start, date_end]

        if exclude_list:
                placeholders = ",".join(["?"] * len(exclude_list))
                where.append(f"t.deid_lpn NOT IN ({placeholders})")
                params.extend(exclude_list)

        if violations:
                placeholders = ",".join(["?"] * len(violations))
                where.append(f"t.violation_desc_long IN ({placeholders})")
                params.extend(list(violations))

        # Params for first-ticket window
        first_params = [date_start, date_end]

        sql = f"""
        WITH firsts AS (
            SELECT
                deid_lpn,
                MIN(ticket_issue_date) AS first_date
            FROM tickets_enriched
            WHERE deid_lpn IS NOT NULL
            GROUP BY 1
        )
        SELECT
            t.deid_lpn,
            t.latitude AS lat,
            t.longitude AS lon,
            t.ticket_issue_date,
            t.ticket_issue_time,
            t.violation_desc_long,
            t.location,
            t.street_name,
            t.street_no,
            t.ticket_number
        FROM tickets_enriched t
        JOIN firsts f
            ON t.deid_lpn = f.deid_lpn
        WHERE f.first_date BETWEEN ? AND ?
            AND {' AND '.join(where)}
        LIMIT ?
        """

        df = con.execute(sql, first_params + params + [limit]).fetchdf()
        if df.empty:
                return df

        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat", "lon"]).copy()
        df["feature_type"] = "First-time offender ticket"
        return df


def q_first_time_offender_time_series(
        con,
        date_start,
        date_end,
        violations,
        exclude_deid_lpns=None,
):
        """Daily ticket counts for the first-time offender cohort."""

        exclude_list = None
        if exclude_deid_lpns:
                exclude_list = [int(x) for x in exclude_deid_lpns]

        where = ["t.ticket_issue_date BETWEEN ? AND ?"]
        params = [date_start, date_end]

        if exclude_list:
                placeholders = ",".join(["?"] * len(exclude_list))
                where.append(f"t.deid_lpn NOT IN ({placeholders})")
                params.extend(exclude_list)

        if violations:
                placeholders = ",".join(["?"] * len(violations))
                where.append(f"t.violation_desc_long IN ({placeholders})")
                params.extend(list(violations))

        first_params = [date_start, date_end]

        sql = f"""
        WITH firsts AS (
            SELECT
                deid_lpn,
                MIN(ticket_issue_date) AS first_date
            FROM tickets_enriched
            WHERE deid_lpn IS NOT NULL
            GROUP BY 1
        )
        SELECT
            t.ticket_issue_date,
            'First-time offenders' AS deid_lpn,
            COUNT(*) AS tickets
        FROM tickets_enriched t
        JOIN firsts f
            ON t.deid_lpn = f.deid_lpn
        WHERE f.first_date BETWEEN ? AND ?
            AND {' AND '.join(where)}
        GROUP BY 1
        ORDER BY 1
        """

        df = con.execute(sql, first_params + params).fetchdf()
        if df.empty:
                return df
        df["ticket_issue_date"] = pd.to_datetime(df["ticket_issue_date"], errors="coerce")
        df = df.dropna(subset=["ticket_issue_date"]).copy()
        return df


def q_first_time_offender_time_heatmap(
        con,
        date_start,
        date_end,
        violations,
        exclude_deid_lpns=None,
):
        """Hour-of-day x day-of-week heatmap for the first-time offender cohort."""

        exclude_list = None
        if exclude_deid_lpns:
                exclude_list = [int(x) for x in exclude_deid_lpns]

        where = ["t.ticket_issue_date BETWEEN ? AND ?"]
        params = [date_start, date_end]

        if exclude_list:
                placeholders = ",".join(["?"] * len(exclude_list))
                where.append(f"t.deid_lpn NOT IN ({placeholders})")
                params.extend(exclude_list)

        if violations:
                placeholders = ",".join(["?"] * len(violations))
                where.append(f"t.violation_desc_long IN ({placeholders})")
                params.extend(list(violations))

        first_params = [date_start, date_end]

        sql = f"""
        WITH firsts AS (
            SELECT
                deid_lpn,
                MIN(ticket_issue_date) AS first_date
            FROM tickets_enriched
            WHERE deid_lpn IS NOT NULL
            GROUP BY 1
        ),
        base AS (
            SELECT
                CAST(t.ticket_issue_date AS DATE) AS d,
                TRY_CAST(SUBSTR(t.ticket_issue_time, 1, 2) AS INTEGER) AS issue_hour
            FROM tickets_enriched t
            JOIN firsts f
                ON t.deid_lpn = f.deid_lpn
            WHERE f.first_date BETWEEN ? AND ?
                AND {' AND '.join(where)}
        )
        SELECT
            STRFTIME(d, '%a') AS day_name,
            CAST(STRFTIME(d, '%w') AS INTEGER) AS day_num,
            issue_hour,
            COUNT(*) AS tickets
        FROM base
        WHERE issue_hour BETWEEN 0 AND 23
        GROUP BY 1,2,3
        ORDER BY 2,3
        """

        return con.execute(sql, first_params + params).fetchdf()


def q_one_time_offender_points(
    con,
    date_start,
    date_end,
    violations,
    limit=50000,
    exclude_deid_lpns=None,
):
    """Return ticket points for offenders who received exactly one ticket in the dataset.

    Notes:
    - Cohort definition is global: COUNT(*) over all of tickets_enriched for a deid_lpn equals 1.
    - Returned points are still limited to tickets within the selected date window.
    """

    limit = int(limit)

    exclude_list = None
    if exclude_deid_lpns:
        exclude_list = [int(x) for x in exclude_deid_lpns]

    where = [
        "t.ticket_issue_date BETWEEN ? AND ?",
        "t.latitude IS NOT NULL",
        "t.longitude IS NOT NULL",
    ]
    params = [date_start, date_end]

    if exclude_list:
        placeholders = ",".join(["?"] * len(exclude_list))
        where.append(f"t.deid_lpn NOT IN ({placeholders})")
        params.extend(exclude_list)

    if violations:
        placeholders = ",".join(["?"] * len(violations))
        where.append(f"t.violation_desc_long IN ({placeholders})")
        params.extend(list(violations))

    sql = f"""
    WITH counts AS (
        SELECT
            deid_lpn,
            COUNT(*) AS n_tickets
        FROM tickets_enriched
        WHERE deid_lpn IS NOT NULL
        GROUP BY 1
    )
    SELECT
        t.deid_lpn,
        t.latitude AS lat,
        t.longitude AS lon,
        t.ticket_issue_date,
        t.ticket_issue_time,
        t.violation_desc_long,
        t.location,
        t.street_name,
        t.street_no,
        t.ticket_number
    FROM tickets_enriched t
    JOIN counts c
        ON t.deid_lpn = c.deid_lpn
    WHERE c.n_tickets = 1
        AND {' AND '.join(where)}
    LIMIT ?
    """

    df = con.execute(sql, params + [limit]).fetchdf()
    if df.empty:
        return df

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()
    df["feature_type"] = "One-time offender ticket"
    return df


def q_one_time_offender_time_series(
    con,
    date_start,
    date_end,
    violations,
    exclude_deid_lpns=None,
):
    """Daily ticket counts for the one-time offender cohort."""

    exclude_list = None
    if exclude_deid_lpns:
        exclude_list = [int(x) for x in exclude_deid_lpns]

    where = ["t.ticket_issue_date BETWEEN ? AND ?"]
    params = [date_start, date_end]

    if exclude_list:
        placeholders = ",".join(["?"] * len(exclude_list))
        where.append(f"t.deid_lpn NOT IN ({placeholders})")
        params.extend(exclude_list)

    if violations:
        placeholders = ",".join(["?"] * len(violations))
        where.append(f"t.violation_desc_long IN ({placeholders})")
        params.extend(list(violations))

    sql = f"""
    WITH counts AS (
        SELECT
            deid_lpn,
            COUNT(*) AS n_tickets
        FROM tickets_enriched
        WHERE deid_lpn IS NOT NULL
        GROUP BY 1
    )
    SELECT
        t.ticket_issue_date,
        'One-time offenders' AS deid_lpn,
        COUNT(*) AS tickets
    FROM tickets_enriched t
    JOIN counts c
        ON t.deid_lpn = c.deid_lpn
    WHERE c.n_tickets = 1
        AND {' AND '.join(where)}
    GROUP BY 1
    ORDER BY 1
    """

    df = con.execute(sql, params).fetchdf()
    if df.empty:
        return df
    df["ticket_issue_date"] = pd.to_datetime(df["ticket_issue_date"], errors="coerce")
    df = df.dropna(subset=["ticket_issue_date"]).copy()
    return df


def q_one_time_offender_time_heatmap(
    con,
    date_start,
    date_end,
    violations,
    exclude_deid_lpns=None,
):
    """Hour-of-day x day-of-week heatmap for the one-time offender cohort."""

    exclude_list = None
    if exclude_deid_lpns:
        exclude_list = [int(x) for x in exclude_deid_lpns]

    where = ["t.ticket_issue_date BETWEEN ? AND ?"]
    params = [date_start, date_end]

    if exclude_list:
        placeholders = ",".join(["?"] * len(exclude_list))
        where.append(f"t.deid_lpn NOT IN ({placeholders})")
        params.extend(exclude_list)

    if violations:
        placeholders = ",".join(["?"] * len(violations))
        where.append(f"t.violation_desc_long IN ({placeholders})")
        params.extend(list(violations))

    sql = f"""
    WITH counts AS (
        SELECT
            deid_lpn,
            COUNT(*) AS n_tickets
        FROM tickets_enriched
        WHERE deid_lpn IS NOT NULL
        GROUP BY 1
    ),
    base AS (
        SELECT
            CAST(t.ticket_issue_date AS DATE) AS d,
            TRY_CAST(SUBSTR(t.ticket_issue_time, 1, 2) AS INTEGER) AS issue_hour
        FROM tickets_enriched t
        JOIN counts c
            ON t.deid_lpn = c.deid_lpn
        WHERE c.n_tickets = 1
            AND {' AND '.join(where)}
    )
    SELECT
        STRFTIME(d, '%a') AS day_name,
        CAST(STRFTIME(d, '%w') AS INTEGER) AS day_num,
        issue_hour,
        COUNT(*) AS tickets
    FROM base
    WHERE issue_hour BETWEEN 0 AND 23
    GROUP BY 1,2,3
    ORDER BY 2,3
    """

    return con.execute(sql, params).fetchdf()


def color_for_deid_lpn(deid_lpn: int, alpha: int = 180) -> list[int]:
    """Stable, visually distinct-ish color per offender id.

    Uses md5 hash -> hue; keeps saturation/value fixed for readability.
    Returns [r, g, b, a].
    """

    h = hashlib.md5(str(int(deid_lpn)).encode("utf-8")).digest()
    hue = int.from_bytes(h[:2], "big") / 65535.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.70, 0.95)
    return [int(r * 255), int(g * 255), int(b * 255), int(alpha)]


def rgba_to_hex(rgba: list[int]) -> str:
    r, g, b = [max(0, min(255, int(x))) for x in rgba[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"

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
                ORDER BY score DESC, deid_lpn ASC
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
        ORDER BY score DESC, deid_lpn ASC
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
                ORDER BY score DESC, street_key ASC
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
        ORDER BY score DESC, street_key ASC
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

header_img = Image.open(IMAGES_DIR / "dashboard_header.png")
st.image(header_img, use_container_width=True)
st.markdown(
    "<h1 style='text-align: center;'>City of Boston Parking Violations Analysis</h1>",
    unsafe_allow_html=True,
)

DISCLAIMER_TEXT = "Unofficial - Only intended as an academic exercise"
DISCLAIMER_HTML = (
    "<div style='text-align:center; font-weight:700; font-size:14px; color: inherit; margin-top: -6px; margin-bottom: 8px;'>"
    + DISCLAIMER_TEXT
    + "</div>"
)
st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)

st.markdown(
    """This dashboard explores City of Boston parking violations in 2024 using pre-aggregated Parquet artifacts and a de-identified ticket dataset. Use the date and violation filters to see how enforcement patterns shift across neighborhoods, streets, and time. The map can be viewed as either point clusters (binned cells) or a heatmap, and the offender views let you compare all vehicles vs. first-time vs. one-time vs. top repeat offenders. The time-pattern charts summarize when violations happen (by day and hour) and help highlight consistent routines versus spikes driven by specific offenders or dates."""
)


con = get_con()

c1, c2, c3, c4 = st.columns([2, 2, 3, 2])

# Fixed defaults (Metric removed from UI)
metric = "total"
top_n = 20

with c1:
    date_start = st.date_input("Start date", value=date(2024, 1, 1)).strftime("%Y-%m-%d")
with c2:
    date_end = st.date_input("End date", value=date(2024, 12, 31)).strftime("%Y-%m-%d")

viol_vals = get_violation_values()

with c3:
    violations = st.multiselect("Violations", options=viol_vals, default=[])
with c4:
    offender_filter = st.selectbox(
        "Parking offenders",
        options=[
            "All offenders",
            "One-time offenders",
            "Top 5 repeat offenders",
            "Top 10 repeat offenders",
            "Top 20 repeat offenders",
            "Enter specific deid_lpn",
        ],
        index=None,
        placeholder="Choose an option",
        help="Controls which offenders are plotted as individual ticket points.",
    )

# Map visualization toggle
map_viz = st.radio(
    "Map visualization",
    options=["Map", "Heatmap"],
    index=0,
    horizontal=True,
    key="map_visualization_mode",
    help="Switch between the existing map style and a heatmap layer.",
)

# Default heatmap rendering parameters (applied automatically; no UI).
# Tweaking knobs:
# - opacity: lower = less intense
# - radius_pixels: smaller = sharper hotspots
# - weight_scale: reduces extreme hotspots for grid heatmap
HEATMAP_OPACITY_DEFAULT = 0.70
HEATMAP_RADIUS_PX_DEFAULT = 13
HEATMAP_WEIGHT_SCALE_DEFAULT = "sqrt"  # one of: "sqrt", "log1p", "linear"

# (Removed: "City Map of Violations" header)

value_col = "total_count" if metric == "total" else "avg_per_day"

# Controls (no expander/frame)
controls = st.container()
with controls:
    include_deid_187 = st.checkbox(
        "Include vehicles with no license plate",
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

# Apply fixed heatmap configuration (no user control).
if map_viz == "Heatmap":
    heatmap_opacity = HEATMAP_OPACITY_DEFAULT
    heatmap_radius_px = HEATMAP_RADIUS_PX_DEFAULT
    heatmap_weight_scale = HEATMAP_WEIGHT_SCALE_DEFAULT
else:
    heatmap_opacity = 0.85
    heatmap_radius_px = 35
    heatmap_weight_scale = "linear"

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
elif offender_filter == "First-time offenders":
    selected_offenders = None
elif offender_filter == "One-time offenders":
    selected_offenders = None
elif offender_filter in {"Top 5 repeat offenders", "Top 10 repeat offenders", "Top 20 repeat offenders"}:
    top_n_sel = int(offender_filter.split()[1])
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

show_offender_layer = offender_filter is not None

# NOTE: Streamlit's st.pydeck_chart does not expose hover/click events from the map.
# We approximate "linking" via an explicit focused offender selector that syncs map + charts.
focus_offender = None
if show_offender_layer and selected_offenders and len(selected_offenders) <= 50:
    focus_choice = st.selectbox(
        "Focus offender (sync map + charts)",
        options=["(none)"] + [str(x) for x in selected_offenders],
        index=0,
        help="Fades other offenders on the map and in the trend chart.",
    )
    if focus_choice != "(none)":
        focus_offender = int(focus_choice)

boundary_geojson = maybe_load_geojson(BOSTON_BOUNDARY_PATH)

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
elif offender_filter == "First-time offenders":
    offender_points = q_first_time_offender_points(
        con,
        date_start,
        date_end,
        violations,
        limit=offender_points_limit,
        exclude_deid_lpns=([187] if not include_deid_187 else None),
    )
elif offender_filter == "One-time offenders":
    offender_points = q_one_time_offender_points(
        con,
        date_start,
        date_end,
        violations,
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
        if focus_offender is not None:
            offender_points["fill_color"] = offender_points["deid_lpn"].apply(
                lambda x: color_for_deid_lpn(int(x), alpha=(230 if int(x) == focus_offender else 40))
            )
            offender_points["radius_px"] = offender_points["deid_lpn"].apply(
                lambda x: (8 if int(x) == focus_offender else 4)
            )
        else:
            offender_points["fill_color"] = offender_points["deid_lpn"].apply(color_for_deid_lpn)
            offender_points["radius_px"] = 5

        # Weight used by the map heatmap layer.
        offender_points["heat_weight"] = 1.0
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
    if "radius_px" not in offender_points.columns:
        offender_points["radius_px"] = 5
    if map_viz == "Heatmap":
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=offender_points,
                get_position="[lon, lat]",
                get_weight="heat_weight",
                radius_pixels=heatmap_radius_px,
                opacity=heatmap_opacity,
                pickable=False,
            )
        )
    else:
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
        if map_viz == "Heatmap":
            # Soften extreme hotspots by scaling weights.
            if not df_map.empty:
                w = pd.to_numeric(df_map[value_col], errors="coerce").fillna(0.0).clip(lower=0.0)
                if heatmap_weight_scale == "sqrt":
                    df_map["heat_weight"] = np.sqrt(w)
                elif heatmap_weight_scale == "log1p":
                    df_map["heat_weight"] = np.log1p(w)
                else:
                    df_map["heat_weight"] = w

            layers.append(
                pdk.Layer(
                    "HeatmapLayer",
                    data=df_map,
                    get_position="[lon_bin, lat_bin]",
                    get_weight="heat_weight",
                    radius_pixels=heatmap_radius_px,
                    opacity=heatmap_opacity,
                    pickable=False,
                )
            )
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
st.pydeck_chart(deck, use_container_width=True, height=700)
if not show_offender_layer and map_viz != "Heatmap":
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

if show_offender_layer:
    exclude_lpns_for_time = [187] if not include_deid_187 else None

    if offender_filter == "First-time offenders":
        df_ts = q_first_time_offender_time_series(
            con,
            date_start,
            date_end,
            violations,
            exclude_deid_lpns=exclude_lpns_for_time,
        )
        df_hm = q_first_time_offender_time_heatmap(
            con,
            date_start,
            date_end,
            violations,
            exclude_deid_lpns=exclude_lpns_for_time,
        )
        hm_focus = None
    elif offender_filter == "One-time offenders":
        df_ts = q_one_time_offender_time_series(
            con,
            date_start,
            date_end,
            violations,
            exclude_deid_lpns=exclude_lpns_for_time,
        )
        df_hm = q_one_time_offender_time_heatmap(
            con,
            date_start,
            date_end,
            violations,
            exclude_deid_lpns=exclude_lpns_for_time,
        )
        hm_focus = None
    else:
        df_ts = q_offender_time_series(
            con,
            date_start,
            date_end,
            violations,
            deid_lpns=selected_offenders,
            exclude_deid_lpns=exclude_lpns_for_time,
        )
        hm_focus = focus_offender if focus_offender is not None else None
        df_hm = q_offender_time_heatmap(
            con,
            date_start,
            date_end,
            violations,
            deid_lpns=([hm_focus] if hm_focus is not None else selected_offenders),
            exclude_deid_lpns=exclude_lpns_for_time,
        )

    st.markdown(
        "<div style='font-size:18px; font-weight:600; margin-top: 8px; margin-bottom: 6px;'>Time patterns</div>",
        unsafe_allow_html=True,
    )

    # Keep the charts readable by default.
    c_opts1, c_opts2, c_opts3 = st.columns([2, 2, 3])
    with c_opts1:
        trend_grain = st.selectbox(
            "Trend granularity",
            options=["Weekly", "Daily"],
            index=0,
            help="Weekly is less noisy and usually more insightful.",
        )
    with c_opts2:
        top_k_series = st.selectbox(
            "Lines shown",
            options=[3, 5, 8, 12],
            index=1,
            help="Shows Total + Top K offenders + Others.",
        )
    with c_opts3:
        st.caption("Tip: use Top 5/10 in Parking offenders for cleaner patterns.")

    # ---- Summary metrics
    df_ts_metrics = df_ts
    if focus_offender is not None and not df_ts.empty:
        df_ts_metrics = df_ts[df_ts["deid_lpn"].astype(str) == str(focus_offender)].copy()

    if not df_ts_metrics.empty:
        df_total_daily = (
            df_ts_metrics.groupby("ticket_issue_date", as_index=False)["tickets"].sum().sort_values("ticket_issue_date")
        )
        total_tickets = int(df_total_daily["tickets"].sum())
        peak_day_row = df_total_daily.loc[df_total_daily["tickets"].idxmax()]
        peak_day = peak_day_row["ticket_issue_date"].date().isoformat()
        peak_day_tickets = int(peak_day_row["tickets"])
    else:
        total_tickets = 0
        peak_day = "—"
        peak_day_tickets = 0

    if not df_hm.empty:
        peak_cell = df_hm.loc[df_hm["tickets"].idxmax()]
        peak_cell_label = f"{peak_cell['day_name']} {int(peak_cell['issue_hour']):02d}:00"
        peak_cell_tickets = int(peak_cell["tickets"])
    else:
        peak_cell_label = "—"
        peak_cell_tickets = 0

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Tickets (focused offender)" if focus_offender is not None else "Tickets (selected offenders)",
        f"{total_tickets:,}",
    )
    m2.metric("Peak day", peak_day, delta=f"{peak_day_tickets:,} tickets")
    m3.metric(
        "Peak hour (focused)" if focus_offender is not None else "Peak hour",
        peak_cell_label,
        delta=f"{peak_cell_tickets:,} tickets",
    )

    c_ts, c_hm = st.columns(2)

    with c_ts:
        if df_ts.empty:
            st.info("No time-series data available for the current offender filters.")
        else:
            df_plot = df_ts.copy()
            df_plot["series"] = df_plot["deid_lpn"].astype(str)
            df_plot["d"] = pd.to_datetime(df_plot["ticket_issue_date"], errors="coerce")
            df_plot = df_plot.dropna(subset=["d"]).copy()

            if trend_grain == "Weekly":
                df_plot["d"] = df_plot["d"].dt.to_period("W-MON").apply(lambda p: p.start_time)

            totals = df_plot.groupby("series", as_index=False)["tickets"].sum().sort_values("tickets", ascending=False)
            top_series = totals.head(int(top_k_series))["series"].tolist()

            focus_series = str(focus_offender) if focus_offender is not None else None
            if focus_series and focus_series in set(df_plot["series"].tolist()):
                # Ensure the focused offender is always shown as its own line.
                top_series = list(dict.fromkeys([focus_series] + top_series))[: int(top_k_series)]

            # Build Total + Top K + Others
            df_top = df_plot[df_plot["series"].isin(top_series)].copy()
            df_others = df_plot[~df_plot["series"].isin(top_series)].copy()
            if not df_others.empty:
                df_others = (
                    df_others.groupby("d", as_index=False)["tickets"].sum().assign(series="Others")
                )
            df_total = df_plot.groupby("d", as_index=False)["tickets"].sum().assign(series="Total")

            df_lines = pd.concat([df_top[["d", "tickets", "series"]], df_others, df_total], ignore_index=True)
            df_lines = df_lines.groupby(["d", "series"], as_index=False)["tickets"].sum()

            # Color mapping (stable per offender; fixed for Total/Others)
            domain = []
            rng = []
            for s in sorted(set(df_lines["series"].tolist()), key=lambda x: (x not in {"Total", "Others"}, x)):
                domain.append(s)
                if s == "Total":
                    rng.append("#ffffff")
                elif s == "Others":
                    rng.append("#9aa0a6")
                else:
                    try:
                        rng.append(rgba_to_hex(color_for_deid_lpn(int(s))))
                    except Exception:
                        rng.append("#4e79a7")

            base = alt.Chart(df_lines).encode(
                x=alt.X("d:T", title=""),
                y=alt.Y("tickets:Q", title="Tickets"),
                color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=rng), legend=alt.Legend(title="")),
                tooltip=[
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("d:T", title="Date"),
                    alt.Tooltip("tickets:Q", title="Tickets", format=",")
                ],
            )

            if focus_series:
                fade = alt.condition(alt.datum.series == focus_series, alt.value(1.0), alt.value(0.08))
                stroke_w = alt.condition(alt.datum.series == focus_series, alt.value(3.5), alt.value(1.5))
                lines = base.mark_line(interpolate="monotone").encode(opacity=fade, strokeWidth=stroke_w)
                chart = (
                    lines
                    .properties(height=280)
                    .configure_view(strokeOpacity=0)
                    .configure_axis(grid=True, gridOpacity=0.15)
                    .configure_legend(orient="bottom")
                )
            else:
                hover = alt.selection_point(fields=["series"], on="mouseover", nearest=True, empty=False)
                base_h = base.encode(opacity=alt.condition(hover, alt.value(1.0), alt.value(0.30)))
                lines = base_h.mark_line(interpolate="monotone", strokeWidth=2)
                points = base_h.mark_circle(size=40).encode(opacity=alt.condition(hover, alt.value(1.0), alt.value(0.0))).add_params(hover)
                chart = (
                    (lines + points)
                    .properties(height=280)
                    .configure_view(strokeOpacity=0)
                    .configure_axis(grid=True, gridOpacity=0.15)
                    .configure_legend(orient="bottom")
                )
            st.altair_chart(chart, use_container_width=True)

    with c_hm:
        if df_hm.empty:
            st.info("No heatmap data available for the current offender filters.")
        else:
            day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            st.vega_lite_chart(
                df_hm,
                {
                    "mark": {"type": "rect"},
                    "encoding": {
                        "x": {
                            "field": "issue_hour",
                            "type": "ordinal",
                            "title": "Hour of day",
                            "sort": "ascending",
                        },
                        "y": {
                            "field": "day_name",
                            "type": "ordinal",
                            "title": "Day of week",
                            "sort": day_order,
                        },
                        "color": {
                            "field": "tickets",
                            "type": "quantitative",
                            "title": "Tickets",
                            "scale": {"scheme": "viridis"},
                        },
                        "tooltip": [
                            {"field": "day_name", "type": "nominal", "title": "Day"},
                            {"field": "issue_hour", "type": "ordinal", "title": "Hour"},
                            {"field": "tickets", "type": "quantitative", "title": "Tickets"},
                        ],
                    },
                    "width": "container",
                    "height": 280,
                    "config": {
                        "view": {"stroke": None},
                        "axis": {"grid": True, "gridOpacity": 0.15},
                    },
                },
                use_container_width=True,
            )

st.divider()

# Controls for the ranking tables
top_n = st.slider(
    "Top N",
    min_value=5,
    max_value=100,
    value=int(top_n),
    step=5,
    key="top_n_tables",
    help="Controls how many rows appear in the Top Offenders / Top Streets tables.",
)

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

offender_options = []
if len(df_off):
    offender_options = (
        pd.to_numeric(df_off["deid_lpn"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )

_DRILLDOWN_KEY = "offender_drilldown_deid_lpn"
if not offender_options:
    st.info("No offenders available for drilldown with the current filters.")
    selected = None
else:
    # Preserve selection across reruns even if the Top Offenders query changes ordering
    # or ties cause an offender to fall out of the list.
    if _DRILLDOWN_KEY not in st.session_state:
        st.session_state[_DRILLDOWN_KEY] = offender_options[0]

    try:
        current = int(st.session_state.get(_DRILLDOWN_KEY))
    except Exception:
        current = offender_options[0]
        st.session_state[_DRILLDOWN_KEY] = current

    if current not in offender_options:
        offender_options = [current] + offender_options
        st.session_state[_DRILLDOWN_KEY] = current

    selected = st.selectbox("Select offender", options=offender_options, key=_DRILLDOWN_KEY)

if selected is not None and selected != "":
    df_break = q_offender_violation_breakdown(con, selected, date_start, date_end, violations)
    st.write("Violation breakdown")
    st.dataframe(df_break, use_container_width=True, height=220)

st.divider()
st.markdown(
    "<div style='text-align:center; font-weight:700; font-size:14px; color: inherit; margin-top: 6px; margin-bottom: 0px;'>"
    + DISCLAIMER_TEXT
    + "</div>",
    unsafe_allow_html=True,
)

