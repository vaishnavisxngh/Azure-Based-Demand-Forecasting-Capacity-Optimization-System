import os
from datetime import date
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# BASIC CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Azure Demand & Capacity Console",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_URL = "http://localhost:5000/api"

# -----------------------------------------------------------------------------
# DARK THEME + NAV BUTTON STYLING + POINTER CURSOR
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
/* App background */
.stApp {
    background: radial-gradient(circle at top left, #020617 0, #020617 40%, #020617 100%);
    color: #e5e7eb;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Header bar */
.app-header {
    background: linear-gradient(120deg, rgba(56,189,248,0.2), rgba(37,99,235,0.2));
    border-radius: 1rem;
    padding: 0.9rem 1.2rem;
    border: 1px solid rgba(59,130,246,0.7);
    box-shadow: 0 18px 45px rgba(15,23,42,0.95);
    margin-bottom: 0.9rem;
}
.app-header-title {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.app-header-sub {
    font-size: 0.82rem;
    color: #cbd5f5;
}

/* Cards */
.card {
    background: rgba(15,23,42,0.96);
    border-radius: 0.9rem;
    padding: 0.9rem 1rem;
    border: 1px solid rgba(55,65,81,0.9);
    box-shadow: 0 14px 40px rgba(0,0,0,0.9);
}

/* KPI */
.kpi {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}
.kpi-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
}
.kpi-value {
    font-size: 1.25rem;
    font-weight: 600;
    color: #f9fafb;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #a5b4fc;
}

/* Info / warning strips */
.info-strip {
    background: rgba(37,99,235,0.12);
    border-left: 3px solid #38bdf8;
    border-radius: 0.5rem;
    padding: 0.4rem 0.7rem;
    font-size: 0.8rem;
    margin-bottom: 0.5rem;
    color: #e5e7eb;
}
.warn-strip {
    background: rgba(120,53,15,0.95);
    border-left: 3px solid #f59e0b;
    border-radius: 0.5rem;
    padding: 0.4rem 0.7rem;
    font-size: 0.8rem;
    margin-bottom: 0.5rem;
    color: #fed7aa;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1f2937;
}
.sidebar-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 0.1rem;
}
.sidebar-sub {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-bottom: 0.7rem;
}
.nav-section-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin: 0.3rem 0 0.2rem 0;
}

/* Nav buttons container */
.nav-btn {
    width: 100%;
}

/* Pointer cursor for all interactive elements */
button,
[data-testid="stDownloadButton"],
[data-baseweb="select"] *,
[role="combobox"],
input,
textarea {
    cursor: pointer !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# API HELPERS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def api_get(endpoint: str, params: Optional[Dict[str, Any]] = None):
    url = f"{BASE_URL}/{endpoint}"
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        # Show once per call-site; no spam
        st.error(f"API error from `{endpoint}`: {e}")
        return None


@st.cache_data(ttl=600)
def load_filters() -> Dict[str, Any]:
    data = api_get("filters/options")
    return data or {}


@st.cache_data(ttl=300)
def load_raw_data() -> pd.DataFrame:
    data = api_get("data/raw")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def filter_dataframe(df: pd.DataFrame, regions, resources, start: Optional[date], end: Optional[date]) -> pd.DataFrame:
    """Apply per-page filters."""
    if df.empty:
        return df.copy()
    out = df.copy()
    if "region" in out.columns and regions:
        out = out[out["region"].isin(regions)]
    if "resource_type" in out.columns and resources:
        out = out[out["resource_type"].isin(resources)]
    if "date" in out.columns and start and end:
        out = out[(out["date"].dt.date >= start) & (out["date"].dt.date <= end)]
    return out


def render_page_filters(prefix: str, df: pd.DataFrame, filters_meta: Dict[str, Any]):
    """Render region/resource/date range filters for each page (independent)."""
    if df.empty:
        return [], [], None, None

    regions_all = filters_meta.get("regions") or (sorted(df["region"].dropna().unique()) if "region" in df.columns else [])
    res_all = filters_meta.get("resource_types") or (
        sorted(df["resource_type"].dropna().unique()) if "resource_type" in df.columns else []
    )

    if "date" in df.columns and not df["date"].empty:
        min_d = df["date"].min().date()
        max_d = df["date"].max().date()
    else:
        min_d = max_d = None

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_regions = st.multiselect(
            "Regions",
            options=regions_all,
            default=regions_all,
            key=f"{prefix}_regions",
        )
    with col2:
        sel_resources = st.multiselect(
            "Resource types",
            options=res_all,
            default=res_all,
            key=f"{prefix}_resources",
        )
    with col3:
        if min_d and max_d:
            date_range = st.date_input(
                "Date range",
                [min_d, max_d],
                min_value=min_d,
                max_value=max_d,
                key=f"{prefix}_dates",
            )
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                start, end = date_range
            else:
                start, end = min_d, max_d
        else:
            start, end = None, None

    return sel_regions, sel_resources, start, end


# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class="app-header">
  <div class="app-header-title">Azure Demand & Capacity Console</div>
  <div class="app-header-sub">
    Unified view of usage trends, forecasts, and capacity recommendations across Azure regions.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# LOAD COMMON DATA
# -----------------------------------------------------------------------------
filters_meta = load_filters()
raw_df = load_raw_data()

# -----------------------------------------------------------------------------
# SIDEBAR – NAV BUTTONS
# -----------------------------------------------------------------------------
PAGES = [
    "Overview",
    "Trends",
    "Regional",
    "Resources",
    "User Activity",
    "Forecasting",
    "Capacity Planning",
    "Multi-Region Compare",
    "Alerts",
]

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Overview"

with st.sidebar:
    st.markdown('<div class="sidebar-title">☁ Azure Console</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Demand Forecasting & Capacity Planning</div>', unsafe_allow_html=True)

    st.markdown('<div class="nav-section-label">Pages</div>', unsafe_allow_html=True)
    for page_name in PAGES:
        if st.button(f"{page_name}", key=f"nav_{page_name}", use_container_width=True):
            st.session_state["current_page"] = page_name

current_page = st.session_state["current_page"]

# -----------------------------------------------------------------------------
# PAGE: OVERVIEW
# -----------------------------------------------------------------------------
if current_page == "Overview":
    st.subheader("📊 Overview & KPIs")

    kpi_data = api_get("kpis")
    spark_data = api_get("sparklines")

    if kpi_data:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            peak_cpu = float(kpi_data.get("peak_cpu") or 0.0)
            avg_cpu = float(kpi_data.get("avg_cpu") or 0.0)
            st.markdown(
                f"""
                <div class="card kpi">
                  <div class="kpi-label">Peak CPU</div>
                  <div class="kpi-value">{peak_cpu:.1f}%</div>
                  <div class="kpi-sub">Average {avg_cpu:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            max_storage = float(kpi_data.get("max_storage") or 0.0)
            avg_storage = float(kpi_data.get("avg_storage") or 0.0)
            st.markdown(
                f"""
                <div class="card kpi">
                  <div class="kpi-label">Max Storage</div>
                  <div class="kpi-value">{max_storage:,.0f} GB</div>
                  <div class="kpi-sub">Average {avg_storage:,.0f} GB</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            peak_users = int(kpi_data.get("peak_users") or 0)
            avg_users = float(kpi_data.get("avg_users") or 0.0)
            st.markdown(
                f"""
                <div class="card kpi">
                  <div class="kpi-label">Peak Active Users</div>
                  <div class="kpi-value">{peak_users:,}</div>
                  <div class="kpi-sub">Average {avg_users:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            hi = kpi_data.get("holiday_impact") or {}
            pct = float(hi.get("percentage") or 0.0)
            sign = "+" if pct >= 0 else ""
            st.markdown(
                f"""
                <div class="card kpi">
                  <div class="kpi-label">Holiday Impact</div>
                  <div class="kpi-value">{sign}{pct:.1f}%</div>
                  <div class="kpi-sub">CPU vs working days</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        dr = kpi_data.get("date_range") or {}
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.metric("Regions", kpi_data.get("total_regions", 0))
        with c6:
            st.metric("Resource types", kpi_data.get("total_resource_types", 0))
        with c7:
            st.metric("Data points", f"{kpi_data.get('data_points', 0):,}")
        with c8:
            st.metric("Time span", f"{dr.get('days', 0)} days")

    st.markdown("---")
    st.markdown("##### Recent behaviour (from backend sparklines)")

    if spark_data:
        col1, col2, col3 = st.columns(3)
        if spark_data.get("cpu_trend"):
            cpu_df = pd.DataFrame(spark_data["cpu_trend"])
            if "date" in cpu_df:
                cpu_df["date"] = pd.to_datetime(cpu_df["date"])
                with col1:
                    fig = px.line(cpu_df, x="date", y="usage_cpu", title="CPU (%)")
                    fig.update_layout(height=240, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, width="stretch")
        if spark_data.get("storage_trend"):
            sto_df = pd.DataFrame(spark_data["storage_trend"])
            if "date" in sto_df:
                sto_df["date"] = pd.to_datetime(sto_df["date"])
                with col2:
                    fig = px.line(sto_df, x="date", y="usage_storage", title="Storage (GB)")
                    fig.update_layout(height=240, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, width="stretch")
        if spark_data.get("users_trend"):
            usr_df = pd.DataFrame(spark_data["users_trend"])
            if "date" in usr_df:
                usr_df["date"] = pd.to_datetime(usr_df["date"])
                with col3:
                    fig = px.line(usr_df, x="date", y="users_active", title="Active users")
                    fig.update_layout(height=240, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("🗂 Data Explorer")

    if raw_df.empty:
        st.info("No raw data returned from backend.")
    else:
        r_regions, r_resources, r_start, r_end = render_page_filters("ov", raw_df, filters_meta)
        df_view = filter_dataframe(raw_df, r_regions, r_resources, r_start, r_end)
        st.caption(f"Filtered rows: {len(df_view):,}")
        st.dataframe(df_view.sort_values("date", ascending=False), height=320)

        if not df_view.empty:
            csv_bytes = df_view.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download filtered data as CSV",
                data=csv_bytes,
                file_name="azure_filtered_data_overview.csv",
                mime="text/csv",
                key="dl_overview_filtered",
            )

# -----------------------------------------------------------------------------
# PAGE: TRENDS
# -----------------------------------------------------------------------------
elif current_page == "Trends":
    st.subheader("📈 Time-series Trends")

    if raw_df.empty:
        st.info("No data for trends.")
    else:
        tr_regions, tr_resources, tr_start, tr_end = render_page_filters("tr", raw_df, filters_meta)
        df_tr = filter_dataframe(raw_df, tr_regions, tr_resources, tr_start, tr_end)

        if df_tr.empty:
            st.info("No rows for selected filters.")
        else:
            metric_map = {
                "CPU (%)": "usage_cpu",
                "Storage (GB)": "usage_storage",
                "Active users": "users_active",
            }
            c1, c2, c3 = st.columns(3)
            with c1:
                metric_label = st.selectbox("Metric", list(metric_map.keys()), key="trend_metric")
                metric_col = metric_map[metric_label]
            with c2:
                win = st.selectbox("Window", ["All", "Last 7 days", "Last 30 days", "Last 90 days"], key="trend_window")
            with c3:
                smooth = st.checkbox("7-day smoothing", value=True, key="trend_smoothing")

            series = (
                df_tr.groupby("date")[metric_col]
                .mean()
                .reset_index()
                .sort_values("date")
            )

            if win != "All" and not series.empty:
                days = int(win.split()[1])
                series = series.tail(days)

            if series.empty:
                st.warning("No data after applying window filter.")
            else:
                if smooth and len(series) > 7:
                    series["smoothed"] = series[metric_col].rolling(window=7, min_periods=1).mean()
                else:
                    series["smoothed"] = series[metric_col]

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=series["date"],
                        y=series[metric_col],
                        mode="lines",
                        name="Raw",
                        line=dict(width=1, color="#6b7280"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=series["date"],
                        y=series["smoothed"],
                        mode="lines",
                        name="Smoothed",
                        line=dict(width=3, color="#22c55e"),
                    )
                )
                fig.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), yaxis_title=metric_label)
                st.plotly_chart(fig, width="stretch")

                csv_trend = series[["date", metric_col, "smoothed"]].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download trend CSV",
                    data=csv_trend,
                    file_name=f"trend_{metric_col}.csv",
                    mime="text/csv",
                    key="dl_trend",
                )

# -----------------------------------------------------------------------------
# PAGE: REGIONAL
# -----------------------------------------------------------------------------
elif current_page == "Regional":
    st.subheader("🌍 Regional Comparison")

    if raw_df.empty:
        st.info("No data for regional view.")
    else:
        reg_regions, reg_resources, reg_start, reg_end = render_page_filters("reg", raw_df, filters_meta)
        df_reg = filter_dataframe(raw_df, reg_regions, reg_resources, reg_start, reg_end)

        if df_reg.empty:
            st.info("No data for selected regions/resources.")
        else:
            metric_map = {
                "CPU (%)": "usage_cpu",
                "Storage (GB)": "usage_storage",
                "Active users": "users_active",
            }
            metric_label = st.selectbox("Metric", list(metric_map.keys()), key="regional_metric")
            metric_col = metric_map[metric_label]

            reg_stats = (
                df_reg.groupby("region")[metric_col]
                .agg(["mean", "max", "std", "count"])
                .reset_index()
                .rename(columns={"mean": "avg", "max": "peak", "std": "volatility", "count": "points"})
            )

            if reg_stats.empty:
                st.info("No regional stats.")
            else:
                fig = px.bar(
                    reg_stats.sort_values("avg", ascending=False),
                    x="region",
                    y="avg",
                    color="avg",
                    color_continuous_scale="viridis",
                    labels={"avg": f"Average {metric_label}", "region": "Region"},
                )
                fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), xaxis_tickangle=-35)
                st.plotly_chart(fig, width="stretch")

                with st.expander("Show regional table"):
                    st.dataframe(reg_stats, height=260)

# -----------------------------------------------------------------------------
# PAGE: RESOURCES
# -----------------------------------------------------------------------------
elif current_page == "Resources":
    st.subheader("⚙️ Resource Type Utilization")

    if raw_df.empty:
        st.info("No data for resource analysis.")
    else:
        res_regions, res_resources, res_start, res_end = render_page_filters("res", raw_df, filters_meta)
        df_res = filter_dataframe(raw_df, res_regions, res_resources, res_start, res_end)

        if df_res.empty:
            st.info("No data for selected filters.")
        else:
            res_stats = (
                df_res.groupby("resource_type")
                .agg(
                    avg_cpu=("usage_cpu", "mean"),
                    avg_storage=("usage_storage", "mean"),
                    avg_users=("users_active", "mean"),
                )
                .reset_index()
            )
            if res_stats.empty:
                st.info("No resource stats.")
            else:
                fig = px.bar(
                    res_stats,
                    x="resource_type",
                    y=["avg_cpu", "avg_storage", "avg_users"],
                    barmode="group",
                    labels={"value": "Average", "resource_type": "Resource"},
                )
                fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, width="stretch")

                csv_res = res_stats.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download resource stats CSV",
                    data=csv_res,
                    file_name="resource_stats.csv",
                    mime="text/csv",
                    key="dl_resources",
                )

# -----------------------------------------------------------------------------
# PAGE: USER ACTIVITY
# -----------------------------------------------------------------------------
elif current_page == "User Activity":
    st.subheader("👥 User Activity")

    if raw_df.empty:
        st.info("No data for user analysis.")
    else:
        ua_regions, ua_resources, ua_start, ua_end = render_page_filters("ua", raw_df, filters_meta)
        df_ua = filter_dataframe(raw_df, ua_regions, ua_resources, ua_start, ua_end)

        if df_ua.empty:
            st.info("No data for selected filters.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.line(df_ua, x="date", y="users_active", color="region", title="Active users over time")
                fig.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, width="stretch")
            with c2:
                fig = px.scatter(
                    df_ua,
                    x="users_active",
                    y="usage_cpu",
                    size="usage_storage",
                    color="region",
                    labels={"users_active": "Active users", "usage_cpu": "CPU (%)"},
                    title="Users vs CPU vs Storage",
                )
                fig.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, width="stretch")

# -----------------------------------------------------------------------------
# PAGE: FORECASTING (Milestone 3 + 4)
# -----------------------------------------------------------------------------
elif current_page == "Forecasting":
    st.subheader("🤖 Forecast Dashboard")

    df = raw_df  # for defaults
    metric_map = {"CPU (%)": "cpu", "Storage (GB)": "storage", "Active users": "users"}

    # Build region/service options from filters or data
    regions_list = filters_meta.get("regions") or (
        sorted(df["region"].dropna().unique()) if "region" in df.columns and not df.empty else []
    )
    service_list = filters_meta.get("resource_types") or (
        sorted(df["resource_type"].dropna().unique()) if "resource_type" in df.columns and not df.empty else []
    )

    default_region = regions_list[0] if regions_list else "East US"
    default_service = service_list[0] if service_list else "Compute"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_label = st.selectbox("Metric", list(metric_map.keys()), key="forecast_metric")
        metric_key = metric_map[metric_label]
    with c2:
        model = st.selectbox("Model", ["best", "arima", "xgboost", "lstm"], key="forecast_model")
    with c3:
        horizon_label = st.selectbox("Horizon", ["7 days", "10 days", "30 days"], key="forecast_horizon")
        horizon = int(horizon_label.split()[0])
    with c4:
        region = st.selectbox("Region", options=regions_list or [default_region], index=0, key="forecast_region")

    service = st.selectbox(
        "Service / Resource type",
        options=service_list or [default_service],
        index=0,
        key="forecast_service",
    )

    if st.button("Get forecast", key="btn_get_forecast"):
        params = {
            "metric": metric_key,
            "model": model,
            "region": region,
            "service": service,
            "horizon": horizon,
        }
        data = api_get("forecast", params)
        if not data:
            st.warning("No forecast returned from backend.")
        else:
            df_f = pd.DataFrame(data)
            if "date" not in df_f or "forecast_value" not in df_f:
                st.error("Unexpected forecast format from backend.")
            else:
                df_f["date"] = pd.to_datetime(df_f["date"])

                # Forecast line with optional actual + CI
                fig = go.Figure()
                if "actual_value" in df_f and df_f["actual_value"].notna().any():
                    fig.add_trace(
                        go.Scatter(
                            x=df_f["date"],
                            y=df_f["actual_value"],
                            mode="lines",
                            name="Actual",
                            line=dict(color="#22c55e", width=2),
                        )
                    )
                fig.add_trace(
                    go.Scatter(
                        x=df_f["date"],
                        y=df_f["forecast_value"],
                        mode="lines",
                        name="Forecast",
                        line=dict(color="#38bdf8", width=2),
                    )
                )
                if "lower_ci" in df_f and "upper_ci" in df_f:
                    fig.add_trace(
                        go.Scatter(
                            x=pd.concat([df_f["date"], df_f["date"][::-1]]),
                            y=pd.concat([df_f["upper_ci"], df_f["lower_ci"][::-1]]),
                            fill="toself",
                            fillcolor="rgba(56,189,248,0.2)",
                            line=dict(color="rgba(0,0,0,0)"),
                            hoverinfo="skip",
                            name="Confidence band",
                        )
                    )
                fig.update_layout(
                    height=420,
                    margin=dict(l=0, r=0, t=40, b=0),
                    yaxis_title=metric_label,
                    title=f"{metric_label} forecast · {region} · {service} · {model.upper()}",
                )
                st.plotly_chart(fig, width="stretch")

                # Download forecast CSV
                csv_forecast = df_f.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download forecast CSV",
                    data=csv_forecast,
                    file_name=f"forecast_{metric_key}_{model}_{horizon}d.csv",
                    mime="text/csv",
                    key="dl_forecast",
                )

                st.markdown("---")
                st.subheader("📊 Capacity Snapshot & Model Health")

                col_snap1, col_snap2 = st.columns(2)

                # Capacity snapshot (using capacity-planning API for same region/service)
                with col_snap1:
                    st.markdown("**Capacity vs Forecast**")
                    cp_params = {"region": region, "service": service, "horizon": horizon}
                    cp_data = api_get("capacity-planning", cp_params)
                    if cp_data:
                        df_cp = pd.DataFrame(cp_data)
                        if not df_cp.empty:
                            df_cp["label"] = df_cp["region"].astype(str) + " – " + df_cp["service"].astype(str)
                            fig_cp = go.Figure()
                            if "forecast_demand" in df_cp:
                                fig_cp.add_trace(
                                    go.Bar(
                                        x=df_cp["label"],
                                        y=df_cp["forecast_demand"],
                                        name="Forecast demand",
                                    )
                                )
                            if "available_capacity" in df_cp:
                                fig_cp.add_trace(
                                    go.Bar(
                                        x=df_cp["label"],
                                        y=df_cp["available_capacity"],
                                        name="Available capacity",
                                    )
                                )
                            fig_cp.update_layout(
                                barmode="group",
                                height=280,
                                margin=dict(l=0, r=0, t=40, b=0),
                                xaxis_tickangle=-25,
                            )
                            st.plotly_chart(fig_cp, width="stretch")
                        else:
                            st.info("No capacity snapshot for selected inputs.")
                    else:
                        st.info("Capacity-planning API returned no data.")

                # Model monitoring snapshot (no separate page, embedded here)
                with col_snap2:
                    st.markdown("**Model Monitoring (Accuracy)**")
                    mon = api_get("monitoring", {"metric": metric_key})
                    if mon:
                        acc = float(mon.get("current_accuracy", 0.0))
                        drift = float(mon.get("drift", 0.0))
                        last_retrain = mon.get("last_retrain", "—")

                        if acc >= 85:
                            light = "🟢 Stable"
                        elif acc >= 70:
                            light = "🟡 Caution"
                        else:
                            light = "🔴 Retrain needed"

                        st.markdown(
                            f"""
                            <div class="card kpi">
                              <div class="kpi-label">Forecast Health</div>
                              <div class="kpi-value">{light}</div>
                              <div class="kpi-sub">Accuracy: {acc:.1f}%  · Drift: {drift:+.1f}%  · Last retrain: {last_retrain}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        df_mon = pd.DataFrame(mon.get("accuracy_trend") or [])
                        if not df_mon.empty and "date" in df_mon and "value" in df_mon:
                            df_mon["date"] = pd.to_datetime(df_mon["date"])
                            fig_mon = px.line(df_mon, x="date", y="value", labels={"value": "Accuracy (%)"})
                            fig_mon.update_layout(height=220, margin=dict(l=0, r=0, t=30, b=0))
                            st.plotly_chart(fig_mon, width="stretch")
                    else:
                        st.info("Monitoring API not available for this metric.")

# -----------------------------------------------------------------------------
# PAGE: CAPACITY PLANNING (Milestone 4)
# -----------------------------------------------------------------------------
elif current_page == "Capacity Planning":
    st.subheader("🏗 Capacity Planning & Recommendations")

    # Dropdowns for region & service instead of typing
    regions_list = filters_meta.get("regions") or (
        sorted(raw_df["region"].dropna().unique()) if "region" in raw_df.columns and not raw_df.empty else []
    )
    service_list = filters_meta.get("resource_types") or (
        sorted(raw_df["resource_type"].dropna().unique()) if "resource_type" in raw_df.columns and not raw_df.empty else []
    )

    default_region_label = "All regions"
    region_options = [default_region_label] + regions_list if regions_list else [default_region_label]

    c1, c2, c3 = st.columns(3)
    with c1:
        region_cp_label = st.selectbox("Region", options=region_options, key="cap_region")
        region_cp = None if region_cp_label == default_region_label else region_cp_label
    with c2:
        if service_list:
            service_cp = st.selectbox("Service / Resource type", options=service_list, key="cap_service")
        else:
            service_cp = st.selectbox(
                "Service / Resource type",
                options=["Compute", "Storage"],
                key="cap_service_fallback",
            )
    with c3:
        horizon_cp = st.selectbox("Horizon (days)", [7, 10, 30], index=2, key="cap_horizon")

    params_cp = {
        "region": region_cp,
        "service": service_cp,
        "horizon": horizon_cp,
    }
    data_cp = api_get("capacity-planning", params_cp)

    if not data_cp:
        st.info(
            "Capacity planning data is not available from backend yet. "
            "Ask backend team to expose `/api/capacity-planning` with forecast_demand & available_capacity."
        )
    else:
        df_cp = pd.DataFrame(data_cp)
        if df_cp.empty:
            st.info("Capacity planning endpoint returned no rows for these filters.")
        else:
            df_cp["label"] = df_cp["region"].astype(str) + " – " + df_cp["service"].astype(str)

            st.markdown("### 📊 Capacity vs Forecast")

            fig = go.Figure()
            if "forecast_demand" in df_cp:
                fig.add_trace(
                    go.Bar(
                        x=df_cp["label"],
                        y=df_cp["forecast_demand"],
                        name="Forecast demand",
                    )
                )
            if "available_capacity" in df_cp:
                fig.add_trace(
                    go.Bar(
                        x=df_cp["label"],
                        y=df_cp["available_capacity"],
                        name="Available capacity",
                    )
                )
            fig.update_layout(
                barmode="group",
                height=420,
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis_tickangle=-25,
            )
            st.plotly_chart(fig, width="stretch")

            st.markdown("### 📌 Recommendations Panel")

            for _, row in df_cp.iterrows():
                risk = row.get("risk_level", "unknown")
                icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(risk, "⚪")
                rec = row.get("recommended_adjustment", "0")
                region_txt = row.get("region", "?")
                service_txt = row.get("service", "?")
                st.write(
                    f"{icon} **{region_txt} – {service_txt}** → adjust capacity by **{rec}**"
                )

            st.markdown("---")
            st.subheader("📄 Capacity Report Download")

            csv_cp = df_cp.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download capacity plan CSV",
                data=csv_cp,
                file_name=f"capacity_plan_{service_cp}_{horizon_cp}d.csv",
                mime="text/csv",
                key="dl_capacity",
            )

# -----------------------------------------------------------------------------
# PAGE: MULTI-REGION CAPACITY COMPARISON (ADD-ON 1)
# -----------------------------------------------------------------------------
elif current_page == "Multi-Region Compare":
    st.subheader("🛰 Multi-Region Capacity Comparison")

    if raw_df.empty:
        st.info("No raw data available for comparison.")
    else:
        regions_all = filters_meta.get("regions") or sorted(raw_df["region"].dropna().unique())
        service_list = filters_meta.get("resource_types") or (
            sorted(raw_df["resource_type"].dropna().unique()) if "resource_type" in raw_df.columns else []
        )

        col_top_1, col_top_2, col_top_3 = st.columns(3)
        with col_top_1:
            selected_regions = st.multiselect(
                "Select regions (2+ recommended)",
                options=regions_all,
                default=regions_all[:2] if len(regions_all) >= 2 else regions_all,
                key="multi_regions",
            )
        with col_top_2:
            service = st.selectbox(
                "Service / Resource type (for forecast)",
                options=service_list or ["Compute", "Storage"],
                key="multi_service",
            )
        with col_top_3:
            horizon_multi = st.selectbox("Forecast horizon", [7, 10, 30], index=0, key="multi_horizon")

        if not selected_regions or len(selected_regions) < 1:
            st.info("Please select at least one region.")
        else:
            df_sel = raw_df[raw_df["region"].isin(selected_regions)].copy()
            if df_sel.empty:
                st.info("No data available for the selected regions.")
            else:
                # ------- Radar chart for average metrics -------
                st.markdown("### 🕸 Radar Chart – Average Metrics per Region")

                radar_stats = (
                    df_sel.groupby("region")
                    .agg(
                        avg_cpu=("usage_cpu", "mean"),
                        avg_storage=("usage_storage", "mean"),
                        avg_users=("users_active", "mean"),
                    )
                    .reset_index()
                )

                if len(radar_stats) >= 1:
                    categories = ["CPU", "Storage", "Users"]
                    fig_radar = go.Figure()
                    for _, row in radar_stats.iterrows():
                        fig_radar.add_trace(
                            go.Scatterpolar(
                                r=[row["avg_cpu"], row["avg_storage"], row["avg_users"]],
                                theta=categories,
                                fill="toself",
                                name=row["region"],
                            )
                        )
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True)),
                        showlegend=True,
                        height=430,
                        margin=dict(l=0, r=0, t=40, b=0),
                    )
                    st.plotly_chart(fig_radar, width="stretch")
                else:
                    st.info("Not enough data for radar chart.")

                st.markdown("---")
                st.markdown("### 📈 Region-wise Time Series")

                metric_map = {
                    "CPU (%)": "usage_cpu",
                    "Storage (GB)": "usage_storage",
                    "Active users": "users_active",
                }
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    metric_label_multi = st.selectbox(
                        "Metric", list(metric_map.keys()), key="multi_metric"
                    )
                    metric_col_multi = metric_map[metric_label_multi]
                with col_m2:
                    window_multi = st.selectbox(
                        "Time window",
                        ["All", "Last 30 days", "Last 90 days"],
                        key="multi_window",
                    )

                ts = (
                    df_sel.groupby(["date", "region"])[metric_col_multi]
                    .mean()
                    .reset_index()
                    .sort_values("date")
                )

                if window_multi != "All" and not ts.empty:
                    days = int(window_multi.split()[1])
                    ts = ts[ts["date"] >= ts["date"].max() - pd.Timedelta(days=days)]

                if ts.empty:
                    st.info("No time-series data after filtering.")
                else:
                    fig_ts = px.line(
                        ts,
                        x="date",
                        y=metric_col_multi,
                        color="region",
                        title=f"{metric_label_multi} across regions",
                    )
                    fig_ts.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_ts, width="stretch")

                st.markdown("---")
                st.markdown("### 🕒 Peak Day & Simple Recommendations")

                peak_rows: List[Dict[str, Any]] = []
                for r in selected_regions:
                    df_r = df_sel[df_sel["region"] == r]
                    if df_r.empty:
                        continue
                    # Peak "hour" is not available (daily data), so we use peak DATE.
                    idx = df_r["usage_cpu"].idxmax()
                    row = df_r.loc[idx]
                    avg_cpu_r = df_r["usage_cpu"].mean()
                    peak_cpu_r = row["usage_cpu"]
                    ratio = peak_cpu_r / avg_cpu_r if avg_cpu_r > 0 else 0
                    if ratio > 1.3:
                        rec = "⚠ Add headroom during peak days."
                    elif ratio < 1.05:
                        rec = "✅ Stable usage profile."
                    else:
                        rec = "🟡 Moderate spikes, monitor capacity."

                    peak_rows.append(
                        {
                            "Region": r,
                            "Peak day": row["date"].date(),
                            "Peak CPU (%)": round(peak_cpu_r, 1),
                            "Average CPU (%)": round(avg_cpu_r, 1),
                            "Recommendation": rec,
                        }
                    )

                if peak_rows:
                    df_peak = pd.DataFrame(peak_rows)
                    st.dataframe(df_peak, height=260)
                else:
                    st.info("No peak-day information computed.")

                # Forecast comparison (optional, multi-region with one metric)
                st.markdown("---")
                st.markdown("### 🔮 Multi-region Forecast (same service & metric)")

                metric_map_fore = {"CPU (%)": "cpu", "Storage (GB)": "storage", "Active users": "users"}
                met_label_f = st.selectbox(
                    "Forecast metric",
                    list(metric_map_fore.keys()),
                    index=0,
                    key="multi_fore_metric",
                )
                met_key_f = metric_map_fore[met_label_f]

                if st.button("Load multi-region forecast", key="btn_multi_forecast"):
                    fig_multi_fore = go.Figure()
                    any_ok = False
                    for r in selected_regions:
                        params = {
                            "metric": met_key_f,
                            "model": "best",
                            "region": r,
                            "service": service,
                            "horizon": horizon_multi,
                        }
                        data_fore = api_get("forecast", params)
                        if not data_fore:
                            continue
                        df_fore_r = pd.DataFrame(data_fore)
                        if "date" not in df_fore_r or "forecast_value" not in df_fore_r:
                            continue
                        df_fore_r["date"] = pd.to_datetime(df_fore_r["date"])
                        fig_multi_fore.add_trace(
                            go.Scatter(
                                x=df_fore_r["date"],
                                y=df_fore_r["forecast_value"],
                                mode="lines",
                                name=r,
                            )
                        )
                        any_ok = True
                    if any_ok:
                        fig_multi_fore.update_layout(
                            height=420,
                            margin=dict(l=0, r=0, t=40, b=0),
                            yaxis_title=met_label_f,
                            title=f"{met_label_f} forecast comparison · service={service}",
                        )
                        st.plotly_chart(fig_multi_fore, width="stretch")
                    else:
                        st.info("No forecast data returned for selected regions.")

# -----------------------------------------------------------------------------
# PAGE: PREDICTION ALERTS (ADD-ON 2)
# -----------------------------------------------------------------------------
elif current_page == "Alerts":
    st.subheader("🔔 Prediction Alerts")

    st.markdown(
        """
<div class="info-strip">
Configure thresholds and see which regions/services are at risk in the next 7 days.
</div>
""",
        unsafe_allow_html=True,
    )

    if raw_df.empty:
        st.info("No data available yet to generate alerts.")
    else:
        metric_map_alert = {"CPU (%)": "cpu", "Storage (GB)": "storage", "Active users": "users"}

        regions_list = filters_meta.get("regions") or (
            sorted(raw_df["region"].dropna().unique()) if "region" in raw_df.columns else []
        )
        service_list = filters_meta.get("resource_types") or (
            sorted(raw_df["resource_type"].dropna().unique()) if "resource_type" in raw_df.columns else []
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_label_a = st.selectbox("Alert metric", list(metric_map_alert.keys()), key="alert_metric")
            metric_key_a = metric_map_alert[metric_label_a]
        with c2:
            region_opt = ["All regions"] + regions_list if regions_list else ["All regions"]
            region_sel = st.selectbox("Region scope", options=region_opt, key="alert_region")
        with c3:
            service_sel = st.selectbox(
                "Service / Resource type",
                options=service_list or ["Compute", "Storage"],
                key="alert_service",
            )

        # Choose threshold (for CPU we treat as %, for others raw units)
        if metric_key_a == "cpu":
            default_thresh = 80.0
            max_val = 100.0
        elif metric_key_a == "storage":
            default_thresh = float(raw_df["usage_storage"].quantile(0.8)) if "usage_storage" in raw_df else 1000.0
            max_val = default_thresh * 2
        else:
            default_thresh = float(raw_df["users_active"].quantile(0.8)) if "users_active" in raw_df else 1000.0
            max_val = default_thresh * 2

        threshold = st.slider(
            "Alert threshold",
            min_value=0.0,
            max_value=float(max_val),
            value=float(default_thresh),
            step=float(max_val / 50) if max_val > 0 else 1.0,
            key="alert_threshold",
        )

        if st.button("Check alerts for next 7 days", key="btn_check_alerts"):
            regions_to_check = regions_list if region_sel == "All regions" else [region_sel]
            alert_rows = []

            for r in regions_to_check:
                params = {
                    "metric": metric_key_a,
                    "model": "best",
                    "region": r,
                    "service": service_sel,
                    "horizon": 7,
                }
                data_fore = api_get("forecast", params)
                if not data_fore:
                    continue
                df_fore = pd.DataFrame(data_fore)
                if "forecast_value" not in df_fore or "date" not in df_fore:
                    continue
                df_fore["date"] = pd.to_datetime(df_fore["date"])
                max_pred = df_fore["forecast_value"].max()
                max_row = df_fore.loc[df_fore["forecast_value"].idxmax()]
                d_str = max_row["date"].date()

                if max_pred >= threshold * 1.1:
                    status = "⚠ High risk"
                elif max_pred >= threshold:
                    status = "🟡 Near limit"
                else:
                    status = "🟢 Safe"

                alert_rows.append(
                    {
                        "Region": r,
                        "Service": service_sel,
                        "Metric": metric_label_a,
                        "Threshold": round(threshold, 2),
                        "Max forecast (next 7d)": round(float(max_pred), 2),
                        "Peak day": d_str,
                        "Status": status,
                    }
                )

            if not alert_rows:
                st.info("No forecast data available to compute alerts.")
            else:
                df_alerts = pd.DataFrame(alert_rows)
                # Show big banner if any high risk
                if (df_alerts["Status"] == "⚠ High risk").any():
                    st.markdown(
                        """
<div class="warn-strip">
🚨 One or more regions are forecasted to exceed the configured threshold. Consider increasing capacity or balancing load.
</div>
""",
                        unsafe_allow_html=True,
                    )
                elif (df_alerts["Status"] == "🟡 Near limit").any():
                    st.markdown(
                        """
<div class="info-strip">
⚠ Some regions are close to the configured threshold. Monitor closely.
</div>
""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
<div class="info-strip">
✅ All regions appear safe for the next 7 days under the current threshold.
</div>
""",
                        unsafe_allow_html=True,
                    )

                st.dataframe(df_alerts, height=300)

                csv_alerts = df_alerts.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download alerts as CSV",
                    data=csv_alerts,
                    file_name="prediction_alerts.csv",
                    mime="text/csv",
                    key="dl_alerts",
                )

        st.markdown("---")
        st.subheader("📧 Email Alert Subscription (UI only)")

        email = st.text_input("Email address for alerts", key="alert_email")
        freq = st.selectbox(
            "Alert frequency",
            ["On high-risk only", "Daily summary", "Weekly summary"],
            key="alert_freq",
        )

        if st.button("Save alert preferences", key="btn_save_alerts"):
            if not email:
                st.warning("Please enter an email address.")
            else:
                # In a full system this would call a backend endpoint that saves the subscription
                st.success(
                    f"Alert preferences saved for {email}."
                )
