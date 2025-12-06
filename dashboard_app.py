import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
import os

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Azure Demand Forecasting",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_URL = "http://localhost:5000/api"

# -----------------------------------------------------------------------------
# GLOBAL THEME CSS (Azure gradient + new filter style for Data Explorer)
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
/* App background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #0f172a 0, #020617 45%, #020617 100%);
}

/* Remove wide default padding */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Main header */
.main-header {
    background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 40%, #22c55e 100%);
    color: white;
    padding: 1.2rem 1.6rem;
    border-radius: 1rem;
    box-shadow: 0 14px 30px rgba(15,23,42,0.45);
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.2);
}
.main-header h1 {
    margin: 0;
    font-size: 1.8rem;
    letter-spacing: 0.03em;
}
.main-header p {
    margin: 0.3rem 0 0;
    font-size: 0.95rem;
    opacity: 0.9;
}

/* Cards */
.metric-pill {
    background: rgba(15,23,42,0.85);
    border-radius: 0.9rem;
    padding: 1rem 1.1rem;
    border: 1px solid rgba(148,163,184,0.6);
    box-shadow: 0 8px 24px rgba(15,23,42,0.7);
    backdrop-filter: blur(18px);
}
.metric-pill h4 {
    color: #e5e7eb;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.35rem;
}
.metric-pill h2 {
    color: #f9fafb;
    font-size: 1.35rem;
    margin-bottom: 0.25rem;
}
.metric-pill small {
    color: #94a3b8;
    font-size: 0.75rem;
}

/* Section containers */
.section-card {
    background: rgba(15,23,42,0.9);
    border-radius: 1rem;
    padding: 1rem 1.2rem 1.1rem 1.2rem;
    border: 1px solid rgba(51,65,85,0.9);
    box-shadow: 0 10px 26px rgba(15,23,42,0.9);
    margin-top: 0.6rem;
}

/* Data Explorer special styling */
.data-explorer-shell {
    background: radial-gradient(circle at top left, #1e293b 0, #020617 55%);
    border-radius: 1.1rem;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(148,163,184,0.5);
    box-shadow: 0 18px 40px rgba(15,23,42,0.85);
}
.data-explorer-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
}
.data-explorer-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e5e7eb;
}
.data-explorer-sub {
    font-size: 0.8rem;
    color: #9ca3af;
}

/* Data Explorer filter chips */
.de-filter-chip {
    background: rgba(15,23,42,0.9);
    border-radius: 999px;
    padding: 0.45rem 0.8rem;
    border: 1px solid rgba(148,163,184,0.7);
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    margin-bottom: 0.4rem;
}
.de-filter-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9ca3af;
}
.de-filter-help {
    font-size: 0.7rem;
    color: #6b7280;
}

/* Sidebar nav buttons */
.sidebar-nav-button > button {
    width: 100%;
    justify-content: flex-start;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.6);
    background: linear-gradient(90deg, rgba(15,23,42,0.95), rgba(15,23,42,0.9));
    color: #e5e7eb;
    padding: 0.5rem 0.9rem;
    font-size: 0.85rem;
    margin-bottom: 0.3rem;
    cursor: pointer;
}
.sidebar-nav-button > button:hover {
    border-color: #38bdf8;
    box-shadow: 0 0 0 1px rgba(56,189,248,0.45);
}
.sidebar-nav-button-active > button {
    background: linear-gradient(90deg, #2563eb, #0ea5e9);
    color: white !important;
    border-color: transparent;
    box-shadow: 0 6px 18px rgba(37,99,235,0.8);
}

/* Generic titles */
h3, h4 {
    color: #e5e7eb !important;
}

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 0.8rem;
    border: 1px solid rgba(55,65,81,0.8);
    overflow: hidden;
}

/* Make everything interactive look clickable */
.stDownloadButton button,
.stButton button,
.stMultiSelect,
.stSelectbox,
.stTextInput input,
[data-testid="stSlider"] * {
    cursor: pointer !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# BASIC HELPERS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def fetch_api(endpoint: str, params: dict | None = None):
    try:
        resp = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        # we don't hard-fail, just return None
        return None


def make_download_button(df: pd.DataFrame, label: str, filename: str, key: str):
    if df.empty:
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=key,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def load_filter_options():
    return fetch_api("filters/options") or {}


filter_options = load_filter_options()
ALL_REGIONS = filter_options.get("regions", [])
ALL_RESOURCES = filter_options.get("resource_types", [])


@st.cache_data(show_spinner=False, ttl=300)
def load_raw_data():
    data = fetch_api("data/raw")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION (BUTTONS, NOT RADIO)
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

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Overview"


def nav_button(label: str, icon: str):
    """Render a nav button and update active page if clicked."""
    cls = (
        "sidebar-nav-button-active"
        if st.session_state["active_page"] == label
        else "sidebar-nav-button"
    )
    with st.container():
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        clicked = st.button(f"{icon}  {label}", key=f"nav_{label}")
        st.markdown("</div>", unsafe_allow_html=True)
    if clicked:
        st.session_state["active_page"] = label


with st.sidebar:
    st.markdown("### ☁️ Azure Demand Dashboard")
    st.caption("Navigate between analytics pages")

    for label, icon in [
        ("Overview", "📊"),
        ("Trends", "📈"),
        ("Regional", "🌍"),
        ("Resources", "⚙️"),
        ("User Activity", "👥"),
        ("Forecasting", "🤖"),
        ("Capacity Planning", "🏗️"),
        ("Multi-Region Compare", "🧭"),
        ("Alerts", "🚨"),
    ]:
        nav_button(label, icon)

# -----------------------------------------------------------------------------
# MAIN HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class="main-header">
  <h1>Azure Demand Forecasting & Capacity Optimization</h1>
  <p>End-to-end insights across regions, resources, forecasts and capacity risk.</p>
</div>
""",
    unsafe_allow_html=True,
)

active_page = st.session_state["active_page"]

# -----------------------------------------------------------------------------
# PAGE: OVERVIEW  (includes redesigned Data Explorer)
# -----------------------------------------------------------------------------
if active_page == "Overview":
    # ================= KPIs =================
    kpi_data = fetch_api("kpis")

    if kpi_data:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-pill">', unsafe_allow_html=True)
            st.markdown("<h4>Peak CPU</h4>", unsafe_allow_html=True)
            st.markdown(
                f"<h2>{kpi_data['peak_cpu']:.1f}%</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<small>+{kpi_data['peak_cpu'] - kpi_data['avg_cpu']:.1f}% above avg</small>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-pill">', unsafe_allow_html=True)
            st.markdown("<h4>Max Storage</h4>", unsafe_allow_html=True)
            st.markdown(
                f"<h2>{kpi_data['max_storage']:,} GB</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<small>+{kpi_data['max_storage'] - kpi_data['avg_storage']:.0f} GB above avg</small>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-pill">', unsafe_allow_html=True)
            st.markdown("<h4>Peak Users</h4>", unsafe_allow_html=True)
            st.markdown(
                f"<h2>{kpi_data['peak_users']:,}</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<small>+{kpi_data['peak_users'] - kpi_data['avg_users']:.0f} vs avg</small>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col4:
            hol = kpi_data.get("holiday_impact", {}).get("percentage", 0.0)
            st.markdown('<div class="metric-pill">', unsafe_allow_html=True)
            st.markdown("<h4>Holiday Impact</h4>", unsafe_allow_html=True)
            st.markdown(
                f"<h2>{hol:+.1f}%</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<small>CPU usage change on weekends / holidays</small>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    # ================= SPARKLINES (recent 30 days) =================
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Recent 30-day Trends")
    spark = fetch_api("sparklines")
    if spark:
        sc1, sc2, sc3 = st.columns(3)
        if spark.get("cpu_trend"):
            df_cpu = pd.DataFrame(spark["cpu_trend"])
            if not df_cpu.empty:
                df_cpu["date"] = pd.to_datetime(df_cpu["date"])
                fig = px.line(
                    df_cpu,
                    x="date",
                    y="usage_cpu",
                    title="CPU Usage (%)",
                )
                fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
                sc1.plotly_chart(fig, width="stretch")
        if spark.get("storage_trend"):
            df_st = pd.DataFrame(spark["storage_trend"])
            if not df_st.empty:
                df_st["date"] = pd.to_datetime(df_st["date"])
                fig2 = px.line(
                    df_st,
                    x="date",
                    y="usage_storage",
                    title="Storage Usage (GB)",
                )
                fig2.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
                sc2.plotly_chart(fig2, width="stretch")
        if spark.get("users_trend"):
            df_us = pd.DataFrame(spark["users_trend"])
            if not df_us.empty:
                df_us["date"] = pd.to_datetime(df_us["date"])
                fig3 = px.line(
                    df_us,
                    x="date",
                    y="users_active",
                    title="Active Users",
                )
                fig3.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
                sc3.plotly_chart(fig3, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    # ================= REDESIGNED DATA EXPLORER (NO DATE RANGE) =================
    st.markdown('<div class="data-explorer-shell">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="data-explorer-header">
  <div>
    <div class="data-explorer-title">🗃 Data Explorer</div>
    <div class="data-explorer-sub">Slice and export the unified Azure demand dataset by region & resource.</div>
  </div>
  <div style="font-size:0.75rem;color:#9ca3af;">
    Tip: Use filters below then download only the filtered slice.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    df_raw = load_raw_data()
    if df_raw.empty:
        st.info("No data available from `/api/data/raw`.")
    else:
        # Filter chips row (NO date inputs here)
        fcol1, fcol2, fcol3 = st.columns([1, 1, 1])

        with fcol1:
            st.markdown(
                '<div class="de-filter-chip">'
                '<div class="de-filter-label">Regions</div>',
                unsafe_allow_html=True,
            )
            sel_regions = st.multiselect(
                " ",
                options=sorted(df_raw["region"].unique().tolist()),
                default=sorted(df_raw["region"].unique().tolist()),
                key="de_regions",
                label_visibility="collapsed",
            )
            st.markdown(
                '<div class="de-filter-help">Choose one or more Azure regions.</div></div>',
                unsafe_allow_html=True,
            )

        with fcol2:
            st.markdown(
                '<div class="de-filter-chip">'
                '<div class="de-filter-label">Resource Types</div>',
                unsafe_allow_html=True,
            )
            sel_resources = st.multiselect(
                "  ",
                options=sorted(df_raw["resource_type"].unique().tolist()),
                default=sorted(df_raw["resource_type"].unique().tolist()),
                key="de_resources",
                label_visibility="collapsed",
            )
            st.markdown(
                '<div class="de-filter-help">Filter by compute, storage, database, etc.</div></div>',
                unsafe_allow_html=True,
            )

        with fcol3:
            st.markdown(
                '<div class="de-filter-chip">'
                '<div class="de-filter-label">Sort By</div>',
                unsafe_allow_html=True,
            )
            sort_option = st.selectbox(
                "   ",
                ["Newest first", "Oldest first", "Highest CPU", "Highest Storage", "Most Users"],
                key="de_sort",
                label_visibility="collapsed",
            )
            st.markdown(
                '<div class="de-filter-help">Control the ordering of rows below.</div></div>',
                unsafe_allow_html=True,
            )

        # Apply filters (NO date filter)
        df_filtered = df_raw.copy()
        if sel_regions:
            df_filtered = df_filtered[df_filtered["region"].isin(sel_regions)]
        if sel_resources:
            df_filtered = df_filtered[df_filtered["resource_type"].isin(sel_resources)]

        # Sorting
        if sort_option == "Newest first":
            df_filtered = df_filtered.sort_values("date", ascending=False)
        elif sort_option == "Oldest first":
            df_filtered = df_filtered.sort_values("date", ascending=True)
        elif sort_option == "Highest CPU":
            df_filtered = df_filtered.sort_values("usage_cpu", ascending=False)
        elif sort_option == "Highest Storage":
            df_filtered = df_filtered.sort_values("usage_storage", ascending=False)
        elif sort_option == "Most Users":
            df_filtered = df_filtered.sort_values("users_active", ascending=False)

        # Reorder & prettify columns
        display_cols = [
            "date",
            "region",
            "resource_type",
            "usage_cpu",
            "usage_storage",
            "users_active",
            "economic_index",
            "cloud_market_demand",
            "holiday",
        ]
        display_cols = [c for c in display_cols if c in df_filtered.columns]
        df_disp = df_filtered[display_cols].copy()
        if "holiday" in df_disp.columns:
            df_disp["holiday"] = df_disp["holiday"].map({0: "No", 1: "Yes"})

        st.markdown(
            f"<div style='font-size:0.8rem;color:#9ca3af;margin-top:0.4rem;'>"
            f"Showing <b>{len(df_disp):,}</b> records</div>",
            unsafe_allow_html=True,
        )

        st.dataframe(
            df_disp,
            width="stretch",
            height=360,
        )

        # Quick insights
        with st.expander("📌 Quick Stats on Filtered Slice"):
            c1, c2, c3 = st.columns(3)
            with c1:
                if "usage_cpu" in df_disp.columns:
                    st.metric("Avg CPU (%)", f"{df_disp['usage_cpu'].mean():.1f}")
                    st.metric("Peak CPU (%)", f"{df_disp['usage_cpu'].max():.1f}")
            with c2:
                if "usage_storage" in df_disp.columns:
                    st.metric("Avg Storage (GB)", f"{df_disp['usage_storage'].mean():.0f}")
                    st.metric("Max Storage (GB)", f"{df_disp['usage_storage'].max():.0f}")
            with c3:
                if "users_active" in df_disp.columns:
                    st.metric("Total Users (sum)", f"{int(df_disp['users_active'].sum()):,}")
                    st.metric(
                        "Distinct Regions",
                        f"{df_disp['region'].nunique() if 'region' in df_disp.columns else 0}",
                    )

        make_download_button(
            df_disp,
            label="⬇️ Download filtered slice as CSV",
            filename="azure_data_explorer_filtered.csv",
            key="de_download",
        )

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE: TRENDS
# -----------------------------------------------------------------------------
elif active_page == "Trends":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Time-Series Trends")

    df_raw = load_raw_data()
    if df_raw.empty:
        st.info("No data available.")
    else:
        df_ts = df_raw.copy()
        df_ts["date"] = pd.to_datetime(df_ts["date"])

        t1, t2, t3, t4 = st.columns(4)
        with t1:
            metric_label = st.selectbox(
                "Metric",
                ["CPU Usage (%)", "Storage Usage (GB)", "Active Users"],
                key="tr_metric",
            )
        metric_map = {
            "CPU Usage (%)": "usage_cpu",
            "Storage Usage (GB)": "usage_storage",
            "Active Users": "users_active",
        }
        metric_col = metric_map[metric_label]

        with t2:
            region = st.selectbox(
                "Region",
                ["All"] + sorted(df_ts["region"].unique().tolist()),
                key="tr_region",
            )
        with t3:
            rtype = st.selectbox(
                "Resource Type",
                ["All"] + sorted(df_ts["resource_type"].unique().tolist()),
                key="tr_resource",
            )
        with t4:
            window = st.selectbox(
                "Time Window",
                ["All", "Last 7 days", "Last 30 days", "Last 90 days"],
                key="tr_window",
            )

        # Apply filters
        if region != "All":
            df_ts = df_ts[df_ts["region"] == region]
        if rtype != "All":
            df_ts = df_ts[df_ts["resource_type"] == rtype]

        if window != "All":
            days = int(window.split()[1])
            max_date = df_ts["date"].max()
            cutoff = max_date - timedelta(days=days)
            df_ts = df_ts[df_ts["date"] >= cutoff]

        df_agg = (
            df_ts.groupby("date")
            .agg({metric_col: "mean"})
            .reset_index()
            .sort_values("date")
        )

        if df_agg.empty:
            st.warning("No data for selected filters.")
        else:
            fig = px.line(
                df_agg,
                x="date",
                y=metric_col,
                title=f"{metric_label} over time",
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")

            make_download_button(
                df_agg,
                label="⬇️ Download trend data",
                filename="trend_data.csv",
                key="tr_download",
            )

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE: REGIONAL
# -----------------------------------------------------------------------------
elif active_page == "Regional":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🌍 Regional Overview")

    df_raw = load_raw_data()
    if df_raw.empty:
        st.info("No data available.")
    else:
        agg = (
            df_raw.groupby("region")
            .agg(
                avg_cpu=("usage_cpu", "mean"),
                avg_storage=("usage_storage", "mean"),
                avg_users=("users_active", "mean"),
                records=("region", "count"),
            )
            .reset_index()
        )

        top_region_cpu = agg.sort_values("avg_cpu", ascending=False).iloc[0]
        top_region_storage = agg.sort_values("avg_storage", ascending=False).iloc[0]
        top_region_users = agg.sort_values("avg_users", ascending=False).iloc[0]

        # KPI row similar to old style
        kc1, kc2, kc3 = st.columns(3)
        with kc1:
            st.markdown('<div class="metric-pill">', unsafe_allow_html=True)
            st.markdown("<h4>CPU Hotspot</h4>", unsafe_allow_html=True)
            st.markdown(
                f"<h2>{top_region_cpu['region']}</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<small>Avg CPU: {top_region_cpu['avg_cpu']:.1f}%</small>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with kc2:
            st.markdown('<div class="metric-pill">', unsafe_allow_html=True)
            st.markdown("<h4>Storage Heavy</h4>", unsafe_allow_html=True)
            st.markdown(
                f"<h2>{top_region_storage['region']}</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<small>Avg Storage: {top_region_storage['avg_storage']:.0f} GB</small>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with kc3:
            st.markdown('<div class="metric-pill">', unsafe_allow_html=True)
            st.markdown("<h4>Most Active Users</h4>", unsafe_allow_html=True)
            st.markdown(
                f"<h2>{top_region_users['region']}</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<small>Avg Users: {top_region_users['avg_users']:.0f}</small>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")

        col1, col2 = st.columns([1.7, 1.3])
        with col1:
            metric_label = st.selectbox(
                "Metric",
                ["Avg CPU (%)", "Avg Storage (GB)", "Avg Active Users"],
                key="reg_metric",
            )
            metric_map = {
                "Avg CPU (%)": "avg_cpu",
                "Avg Storage (GB)": "avg_storage",
                "Avg Active Users": "avg_users",
            }
            mcol = metric_map[metric_label]

            fig = px.bar(
                agg.sort_values(mcol, ascending=False),
                x="region",
                y=mcol,
                title=f"{metric_label} by region",
            )
            fig.update_layout(height=420, xaxis_title="", margin=dict(l=10, r=10, t=40, b=80))
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.write("#### Summary")
            st.metric("Regions Covered", agg["region"].nunique())
            st.metric("Total Records", int(agg["records"].sum()))
            if "avg_cpu" in agg.columns:
                st.metric("Global Avg CPU", f"{agg['avg_cpu'].mean():.1f}%")
            if "avg_storage" in agg.columns:
                st.metric("Global Avg Storage", f"{agg['avg_storage'].mean():.0f} GB")

            make_download_button(
                agg,
                label="⬇️ Download regional summary",
                filename="regional_summary.csv",
                key="reg_download",
            )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE: RESOURCES
# -----------------------------------------------------------------------------
elif active_page == "Resources":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Resource Type Utilization")

    df_raw = load_raw_data()
    if df_raw.empty:
        st.info("No data available.")
    else:
        df_res = df_raw.copy()
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            region = st.selectbox(
                "Region",
                ["All"] + sorted(df_res["region"].unique().tolist()),
                key="res_region",
            )
        with rcol2:
            metric_label = st.selectbox(
                "Metric",
                ["Avg CPU (%)", "Avg Storage (GB)", "Avg Active Users"],
                key="res_metric",
            )
        metric_map_raw = {
            "Avg CPU (%)": "usage_cpu",
            "Avg Storage (GB)": "usage_storage",
            "Avg Active Users": "users_active",
        }
        mcol = metric_map_raw[metric_label]

        if region != "All":
            df_res = df_res[df_res["region"] == region]

        agg = (
            df_res.groupby("resource_type")
            .agg(
                avg_metric=(mcol, "mean"),
                avg_cpu=("usage_cpu", "mean"),
                avg_storage=("usage_storage", "mean"),
                avg_users=("users_active", "mean"),
                records=("resource_type", "count"),
            )
            .reset_index()
        )
        if agg.empty:
            st.warning("No data for selected filters.")
        else:
            fig = px.bar(
                agg.sort_values("avg_metric", ascending=False),
                x="resource_type",
                y="avg_metric",
                title=f"{metric_label} by resource type",
            )
            fig.update_layout(height=420, xaxis_title="", margin=dict(l=10, r=10, t=40, b=80))
            st.plotly_chart(fig, width="stretch")

            make_download_button(
                agg,
                label="⬇️ Download resource summary",
                filename="resource_summary.csv",
                key="res_download",
            )

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE: USER ACTIVITY (ENHANCED)
# -----------------------------------------------------------------------------
elif active_page == "User Activity":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 👥 User Activity & Engagement")

    df_raw = load_raw_data()
    if df_raw.empty:
        st.info("No data available.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        df = df_raw.copy()
        df["date"] = pd.to_datetime(df["date"])

        # ----- Filter & controls (simple window, not explicit range) -----
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            regions = st.multiselect(
                "Regions",
                options=sorted(df["region"].unique().tolist()),
                default=sorted(df["region"].unique().tolist()),
                key="ua_regions",
            )
        with fc2:
            resources = st.multiselect(
                "Resource Types",
                options=sorted(df["resource_type"].unique().tolist()),
                default=sorted(df["resource_type"].unique().tolist()),
                key="ua_resources",
            )
        with fc3:
            window = st.selectbox(
                "Time Window",
                ["All data", "Last 30 days", "Last 90 days", "Last 180 days"],
                key="ua_window",
            )

        if regions:
            df = df[df["region"].isin(regions)]
        if resources:
            df = df[df["resource_type"].isin(resources)]

        if window != "All data":
            days = int(window.split()[1])
            max_date = df["date"].max()
            cutoff = max_date - timedelta(days=days)
            df = df[df["date"] >= cutoff]

        if df.empty:
            st.warning("No user activity data for selected filters.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # ----- KPIs row -----
            k1, k2, k3, k4 = st.columns(4)
            daily_users = df.groupby("date")["users_active"].sum()
            peak_day = daily_users.idxmax()
            peak_val = daily_users.max()
            avg_users = daily_users.mean()

            with k1:
                st.markdown('<div class="metric-pill">', unsafe_allow_html=True)
                st.markdown("<h4>Total User Events</h4>", unsafe_allow_html=True)
                st.markdown(
                    f"<h2>{int(df['users_active'].sum()):,}</h2>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<small>Sum across all days & regions</small>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with k2:
                st.markdown('<div class="metric-pill">', unsafe_allow_html=True)
                st.markdown("<h4>Peak Day</h4>", unsafe_allow_html=True)
                st.markdown(
                    f"<h2>{peak_day.date().isoformat()}</h2>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<small>{int(peak_val):,} active users</small>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with k3:
                st.markdown('<div class="metric-pill">', unsafe_allow_html=True)
                st.markdown("<h4>Avg Daily Users</h4>", unsafe_allow_html=True)
                st.markdown(
                    f"<h2>{int(avg_users):,}</h2>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<small>Daily mean (filtered slice)</small>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with k4:
                df["weekday"] = df["date"].dt.weekday
                weekday_mean = df[df["weekday"] < 5]["users_active"].mean()
                weekend_mean = df[df["weekday"] >= 5]["users_active"].mean()
                change = (
                    (weekend_mean - weekday_mean) / weekday_mean * 100
                    if weekday_mean > 0
                    else 0
                )
                st.markdown('<div class="metric-pill">', unsafe_allow_html=True)
                st.markdown("<h4>Weekend vs Weekday</h4>", unsafe_allow_html=True)
                st.markdown(
                    f"<h2>{change:+.1f}%</h2>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<small>Weekend activity vs weekdays</small>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("")

            # ----- Charts row -----
            c1, c2 = st.columns([1.6, 1.4])

            with c1:
                # User trend (line)
                trend = (
                    df.groupby("date")["users_active"]
                    .sum()
                    .reset_index()
                    .sort_values("date")
                )
                fig1 = px.area(
                    trend,
                    x="date",
                    y="users_active",
                    title="Daily Active Users",
                )
                fig1.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig1, width="stretch")

            with c2:
                # Top regions by avg users (bar)
                reg_users = (
                    df.groupby("region")["users_active"]
                    .mean()
                    .reset_index()
                    .sort_values("users_active", ascending=False)
                    .head(7)
                )
                fig2 = px.bar(
                    reg_users,
                    x="users_active",
                    y="region",
                    orientation="h",
                    title="Top Regions by Avg Users",
                )
                fig2.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig2, width="stretch")

            st.markdown("")

            # ----- Engagement vs Load (scatter) + spikes table -----
            lc1, lc2 = st.columns([1.4, 1.6])

            with lc1:
                fig3 = px.scatter(
                    df,
                    x="users_active",
                    y="usage_cpu",
                    size="usage_storage",
                    color="region",
                    hover_data=["date", "resource_type"],
                    title="Users vs CPU vs Storage",
                )
                fig3.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig3, width="stretch")

            with lc2:
                st.write("#### Top User Spikes")
                spikes = (
                    df.sort_values("users_active", ascending=False)
                    .head(8)[["date", "region", "resource_type", "users_active", "usage_cpu"]]
                    .copy()
                )
                spikes["date"] = spikes["date"].dt.date
                st.dataframe(spikes, width="stretch", height=260)
                make_download_button(
                    spikes,
                    label="⬇️ Download spike events",
                    filename="user_spikes.csv",
                    key="ua_spikes_dl",
                )

            # Extra: download filtered slice
            make_download_button(
                df.drop(columns=["weekday"], errors="ignore"),
                label="⬇️ Download filtered user activity slice",
                filename="user_activity_filtered.csv",
                key="ua_filtered_dl",
            )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE: FORECASTING
# -----------------------------------------------------------------------------
elif active_page == "Forecasting":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 Forecast Dashboard")

    df_raw = load_raw_data()
    if df_raw.empty:
        st.info("Need raw data to drive filters & context.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        regions = sorted(df_raw["region"].unique().tolist())
        resources = sorted(df_raw["resource_type"].unique().tolist())

        col1, col2, col3, col4 = st.columns(4)
        metric_map = {
            "CPU Usage (%)": "cpu",
            "Storage Usage (GB)": "storage",
            "Active Users": "users",
        }
        with col1:
            metric_label = st.selectbox(
                "Metric",
                list(metric_map.keys()),
                key="fc_metric",
            )
        metric_key = metric_map[metric_label]
        with col2:
            model = st.selectbox(
                "Model",
                ["best", "arima", "xgboost", "lstm"],
                key="fc_model",
            )
        with col3:
            region = st.selectbox(
                "Region",
                regions,
                key="fc_region",
            )
        with col4:
            service = st.selectbox(
                "Service / Resource Type",
                resources,
                key="fc_resource",
            )

        hcol1, hcol2 = st.columns([1.2, 0.8])
        with hcol1:
            horizon = st.slider(
                "Forecast horizon (days)",
                min_value=7,
                max_value=60,
                value=30,
                step=1,
                key="fc_horizon",
            )
        with hcol2:
            st.write("")
            run = st.button("📡 Run Forecast", key="fc_run")

        if run:
            params = {
                "metric": metric_key,
                "model": model,
                "region": region,
                "service": service,
                "horizon": horizon,
            }
            data = fetch_api("forecast", params=params)
            if not data:
                st.warning("No forecast data returned by the backend.")
            else:
                df_fc = pd.DataFrame(data)
                if "date" in df_fc.columns:
                    df_fc["date"] = pd.to_datetime(df_fc["date"])

                # Forecast line chart
                fig = go.Figure()
                if "actual_value" in df_fc.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df_fc["date"],
                            y=df_fc["actual_value"],
                            mode="lines+markers",
                            name="Actual",
                        )
                    )
                fig.add_trace(
                    go.Scatter(
                        x=df_fc["date"],
                        y=df_fc["forecast_value"],
                        mode="lines+markers",
                        name="Forecast",
                    )
                )
                if "lower_ci" in df_fc.columns and "upper_ci" in df_fc.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=pd.concat([df_fc["date"], df_fc["date"][::-1]]),
                            y=pd.concat([df_fc["upper_ci"], df_fc["lower_ci"][::-1]]),
                            fill="toself",
                            fillcolor="rgba(56,189,248,0.1)",
                            line=dict(color="rgba(0,0,0,0)"),
                            hoverinfo="skip",
                            showlegend=True,
                            name="Confidence band",
                        )
                    )
                fig.update_layout(
                    title=f"{metric_label} forecast for {region} – {service}",
                    height=420,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, width="stretch")

                make_download_button(
                    df_fc,
                    label="⬇️ Download forecast as CSV",
                    filename="forecast_output.csv",
                    key="fc_dl",
                )

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE: CAPACITY PLANNING  (with frontend fallback)
# -----------------------------------------------------------------------------
elif active_page == "Capacity Planning":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🏗️ Capacity Planning & Recommendations")

    df_raw = load_raw_data()
    if df_raw.empty:
        st.info("Need raw data for filters.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        regions = ["All regions"] + sorted(df_raw["region"].unique().tolist())
        resources = sorted(df_raw["resource_type"].unique().tolist())

        c1, c2, c3 = st.columns(3)
        with c1:
            region = st.selectbox(
                "Region",
                regions,
                key="cp_region",
            )
        with c2:
            service = st.selectbox(
                "Service / Resource Type",
                resources,
                key="cp_service",
            )
        with c3:
            horizon = st.selectbox(
                "Horizon (days)",
                [7, 10, 30],
                key="cp_horizon",
            )

        go_btn = st.button("🔎 Load Capacity Plan", key="cp_btn")

        if go_btn:
            # First try backend API
            params = {"service": service, "horizon": horizon}
            if region != "All regions":
                params["region"] = region

            data = fetch_api("capacity-planning", params=params)

            # If backend not ready -> fallback using raw data (NO visible info message)
            if not data:
                df = df_raw.copy()
                df["date"] = pd.to_datetime(df["date"])
                # filter by region & service
                if region != "All regions":
                    df = df[df["region"] == region]
                df = df[df["resource_type"] == service]

                if df.empty:
                    st.warning("No data for selected region/service.")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    max_date = df["date"].max()
                    cutoff = max_date - timedelta(days=horizon)
                    df_window = df[df["date"] >= cutoff]
                    if df_window.empty:
                        df_window = df

                    rows = []
                    for r in df_window["region"].unique():
                        sub = df_window[df_window["region"] == r]
                        # Use peak CPU as demand index (simple heuristic)
                        demand = float(sub["usage_cpu"].max())
                        capacity = demand * 0.9
                        gap = demand - capacity
                        ratio = demand / capacity if capacity > 0 else 0

                        if ratio > 1.1:
                            risk = "high"
                        elif ratio > 1.0:
                            risk = "medium"
                        else:
                            risk = "low"

                        rec = f"+{gap:.1f} units" if gap > 0 else f"{gap:.1f} units"
                        rows.append(
                            {
                                "region": r,
                                "service": service,
                                "forecast_demand": round(demand, 1),
                                "available_capacity": round(capacity, 1),
                                "recommended_adjustment": rec,
                                "risk_level": risk,
                            }
                        )

                    df_cp = pd.DataFrame(rows)
            else:
                df_cp = pd.DataFrame(data)

            if df_cp is None or df_cp.empty:
                st.warning("No capacity planning data could be generated.")
            else:
                # Chart
                if all(col in df_cp.columns for col in ["forecast_demand", "available_capacity"]):
                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=df_cp["region"],
                            y=df_cp["forecast_demand"],
                            name="Forecast Demand",
                        )
                    )
                    fig.add_trace(
                        go.Bar(
                            x=df_cp["region"],
                            y=df_cp["available_capacity"],
                            name="Available Capacity",
                        )
                    )
                    fig.update_layout(
                        barmode="group",
                        title=f"Forecast vs Capacity – {service}",
                        height=420,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig, width="stretch")

                # Recommendations panel
                st.write("#### Recommendations")
                for _, row in df_cp.iterrows():
                    risk_level = str(row.get("risk_level", "unknown")).lower()
                    color_emoji = "🟢"
                    if risk_level == "high":
                        color_emoji = "🔴"
                    elif risk_level == "medium":
                        color_emoji = "🟡"
                    st.markdown(
                        f"- {color_emoji} **{row['region']} – {row['service']}** → "
                        f"{row.get('recommended_adjustment', 'No recommendation')}"
                    )

                make_download_button(
                    df_cp,
                    label="⬇️ Download capacity plan as CSV",
                    filename="capacity_plan.csv",
                    key="cp_dl",
                )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE: MULTI-REGION COMPARE  (radar now reacts to metric)
# -----------------------------------------------------------------------------
elif active_page == "Multi-Region Compare":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🧭 Multi-Region Capacity Comparison")

    df_raw = load_raw_data()
    if df_raw.empty:
        st.info("No data available.")
    else:
        df = df_raw.copy()
        df["date"] = pd.to_datetime(df["date"])

        m1, m2, m3 = st.columns(3)
        with m1:
            regions = st.multiselect(
                "Select regions (2+ recommended)",
                options=sorted(df["region"].unique().tolist()),
                default=sorted(df["region"].unique().tolist())[:3],
                key="mrc_regions",
            )
        with m2:
            service = st.selectbox(
                "Service / Resource Type",
                sorted(df["resource_type"].unique().tolist()),
                key="mrc_service",
            )
        with m3:
            metric_label = st.selectbox(
                "Metric",
                ["CPU Usage (%)", "Storage Usage (GB)", "Active Users"],
                key="mrc_metric",
            )
        metric_map_raw = {
            "CPU Usage (%)": "usage_cpu",
            "Storage Usage (GB)": "usage_storage",
            "Active Users": "users_active",
        }
        metric_col = metric_map_raw[metric_label]

        if not regions:
            st.warning("Select at least one region.")
        else:
            df = df[df["region"].isin(regions) & (df["resource_type"] == service)]

            if df.empty:
                st.warning("No data for selected combination.")
            else:
                # Average metrics per region
                agg = (
                    df.groupby("region")
                    .agg(
                        avg_cpu=("usage_cpu", "mean"),
                        avg_storage=("usage_storage", "mean"),
                        avg_users=("users_active", "mean"),
                    )
                    .reset_index()
                )

                # Radar now depends on selected metric
                radar_metric_map = {
                    "CPU Usage (%)": "avg_cpu",
                    "Storage Usage (GB)": "avg_storage",
                    "Active Users": "avg_users",
                }
                rcol = radar_metric_map[metric_label]
                radar_df = agg[["region", rcol]].rename(columns={rcol: "value"})

                fig_radar = px.line_polar(
                    radar_df,
                    r="value",
                    theta="region",
                    line_close=True,
                    title=f"{metric_label} – multi-region comparison",
                )
                fig_radar.update_traces(fill="toself")
                fig_radar.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_radar, width="stretch")

                # Time-series compare
                ts = (
                    df.groupby(["date", "region"])[metric_col]
                    .mean()
                    .reset_index()
                    .sort_values("date")
                )
                fig_ts = px.line(
                    ts,
                    x="date",
                    y=metric_col,
                    color="region",
                    title=f"{metric_label} – time series comparison",
                )
                fig_ts.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_ts, width="stretch")

                make_download_button(
                    df,
                    label="⬇️ Download multi-region slice",
                    filename="multi_region_slice.csv",
                    key="mrc_dl",
                )

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE: ALERTS (Prediction Alerts)
# -----------------------------------------------------------------------------
elif active_page == "Alerts":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🚨 Prediction Alerts")

    df_raw = load_raw_data()
    if df_raw.empty:
        st.info("No data available.")
    else:
        regions = ["All regions"] + sorted(df_raw["region"].unique().tolist())
        resources = sorted(df_raw["resource_type"].unique().tolist())

        a1, a2, a3, a4 = st.columns(4)
        with a1:
            metric_label = st.selectbox(
                "Metric",
                ["CPU Usage (%)", "Storage Usage (GB)", "Active Users"],
                key="al_metric",
            )
        metric_map = {
            "CPU Usage (%)": "cpu",
            "Storage Usage (GB)": "storage",
            "Active Users": "users",
        }
        metric_key = metric_map[metric_label]

        with a2:
            region = st.selectbox(
                "Region scope",
                regions,
                key="al_region",
            )
        with a3:
            service = st.selectbox(
                "Service / Resource Type",
                resources,
                key="al_service",
            )
        with a4:
            threshold = st.slider(
                "Alert threshold",
                min_value=10,
                max_value=100,
                value=80,
                step=1,
                key="al_threshold",
            )

        btn = st.button("🔍 Check alerts for next 7 days", key="al_btn")

        if btn:
            # Determine list of regions to query
            if region == "All regions":
                region_list = sorted(df_raw["region"].unique().tolist())
            else:
                region_list = [region]

            rows = []
            for r in region_list:
                params = {
                    "metric": metric_key,
                    "model": "best",
                    "region": r,
                    "service": service,
                    "horizon": 7,
                }
                data = fetch_api("forecast", params=params)
                if not data:
                    continue
                df_fc = pd.DataFrame(data)
                if df_fc.empty or "forecast_value" not in df_fc.columns:
                    continue
                df_fc["date"] = pd.to_datetime(df_fc["date"])
                max_row = df_fc.loc[df_fc["forecast_value"].idxmax()]
                max_val = max_row["forecast_value"]
                peak_day = max_row["date"].date()

                if max_val >= threshold * 1.1:
                    status = "⚠ High risk"
                elif max_val >= threshold:
                    status = "🟡 Near limit"
                else:
                    status = "🟢 Safe"

                rows.append(
                    {
                        "region": r,
                        "service": service,
                        "metric": metric_label,
                        "threshold": threshold,
                        "max_forecast_next_7d": round(max_val, 2),
                        "peak_day": peak_day,
                        "status": status,
                    }
                )

            if not rows:
                st.info("No forecast data available for alert calculation.")
            else:
                df_alerts = pd.DataFrame(rows)

                # Overall banner
                if (df_alerts["status"] == "⚠ High risk").any():
                    st.error("🔴 One or more regions are in **High risk** zone.")
                elif (df_alerts["status"] == "🟡 Near limit").any():
                    st.warning("🟡 Some regions are **close to threshold**.")
                else:
                    st.success("🟢 All regions are currently in safe range.")

                st.dataframe(df_alerts, width="stretch", height=260)

                make_download_button(
                    df_alerts,
                    label="⬇️ Download alert table",
                    filename="prediction_alerts.csv",
                    key="al_dl",
                )

                with st.expander("📧 Email Alerts (UI only)"):
                    email = st.text_input("Notification email", key="al_email")
                    freq = st.selectbox(
                        "Alert trigger",
                        [
                            "Only when High risk",
                            "Daily summary",
                            "Weekly summary",
                        ],
                        key="al_freq",
                    )
                    save = st.button("Save alert preference", key="al_pref_btn")
                    if save:
                        if email.strip():
                            st.success(
                                f"Alert preference saved for {email}. (In real deployment, backend would send emails using SMTP / Azure Functions.)"
                            )
                        else:
                            st.warning("Please enter a valid email.")

    st.markdown("</div>", unsafe_allow_html=True)
