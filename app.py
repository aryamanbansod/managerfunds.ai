import streamlit as st
import pandas as pd
import requests
import os
import datetime
import pytz

import backend_engine

# =============================================================================
# CONFIG
# =============================================================================
SIGNALS_CSV = "data/signals/smart_money_tradr_signals.csv"
SYSTEM_TS_FILE = "data/system/last_successful_scan.txt"

# =============================================================================
# SYSTEM STATUS (QUIET, FACTUAL)
# =============================================================================
def render_system_status():
    st.subheader("🛰️ System Status")

    if not os.path.exists(SYSTEM_TS_FILE):
        st.info("System initializing — no scan executed yet.")
        return

    try:
        with open(SYSTEM_TS_FILE, "r") as f:
            ts = f.read().strip()

        last_scan = datetime.datetime.fromisoformat(ts)
        if last_scan.tzinfo is None:
            last_scan = last_scan.replace(tzinfo=datetime.timezone.utc)

        now = datetime.datetime.now(datetime.timezone.utc)
        mins_ago = int((now - last_scan).total_seconds() / 60)

        ist = pytz.timezone("Asia/Kolkata")
        ts_ist = last_scan.astimezone(ist).strftime("%d %b %Y, %H:%M IST")

        if mins_ago <= 30:
            st.success(f"Last scan: {ts_ist}")
        else:
            st.warning(f"Last scan: {ts_ist} (stale — run fresh scan if needed)")

    except Exception:
        st.warning("Unable to read last scan timestamp.")

# =============================================================================
# MANUAL GITHUB SCAN TRIGGER (USED ONLY IN WEEKLY OPPORTUNITIES)
# =============================================================================
def trigger_manual_scan():
    try:
        token = st.secrets["GITHUB_TOKEN"]
        owner = st.secrets["GITHUB_OWNER"]
        repo = st.secrets["GITHUB_REPO"]
        event_type = st.secrets["GITHUB_WORKFLOW_DISPATCH_EVENT"]
    except KeyError:
        st.error("GitHub secrets not configured.")
        return

    url = f"https://api.github.com/repos/{owner}/{repo}/dispatches"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    payload = {
        "event_type": event_type
    }

    r = requests.post(url, headers=headers, json=payload)

    if r.status_code == 204:
        st.success("Scan request sent to GitHub.")
    else:
        st.error(f"Failed to trigger scan (HTTP {r.status_code}).")

# =============================================================================
# PAGE: DASHBOARD
# =============================================================================
def page_dashboard():
    st.title("📊 Dashboard")
    render_system_status()
    backend_engine.render_dashboard()

# =============================================================================
# PAGE: WEEKLY OPPORTUNITIES (SCAN BUTTON LIVES HERE)
# =============================================================================
def page_weekly_opportunities():
    st.title("📘 Weekly Opportunity Book")

    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔁 Run Fresh Scan"):
            trigger_manual_scan()

    if not os.path.exists(SIGNALS_CSV):
        st.info("No actionable setups found.")
        return

    df = pd.read_csv(SIGNALS_CSV)

    if df.empty:
        st.info("No actionable setups found.")
        return

    display_cols = [
        "Ticker", "Setup", "Entry", "StopLoss",
        "Target1", "Target2", "Risk_Pct",
        "Status", "Detected_Date"
    ]

    df = df[[c for c in display_cols if c in df.columns]]
    df = df.sort_values("Detected_Date", ascending=False)

    st.dataframe(df, use_container_width=True)

# =============================================================================
# PAGE: TRADE EXECUTION
# =============================================================================
def page_trade_execution():
    st.title("⚙️ Trade Execution")
    backend_engine.render_trade_execution()

# =============================================================================
# PAGE: PERFORMANCE & ANALYTICS
# =============================================================================
def page_performance():
    st.title("📈 Performance & Analytics")
    backend_engine.render_performance_analytics()

# =============================================================================
# APP ENTRY
# =============================================================================
def main():
    st.set_page_config(layout="wide", page_title="AI Fund Manager")

    with st.sidebar:
        st.header("Navigate")
        page = st.radio(
            "",
            [
                "Dashboard",
                "Weekly Opportunities",
                "Trade Execution",
                "Performance & Analytics"
            ]
        )

    if page == "Dashboard":
        page_dashboard()
    elif page == "Weekly Opportunities":
        page_weekly_opportunities()
    elif page == "Trade Execution":
        page_trade_execution()
    elif page == "Performance & Analytics":
        page_performance()

if __name__ == "__main__":
    main()
