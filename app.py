# ==============================================================================
# 📊 AI FUNDS MANAGER DASHBOARD (OPTION A: DIRECT SIGNALS WIRING)
# ==============================================================================
# ARCHITECTURE:
# 1. DATA SOURCE: Reads directly from 'data/signals/smart_money_tradr_signals.csv'
# 2. STATUS SOURCE: Derived from the modification timestamp of the Signals CSV.
# 3. ACTION: Triggers GitHub Actions via API (No local execution).
# 4. SAFETY: Defensive coding, no crashes if files are missing.
# ==============================================================================

import streamlit as st
import pandas as pd
import os
import datetime
import pytz
import requests

# --- DEFENSIVE BACKEND IMPORT ---
try:
    import backend_engine
    BACKEND_AVAILABLE = True
except ImportError:
    backend_engine = None
    BACKEND_AVAILABLE = False

# ==============================================================================
# 1. CONFIGURATION & PATHS
# ==============================================================================
st.set_page_config(
    page_title="AI Fund Manager",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FIXED PATHS (Option A Configuration)
BASE_DIR = "data"
SIGNALS_DIR = f"{BASE_DIR}/signals"
# We now treat the signals file as the single source of truth for data AND status
MAIN_DATA_FILE = f"{SIGNALS_DIR}/smart_money_tradr_signals.csv"

# ==============================================================================
# 2. SYSTEM STATUS (DERIVED FROM FILE TIMESTAMP)
# ==============================================================================
def get_system_health():
    """
    Determines system health by checking the modification time of the signals CSV.
    Returns: (status_type, message, minutes_ago)
    """
    if not os.path.exists(MAIN_DATA_FILE):
        return "no_data", "System initializing — no scan executed yet.", 0

    try:
        # Get file modification time
        mtime = os.path.getmtime(MAIN_DATA_FILE)
        last_scan_time = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
        
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        diff_minutes = int((now_utc - last_scan_time).total_seconds() / 60)
        
        if diff_minutes <= 20:
            return "healthy", "Scanner healthy — data fresh", diff_minutes
        elif diff_minutes <= 60:
            return "catching_up", "Scanner catching up", diff_minutes
        else:
            return "stale", "Scanner stale — execution delayed", diff_minutes
    except Exception as e:
        return "error", f"Status check error: {str(e)}", 0

def trigger_github_scan():
    """Triggers GitHub Action safely via Secrets. No local execution."""
    try:
        if "GITHUB_TOKEN" not in st.secrets:
            st.error("❌ GitHub Secrets missing. Please configure them in Streamlit.")
            return

        token = st.secrets["GITHUB_TOKEN"]
        owner = st.secrets["GITHUB_OWNER"]
        repo = st.secrets["GITHUB_REPO"]
        event_type = st.secrets.get("GITHUB_WORKFLOW_DISPATCH_EVENT", "manual_scan_triggered")
        
        url = f"https://api.github.com/repos/{owner}/{repo}/dispatches"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        payload = {"event_type": event_type}
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 204:
            st.success("✅ Scan request sent to GitHub. Results will appear shortly.")
        else:
            st.error(f"❌ Failed to trigger. GitHub Code: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Trigger Error: {e}")

def load_data():
    """Safely loads the signals CSV."""
    if not os.path.exists(MAIN_DATA_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(MAIN_DATA_FILE)
    except Exception:
        return pd.DataFrame()

# ==============================================================================
# 3. PAGE RENDERERS
# ==============================================================================

def render_dashboard_page():
    st.title("📡 System Dashboard")
    
    # 1. System Health Status
    status, msg, mins = get_system_health()
    if status == "healthy":
        st.success(f"🟢 **{msg}** ({mins} min ago)")
    elif status == "catching_up":
        st.warning(f"🟡 **{msg}** ({mins} min ago)")
        st.caption("ℹ️ Scheduler delay detected (System self-healing). No action required.")
    elif status == "stale":
        st.error(f"🔴 **{msg}** ({mins} min ago)")
        st.caption("ℹ️ Execution delayed. Check GitHub Actions.")
    else:
        st.info(f"⚪ {msg}")

    st.markdown("---")

    # 2. Defensive Backend Integration
    if BACKEND_AVAILABLE and hasattr(backend_engine, "render_dashboard"):
        backend_engine.render_dashboard()
    else:
        st.info("📊 Detailed analytics view. (Standard dashboard loaded).")
        st.caption("Backend reporting module not detected or not configured.")

def render_weekly_opportunities_page():
    st.title("📖 Weekly Opportunities")
    
    # 1. ACTION BUTTON
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔁 Run Fresh Scan Now", type="primary", use_container_width=True):
            trigger_github_scan()
    
    st.markdown("---")

    # 2. DATA DISPLAY
    df = load_data()
    
    if df.empty:
        st.info("📭 No actionable setups found.")
    else:
        # Sort by Date if available
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values(by='Date', ascending=False)
            except: pass
            
        st.metric("Potential Candidates", len(df))
        
        # Display Table (Dynamic Columns)
        # We try to match preferred columns, but fallback to whatever exists in the CSV
        preferred_cols = [
            'Ticker', 'SETUP_TYPE', 'Close', 'swing_low', 'projected_target', 
            'super_score', 'Date'
        ]
        
        # Filter to ensure we don't crash if a column is missing in the raw signals file
        valid_cols = [c for c in preferred_cols if c in df.columns]
        
        if not valid_cols:
            valid_cols = df.columns.tolist() # Fallback to showing everything
        
        st.dataframe(
            df[valid_cols],
            use_container_width=True,
            hide_index=True
        )

def render_trade_execution_page():
    st.title("⚡ Trade Execution")
    
    if BACKEND_AVAILABLE and hasattr(backend_engine, "render_trade_execution"):
        backend_engine.render_trade_execution()
    else:
        st.info("🚧 Manual Execution Mode")
        st.markdown("""
        **Protocol:**
        1. Review setups in 'Weekly Opportunities'.
        2. Execute orders in your broker terminal.
        3. No automated execution connected.
        """)

def render_analytics_page():
    st.title("📈 Performance & Analytics")
    
    if BACKEND_AVAILABLE and hasattr(backend_engine, "render_performance_analytics"):
        backend_engine.render_performance_analytics()
    else:
        st.info("🚧 Analytics module placeholder.")
        st.caption("Performance tracking logic not currently loaded.")

# ==============================================================================
# 4. MAIN NAVIGATION
# ==============================================================================

def main():
    st.sidebar.title("🔍 Navigation")
    
    page = st.sidebar.radio("Go to", [
        "Dashboard", 
        "Weekly Opportunities", 
        "Trade Execution", 
        "Performance & Analytics"
    ])

    if page == "Dashboard":
        render_dashboard_page()
    elif page == "Weekly Opportunities":
        render_weekly_opportunities_page()
    elif page == "Trade Execution":
        render_trade_execution_page()
    elif page == "Performance & Analytics":
        render_analytics_page()

if __name__ == "__main__":
    main()