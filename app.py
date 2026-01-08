# ==============================================================================
# 📊 AI FUNDS MANAGER DASHBOARD (UI LAYER)
# ==============================================================================
# PURPOSE: Visualize the trading book, system health, and execution plans.
# READS FROM: data/weekly/weekly_setups.csv & data/system/last_successful_scan.txt
# ACTIONS: Triggers GitHub Actions for remote scanning (No local execution).
# ==============================================================================

import streamlit as st
import pandas as pd
import os
import datetime
import pytz
import requests

# ==============================================================================
# 1. CONFIGURATION & PAGE SETUP
# ==============================================================================
st.set_page_config(
    page_title="AI Fund Manager",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Paths (Must match ai_stock_system.py structure)
BASE_DIR = "data"
SYSTEM_DIR = f"{BASE_DIR}/system"
WEEKLY_DIR = f"{BASE_DIR}/weekly"
GUARD_FILE = f"{SYSTEM_DIR}/last_successful_scan.txt"
WEEKLY_CSV = f"{WEEKLY_DIR}/weekly_setups.csv"

# ==============================================================================
# 2. GITHUB TRIGGER LOGIC (MANUAL SCAN)
# ==============================================================================
def trigger_github_scan():
    """
    Triggers the GitHub Action workflow via the repository_dispatch event.
    Requires st.secrets to be configured.
    """
    # 1. Fetch Secrets safely
    try:
        token = st.secrets["GITHUB_TOKEN"]
        owner = st.secrets["GITHUB_OWNER"]
        repo = st.secrets["GITHUB_REPO"]
        event_type = st.secrets.get("GITHUB_WORKFLOW_DISPATCH_EVENT", "manual_scan_triggered")
    except Exception:
        st.error("❌ GitHub Secrets not found. Please configure GITHUB_TOKEN, OWNER, and REPO in Streamlit secrets.")
        return

    # 2. Prepare API Request
    url = f"https://api.github.com/repos/{owner}/{repo}/dispatches"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {"event_type": event_type}

    # 3. Send Request
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 204:
            st.success("✅ Scan request sent to GitHub. The scanner will run remotely and update results shortly.")
        else:
            st.error(f"❌ Failed to trigger scan. GitHub Response: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Connection error triggering GitHub: {e}")

# ==============================================================================
# 3. SYSTEM STATUS LOGIC (FRESHNESS-BASED)
# ==============================================================================
def get_system_health():
    """
    Reads the last scan timestamp and determines system health based on data freshness.
    Returns: (status_type, message, minutes_ago, timestamp_obj)
    """
    if not os.path.exists(GUARD_FILE):
        return "no_data", "System initializing... (Waiting for first scan)", 0, None

    try:
        with open(GUARD_FILE, 'r') as f:
            last_ts_str = f.read().strip()
        
        if not last_ts_str:
            return "no_data", "System initializing... (Log empty)", 0, None

        # Parse Timestamp (Handle UTC)
        last_scan_time = datetime.datetime.fromisoformat(last_ts_str)
        # Ensure timezone awareness
        if last_scan_time.tzinfo is None:
            last_scan_time = last_scan_time.replace(tzinfo=datetime.timezone.utc)
            
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        diff_seconds = (now_utc - last_scan_time).total_seconds()
        minutes_ago = int(diff_seconds / 60)
        
        # Freshness Logic
        if minutes_ago <= 20:
            return "healthy", f"Scanner healthy — data fresh", minutes_ago, last_scan_time
        elif minutes_ago <= 60:
            return "catching_up", f"Scanner catching up", minutes_ago, last_scan_time
        else:
            return "stale", f"Scanner stale — execution delayed", minutes_ago, last_scan_time
            
    except Exception as e:
        return "error", f"Status check error: {str(e)}", 0, None

def render_status_section():
    """Renders the status banner, refresh button, and manual trigger."""
    st.markdown("### 📡 System Status")
    
    # Layout: Status Banner (Left) | Action Buttons (Right)
    col_status, col_actions = st.columns([3, 2])
    
    with col_actions:
        # Action Buttons Layout
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Refresh UI", use_container_width=True):
                st.rerun()
        with c2:
            # Manual Trigger Button
            if st.button("🔁 Run Fresh Scan", type="primary", use_container_width=True):
                trigger_github_scan()

    with col_status:
        status, msg, mins, ts_obj = get_system_health()
        
        # IST Conversion for readability
        if ts_obj:
            ist_tz = pytz.timezone('Asia/Kolkata')
            ts_ist = ts_obj.astimezone(ist_tz).strftime('%H:%M IST')
            time_label = f"(Last scan: {ts_ist} • {mins} min ago)"
        else:
            time_label = ""

        if status == "healthy":
            st.success(f"🟢 **{msg}** {time_label}")
            
        elif status == "catching_up":
            st.warning(f"🟡 **{msg}** {time_label}")
            st.caption("ℹ️ Scheduler delay detected (System self-healing active). No action required.")
            
        elif status == "stale":
            st.error(f"🔴 **{msg}** {time_label}")
            st.caption("ℹ️ Execution delayed > 1 hour. Click 'Run Fresh Scan' to force update.")
            
        elif status == "no_data":
            st.info(f"⚪ **{msg}**")
            
        else:
            st.error(f"❌ {msg}")

# ==============================================================================
# 4. DATA LOADING
# ==============================================================================
def load_weekly_book():
    if not os.path.exists(WEEKLY_CSV):
        return pd.DataFrame()
    try:
        return pd.read_csv(WEEKLY_CSV)
    except Exception as e:
        st.error(f"Error reading weekly book: {e}")
        return pd.DataFrame()

# ==============================================================================
# 5. MAIN DASHBOARD UI
# ==============================================================================
def main():
    st.title("🏦 AI Fund Manager Console")
    st.markdown("---")
    
    # 1. RENDER STATUS PANEL (Top Priority)
    render_status_section()
    
    st.markdown("---")
    
    # 2. LOAD DATA
    df = load_weekly_book()
    
    # 3. DISPLAY WEEKLY BOOK
    st.header("📖 Weekly Opportunity Book")
    
    if df.empty:
        # FIX 1: Neutral message (No discipline psychology)
        st.info("📭 No actionable setups found.")
    else:
        # Sort by Detected Date (Newest first)
        if 'Detected_Date' in df.columns:
            df['Detected_Date'] = pd.to_datetime(df['Detected_Date'])
            df = df.sort_values(by='Detected_Date', ascending=False)
        
        # Display Metrics
        active_count = len(df[df['Status'] == 'ACTIVE']) if 'Status' in df.columns else len(df)
        st.metric("Active Setups in Book", active_count)
        
        # FIX 2: Removed 'Fund_Status' from display columns
        display_cols = [
            'Ticker', 'Setup', 'Entry', 'StopLoss', 'Target1', 
            'Target2', 'Risk_Pct', 'Status', 'Detected_Date'
        ]
        # Filter columns that actually exist in the CSV
        valid_cols = [c for c in display_cols if c in df.columns]
        
        st.dataframe(
            df[valid_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Entry": st.column_config.NumberColumn(format="₹%.2f"),
                "StopLoss": st.column_config.NumberColumn(format="₹%.2f"),
                "Target1": st.column_config.NumberColumn(format="₹%.2f"),
                "Target2": st.column_config.NumberColumn(format="₹%.2f"),
                "Risk_Pct": st.column_config.NumberColumn(format="%.2f%%"),
                "Detected_Date": st.column_config.DateColumn("Found On", format="DD MMM"),
            }
        )

if __name__ == "__main__":
    main()