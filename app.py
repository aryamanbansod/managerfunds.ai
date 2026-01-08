import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date, timedelta
import backend_engine  # Phase 1 Logic
import sqlite3
import os
import yfinance as yf 
import time

# ========================================================
# CONFIGURATION & CONSTANTS
# ========================================================
st.set_page_config(
    page_title="AI Fund Manager | Pro Desk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Path to the AI Brain's output CSV
AI_SIGNALS_PATH = "data/signals/smart_money_tradr_signals.csv" 
CONFIDENCE_THRESHOLD = 90.0 # Score to be considered "TRADR-GRADE" (A+)
MAX_HOLDING_DAYS = 20

# ========================================================
# DATABASE & INITIALIZATION CHECKS
# ========================================================

def ensure_tables_exist():
    """Ensures DB tables exist. If not, creates them."""
    backend_engine.init_db()

def is_account_initialized():
    """Checks if the account_state table has data."""
    try:
        conn = backend_engine.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM account_state")
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    except sqlite3.OperationalError:
        return False
    except Exception as e:
        return False

# ========================================================
# DATA SYNC & LOADING
# ========================================================

def sync_ai_data():
    """
    WIRES THE CSV TO THE DATABASE.
    Checks for the AI signal CSV and loads it into SQLite via backend_engine.
    This ensures the Dashboard always sees what the AI saw.
    """
    if os.path.exists(AI_SIGNALS_PATH):
        try:
            # We use the backend function to ingest CSV safely
            # It handles duplicates (UPSERT) automatically
            backend_engine.load_ai_weekly_setups(AI_SIGNALS_PATH)
            return True
        except Exception as e:
            st.error(f"Error syncing AI data: {e}")
            return False
    return False

def load_data():
    """Fetches fresh data from backend engine after syncing."""
    try:
        # 1. Sync latest AI signals first
        sync_ai_data()

        # 2. Fetch from DB (Single Source of Truth)
        account_state = backend_engine.get_account_state()
        active_trades = backend_engine.get_active_trades()
        weekly_setups = backend_engine.get_weekly_setups()
        
        # Load all trades (active + closed) for performance analysis
        conn = backend_engine.get_db_connection()
        all_trades = pd.read_sql_query("SELECT * FROM trades", conn)
        trade_actions = pd.read_sql_query("SELECT * FROM trade_actions", conn)
        conn.close()
        
        return account_state, active_trades, weekly_setups, all_trades, trade_actions
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# ========================================================
# UTILITY FUNCTIONS
# ========================================================
def format_currency(value):
    if value is None: return "₹0.00"
    return f"₹{value:,.2f}"

def calculate_days_in_trade(entry_date_str):
    try:
        entry_dt = pd.to_datetime(entry_date_str)
        now = datetime.now()
        delta = now - entry_dt
        return delta.days
    except:
        return 0

def get_current_week_dates():
    today = date.today()
    start_week = today - timedelta(days=today.weekday())
    end_week = start_week + timedelta(days=4)
    return start_week, end_week

def skip_setup(setup_id):
    """UI helper to mark setup as skipped"""
    conn = backend_engine.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE weekly_setups SET status='SKIPPED' WHERE id=?", (setup_id,))
    conn.commit()
    conn.close()

def get_live_cmp(ticker):
    """Fetches live CMP from yfinance for dashboard context (Read-Only)."""
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
    except:
        pass
    return None

def get_scan_status_panel():
    """Generates the Live Scan Status Panel HTML with Delay Logic."""
    if os.path.exists(AI_SIGNALS_PATH):
        mtime = os.path.getmtime(AI_SIGNALS_PATH)
        # Convert File Time (Server) to IST (UTC+5:30)
        dt_utc = datetime.utcfromtimestamp(mtime)
        dt_ist = dt_utc + timedelta(hours=5, minutes=30)
        last_scan_str = dt_ist.strftime("%d %b %Y, %H:%M IST")
        
        # Next Scan Calculation (Approx +10 mins)
        next_scan_dt = dt_ist + timedelta(minutes=10)
        next_scan_str = next_scan_dt.strftime("%H:%M IST")
        
        # Current Time in IST
        now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
        remaining = next_scan_dt - now_ist
        
        # --- ROBUSTNESS FIX ---
        # If remaining is less than -3 minutes (i.e. > 3 mins late), show delayed status
        is_delayed = remaining.total_seconds() < -180
        
        if is_delayed:
             time_display = f"<span style='color: #FFA500; font-weight: bold;'>⚠️ Scan delayed (GitHub queue)</span>"
             sub_text = "GitHub Actions often queue jobs. Data will update automatically on completion."
        elif remaining.total_seconds() < 0:
            time_display = "<span style='color: #888;'>00:00 (Processing...)</span>"
            sub_text = "Action: Stand down. Capital preserved."
        else:
            mins, secs = divmod(int(remaining.total_seconds()), 60)
            remaining_str = f"{mins:02d}:{secs:02d}"
            time_display = f"<span style='color: #FFD700;'>⏳ Next scan in&nbsp;: {remaining_str}</span>"
            sub_text = "Action: Stand down. Capital preserved."
            
        return f"""
        <div style="padding: 15px; border: 1px solid #444; border-radius: 8px; background-color: #0E1117; margin-top: 10px;">
            <h4 style="margin-top:0; color: #00FF00; display: flex; align-items: center;">
                <span style="font-size: 1.2em; margin-right: 10px;">●</span> SYSTEM STATUS: ACTIVE
            </h4>
            <div style="font-family: monospace; color: #E0E0E0; font-size: 0.95em; line-height: 1.6;">
                Last Scan&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {last_scan_str}<br>
                Scan Frequency : Every 10 minutes (Market & Off-Market)<br>
                Next Scan&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {next_scan_str}<br>
                {time_display}
            </div>
            <hr style="margin: 12px 0; border-color: #333;">
            <div style="color: #888; font-size: 0.9em; font-style: italic;">
                {sub_text}
            </div>
        </div>
        """
    else:
        return """
        <div style="padding: 10px; border: 1px solid #444; border-radius: 5px; background-color: #262730;">
            <h4 style="margin:0; color: #FFA500;">⚠️ System Status: Waiting for first scan...</h4>
        </div>
        """

# ========================================================
# LOGIC LAYERS (3A, 3B, 3C)
# ========================================================

def determine_lifecycle_state(row):
    """Phase 3C: Maps raw data to Canonical Lifecycle States."""
    if 'trade_status' not in row: 
        return "AWAITING_ENTRY"
    
    status = row['trade_status']
    initial_qty = row['initial_quantity']
    rem_qty = row['remaining_quantity']
    
    if status == 'CLOSED':
        return "EXITED"
    elif status == 'ACTIVE':
        if rem_qty == initial_qty:
            return "ACTIVE (Full Position)"
        elif rem_qty < initial_qty and rem_qty > 0:
            return "RUNNER ACTIVE"
    return "UNKNOWN"

def get_guided_exit_message(row, cmp):
    """Phase 3C: Generates context-aware guidance."""
    state = determine_lifecycle_state(row)
    
    if state == "ACTIVE (Full Position)":
        if cmp and cmp >= row['target1']:
            return "🎯 **TP1 LEVEL REACHED:** Plan suggests booking partial profits (50%)."
        elif cmp and cmp <= row['stop_loss']:
            return "❌ **SL CONDITION MET:** Discipline check. Confirm exit if valid."
        else:
            return "🔵 **HOLDING:** Monitoring for TP1 or SL."
            
    elif state == "RUNNER ACTIVE":
        if cmp and cmp >= row['target2']:
            return "🚀 **TP2 LEVEL REACHED:** Plan suggests fully closing the trade."
        else:
            return "🏃 **RUNNER ACTIVE:** SL should be at Breakeven. Hold for expansion."
            
    return "⚪ Awaiting Price Action."

def enrich_setup_data(df, total_capital):
    """Combines Logic from 3A (Decision) and 3B (Capital)."""
    if df.empty: return df

    def get_grade(score):
        if score >= 90: return "A+"
        elif score >= 80: return "A"
        return "B"
    
    def get_confidence(grade):
        if grade in ["A+", "A"]: return "High"
        return "Medium"
        
    def get_reason(setup_type):
        reasons = {
            "VCP_COMPRESSION": "Volatility contraction + Vol dry-up",
            "HIGH_TIGHT_FLAG": "Momentum thrust + Tight consolidation",
            "NR7_CONTRACTION": "Extreme range contraction (Imminent Move)",
            "44MA_SUPPORT_BUY": "Institutional support pullback",
            "ACCUMULATION_SPIKE": "Institutional volume spike (Ignition)"
        }
        return reasons.get(setup_type, "Technical Setup")

    def calculate_risk_metrics(row):
        entry = row['entry_price']
        sl = row['stop_loss']
        qty = row['quantity']
        risk_per_share = entry - sl
        total_risk_rupees = risk_per_share * qty
        risk_pct_capital = (total_risk_rupees / total_capital) * 100 if total_capital > 0 else 0
        return total_risk_rupees, risk_pct_capital

    def get_risk_tag(risk_pct):
        if risk_pct < 1.5: return "🟢 SAFE"
        elif risk_pct <= 2.5: return "🟡 NORMAL"
        else: return "🔴 AGGRESSIVE"

    df['Grade'] = df['confidence_score'].apply(get_grade)
    df['Confidence'] = df['Grade'].apply(get_confidence)
    df['Logic'] = df['setup_type'].apply(get_reason)
    
    risk_data = df.apply(calculate_risk_metrics, axis=1, result_type='expand')
    df['Risk_Rupees'] = risk_data[0]
    df['Risk_Pct'] = risk_data[1]
    df['Risk_Tag'] = df['Risk_Pct'].apply(get_risk_tag)

    return df

# ========================================================
# VIEW: INITIALIZATION SCREEN
# ========================================================
def render_initialization_screen():
    st.markdown("""<style>.block-container {padding-top: 3rem;}</style>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/4207/4207247.png", width=100)
        st.title("Initialize Virtual Fund Manager")
        st.markdown("### Welcome to your AI Trading Desk")
        st.info("First run detected. Please set up your initial capital parameters.")
        with st.form("init_form"):
            initial_capital = st.number_input("Starting Capital (INR)", min_value=1000.0, value=100000.0, step=1000.0, format="%.2f")
            risk_pct = st.number_input("Weekly Loss Limit (%)", min_value=0.5, max_value=10.0, value=2.5, step=0.1)
            submitted = st.form_submit_button("🚀 Initialize Fund & Launch Dashboard", type="primary", use_container_width=True)
            if submitted:
                try:
                    ensure_tables_exist()
                    backend_engine.initialize_account(initial_capital, risk_pct)
                    st.success("Fund Initialized Successfully! Reloading...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Initialization Failed: {e}")

# ========================================================
# VIEW 1: DASHBOARD
# ========================================================
def render_dashboard(account_df, trades_df):
    st.title("🏦 Capital & Portfolio Overview")
    st.markdown("---")
    if account_df.empty:
        st.warning("⚠️ Account state not readable.")
        return

    account = account_df.iloc[0]
    total_cap = account['total_capital']
    free_cap = account['free_capital']
    deployed_cap = account['deployed_capital']
    realized_pnl = account['realized_pnl']
    weekly_pnl = account['weekly_pnl']
    is_halted = account['trading_halted']
    loss_limit_pct = account['weekly_loss_limit_pct']

    # 3B: Capital Deployment
    pct_deployed = (deployed_cap / total_cap) * 100 if total_cap > 0 else 0
    if pct_deployed < 50: exposure_status = "🟢 Comfortable"
    elif pct_deployed <= 70: exposure_status = "🟡 Moderate Exposure"
    else: exposure_status = "🔴 High Exposure (Size Down)"

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("💰 Total Capital", format_currency(total_cap))
    with col2: st.metric("🔴 Deployed Capital", format_currency(deployed_cap), delta=f"{pct_deployed:.1f}% Used", delta_color="inverse")
    with col3: st.metric("🟢 Free Capital", format_currency(free_cap))
    with col4: st.metric("Exposure Status", exposure_status)

    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Realized P&L (All Time)", format_currency(realized_pnl), delta=f"{realized_pnl:,.2f}")
    with c2: st.metric("Weekly P&L", format_currency(weekly_pnl), delta=f"{weekly_pnl:,.2f}")
    with c3:
        max_loss_amt = total_cap * (loss_limit_pct / 100.0)
        remaining_risk = max_loss_amt + weekly_pnl
        if remaining_risk < 0: remaining_risk = 0
        st.metric("Remaining Weekly Risk Buffer", format_currency(remaining_risk))

    st.subheader("📋 Active Trades Book")
    if trades_df.empty:
        st.info("No active trades currently open.")
    else:
        display_df = trades_df.copy()
        display_df['Days in Trade'] = display_df['entry_date'].apply(calculate_days_in_trade)
        # 3C: Apply Lifecycle State
        display_df['Lifecycle'] = display_df.apply(determine_lifecycle_state, axis=1)
        
        final_view = display_df[['ticker', 'Lifecycle', 'entry_price', 'stop_loss', 'target1', 'target2', 'remaining_quantity', 'Days in Trade']]
        st.dataframe(
            final_view, use_container_width=True, hide_index=True,
            column_config={
                "entry_price": st.column_config.NumberColumn("Entry", format="₹%.2f"),
                "stop_loss": st.column_config.NumberColumn("SL", format="₹%.2f"),
                "target1": st.column_config.NumberColumn("TP1", format="₹%.2f"),
                "target2": st.column_config.NumberColumn("TP2", format="₹%.2f"),
            }
        )

# ========================================================
# VIEW 2: WEEKLY OPPORTUNITY BOOK
# ========================================================
def render_opportunity_book(setups_df, account_df, trades_df):
    st.title("📅 Weekly Opportunity Book")
    start_date, end_date = get_current_week_dates()
    today_date = date.today()
    total_capital = account_df.iloc[0]['total_capital'] if not account_df.empty else 100000

    st.markdown(f"#### **Trading Week:** {start_date.strftime('%d %b')} — {end_date.strftime('%d %b %Y')}")
    
    current_week_setups = pd.DataFrame()
    if not setups_df.empty:
        setups_df['week_start_dt'] = pd.to_datetime(setups_df['week_start_date']).dt.date
        current_week_setups = setups_df[setups_df['week_start_dt'] == start_date].copy()
        current_week_setups['confidence_score'] = pd.to_numeric(current_week_setups['confidence_score'], errors='coerce').fillna(0)
        current_week_setups = enrich_setup_data(current_week_setups, total_capital)

    today_signals = pd.DataFrame()
    if not current_week_setups.empty:
        current_week_setups['detected_dt'] = pd.to_datetime(current_week_setups['first_detected_date']).dt.date
        today_signals = current_week_setups[current_week_setups['detected_dt'] == today_date]

    # --- DECISION CLARITY BANNER & LIVE STATUS ---
    if not today_signals.empty:
        st.success("🟢 **UPL-GRADE SETUP ACTIVE — EXECUTION WINDOW OPEN**")
        st.markdown(f"**Action Required:** Review {len(today_signals)} new high-confidence setups below.")
    elif not current_week_setups.empty:
        st.warning("🟡 **NO NEW SETUP TODAY — WEEKLY SCAN ACTIVE**")
        st.markdown("**Action:** Monitor existing watchlist. No new aggressive entries today.")
    else:
        st.info("🔵 **NO HIGH-CONVICTION SETUPS THIS WEEK — CAPITAL PRESERVED**")
        # LIVE SCAN STATUS PANEL REPLACES STATIC TEXT HERE
        st.markdown(get_scan_status_panel(), unsafe_allow_html=True)
        
    st.markdown("---")

    if current_week_setups.empty:
        return

    tradr_grade = current_week_setups[current_week_setups['Grade'].isin(["A+", "A"])]
    alternatives = current_week_setups[current_week_setups['Grade'] == "B"]

    st.subheader("🚀 High-Confidence Setups (A+ / A)")
    if tradr_grade.empty:
        st.caption("No high-confidence setups found this week.")
    else:
        display_cols = ['ticker', 'Grade', 'Risk_Tag', 'Logic', 'Risk_Pct', 'Risk_Rupees', 'entry_price', 'stop_loss', 'target1', 'expected_move_pct', 'status']
        st.dataframe(
            tradr_grade[display_cols], use_container_width=True, hide_index=True,
            column_config={
                "ticker": "Ticker", "Grade": "Grade", "Risk_Tag": "Risk Level",
                "Logic": st.column_config.TextColumn("Why This Exists", width="medium"),
                "Risk_Pct": st.column_config.NumberColumn("Risk %", format="%.2f%%"),
                "Risk_Rupees": st.column_config.NumberColumn("Risk ₹", format="₹%.2f"),
                "entry_price": st.column_config.NumberColumn("Entry", format="₹%.2f"),
                "stop_loss": st.column_config.NumberColumn("SL", format="₹%.2f"),
                "target1": st.column_config.NumberColumn("TP1", format="₹%.2f"),
                "expected_move_pct": st.column_config.NumberColumn("Pot. ROI", format="%.1f%%"),
                "status": st.column_config.TextColumn("Status"),
            }
        )

    st.markdown("---")

    st.subheader("⚠️ Alternative Setups (Grade B)")
    if alternatives.empty:
        st.caption("No alternative setups found.")
    else:
        display_cols = ['ticker', 'Grade', 'Risk_Tag', 'Logic', 'Risk_Pct', 'entry_price', 'stop_loss', 'confidence_score', 'status']
        st.dataframe(
            alternatives[display_cols], use_container_width=True, hide_index=True,
            column_config={
                "Logic": st.column_config.TextColumn("Why This Exists", width="medium"),
                "Risk_Pct": st.column_config.NumberColumn("Risk %", format="%.2f%%"),
                "entry_price": st.column_config.NumberColumn("Entry", format="₹%.2f"),
                "stop_loss": st.column_config.NumberColumn("SL", format="₹%.2f"),
                "confidence_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
            }
        )

# ========================================================
# VIEW 3: TRADE EXECUTION (PHASE 3C ENHANCED)
# ========================================================
def render_trade_execution(setups_df, trades_df):
    st.title("⚡ Trade Execution Desk")
    
    active_count = 0
    runner_count = 0
    nearing_exit = 0
    
    if not trades_df.empty:
        trades_df['Lifecycle'] = trades_df.apply(determine_lifecycle_state, axis=1)
        trades_df['Days in Trade'] = trades_df['entry_date'].apply(calculate_days_in_trade)
        
        active_count = len(trades_df[trades_df['Lifecycle'] == "ACTIVE (Full Position)"])
        runner_count = len(trades_df[trades_df['Lifecycle'] == "RUNNER ACTIVE"])
        nearing_exit = len(trades_df[trades_df['Days in Trade'] >= (MAX_HOLDING_DAYS - 5)])
    
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("🔵 Active Full Positions", active_count)
    with c2: st.metric("🏃 Runners Active", runner_count)
    with c3: 
        if nearing_exit > 0:
            st.metric("⏳ Nearing Time Exit", nearing_exit, delta="Action Needed", delta_color="inverse")
        else:
            st.metric("⏳ Nearing Time Exit", 0)
            
    st.markdown("---")

    # --- 3B: Risk Buffer (Retained) ---
    account_df = backend_engine.get_account_state()
    if not account_df.empty:
        total_cap = account_df.iloc[0]['total_capital']
        loss_limit_pct = account_df.iloc[0]['weekly_loss_limit_pct']
        weekly_pnl = account_df.iloc[0]['weekly_pnl']
        max_loss_amt = total_cap * (loss_limit_pct / 100.0)
        remaining_buffer = max_loss_amt + weekly_pnl 
        if remaining_buffer < 0: remaining_buffer = 0
    else:
        remaining_buffer = 2500

    mode = st.radio("Select List:", ["New Setups (Awaiting Entry)", "Active Trades (Open)"], horizontal=True)
    selected_trade = None
    trade_source = None 

    if mode == "New Setups (Awaiting Entry)":
        start_date, end_date = get_current_week_dates()
        if setups_df.empty:
            st.info("No setups loaded.")
            candidates = pd.DataFrame()
        else:
            setups_df['week_start_dt'] = pd.to_datetime(setups_df['week_start_date']).dt.date
            candidates = setups_df[(setups_df['week_start_dt'] == start_date) & (setups_df['status'] == 'AWAITING_ENTRY')]

        if candidates.empty:
            st.info("No actionable setups awaiting entry for this week.")
        else:
            options = candidates['ticker'].tolist()
            selected_ticker = st.selectbox("Select Ticker to Execute:", options)
            if selected_ticker:
                selected_trade = candidates[candidates['ticker'] == selected_ticker].iloc[0]
                trade_source = 'setup'
    else:
        if trades_df.empty:
            st.info("No active trades.")
        else:
            options = trades_df['ticker'].tolist()
            selected_ticker = st.selectbox("Select Active Trade:", options)
            if selected_ticker:
                selected_trade = trades_df[trades_df['ticker'] == selected_ticker].iloc[0]
                trade_source = 'trade'

    st.divider()

    if selected_trade is not None:
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown(f"### 🎫 {selected_trade['ticker']}")
            if trade_source == 'setup':
                st.caption(f"Pattern: {selected_trade['setup_type']}")
                st.metric("Entry Trigger", format_currency(selected_trade['entry_price']))
                st.metric("Stop Loss", format_currency(selected_trade['stop_loss']))
                st.metric("Target 1", format_currency(selected_trade['target1']))
                st.metric("Quantity", selected_trade['quantity'])
                st.caption(f"Confidence: {selected_trade.get('confidence_score', 0)}")
            else:
                days_held = calculate_days_in_trade(selected_trade['entry_date'])
                lifecycle_state = determine_lifecycle_state(selected_trade)
                
                st.caption(f"State: **{lifecycle_state}**")
                st.metric("Entry Price", format_currency(selected_trade['entry_price']))
                st.metric("Remaining Qty", selected_trade['remaining_quantity'])
                
                st.write(f"**Timeline:** Day {days_held} / {MAX_HOLDING_DAYS}")
                st.progress(min(days_held / MAX_HOLDING_DAYS, 1.0))
                
                cmp = get_live_cmp(selected_trade['ticker'])
                if cmp:
                    st.metric("Live Price (Est.)", format_currency(cmp), delta=f"{((cmp - selected_trade['entry_price'])/selected_trade['entry_price'])*100:.2f}%")
                
                guidance = get_guided_exit_message(selected_trade, cmp)
                st.info(guidance)

        with c2:
            st.subheader("⚙️ Execution Console")
            
            if trade_source == 'setup':
                risk_per_share = selected_trade['entry_price'] - selected_trade['stop_loss']
                trade_risk = risk_per_share * selected_trade['quantity']
                st.info(f"💡 **Confirm Entry:** Current Price should be near {format_currency(selected_trade['entry_price'])}")
                st.write(f"📉 **Trade Risk:** {format_currency(trade_risk)}")
                if trade_risk > remaining_buffer:
                    st.warning(f"⚠️ **WARNING:** Risk ({format_currency(trade_risk)}) > Buffer ({format_currency(remaining_buffer)}).")

                col_ex1, col_ex2 = st.columns(2)
                with col_ex1:
                    if st.button("✅ CONFIRM ENTRY TAKEN", type="primary", use_container_width=True):
                        success = backend_engine.confirm_trade_entry(int(selected_trade['id']))
                        if success:
                            st.success(f"Trade for {selected_trade['ticker']} ACTIVATED!")
                            st.rerun()
                with col_ex2:
                    if st.button("❌ SKIP TRADE", use_container_width=True):
                        skip_setup(int(selected_trade['id']))
                        st.warning("Skipped.")
                        st.rerun()

            elif trade_source == 'trade':
                trade_id = int(selected_trade['id'])
                current_qty = int(selected_trade['remaining_quantity'])
                action = st.selectbox("Select Action:", ["Select...", "🎯 Book Target 1", "🎯 Book Target 2", "❌ Stop Loss Exit", "⏳ Time Exit"])
                
                if action == "🎯 Book Target 1":
                    st.markdown("#### TP1 Execution")
                    rec_qty = int(current_qty * 0.5) 
                    exit_qty = st.number_input("Quantity Exited", min_value=1, max_value=current_qty, value=rec_qty)
                    exit_price = st.number_input("Exit Price", value=float(selected_trade['target1']))
                    brokerage = st.number_input("Total Brokerage & Charges (₹)", min_value=0.0, step=10.0)
                    if st.button("Confirm TP1 Booking"):
                        backend_engine.confirm_exit_action(trade_id, "TP1", exit_price, exit_qty, brokerage)
                        st.success("TP1 Booked.")
                        st.rerun()
                elif action == "🎯 Book Target 2":
                    st.markdown("#### TP2 Execution (Final)")
                    exit_qty = st.number_input("Quantity Exited", min_value=1, max_value=current_qty, value=current_qty)
                    exit_price = st.number_input("Exit Price", value=float(selected_trade['target2']))
                    brokerage = st.number_input("Total Brokerage & Charges (₹)", min_value=0.0, step=10.0)
                    if st.button("Confirm TP2 Booking"):
                        backend_engine.confirm_exit_action(trade_id, "TP2", exit_price, exit_qty, brokerage)
                        st.success("TP2 Booked.")
                        st.rerun()
                elif action == "❌ Stop Loss Exit":
                    st.markdown("#### Stop Loss Execution")
                    exit_qty = st.number_input("Quantity Exited", min_value=1, max_value=current_qty, value=current_qty)
                    exit_price = st.number_input("Exit Price", value=float(selected_trade['stop_loss']))
                    brokerage = st.number_input("Total Brokerage & Charges (₹)", min_value=0.0, step=10.0)
                    if st.button("Confirm SL Exit", type="primary"):
                        backend_engine.confirm_exit_action(trade_id, "SL", exit_price, exit_qty, brokerage)
                        st.error("Stop Loss Hit.")
                        st.rerun()
                elif action == "⏳ Time Exit":
                    st.markdown("#### Manual/Time Exit")
                    exit_qty = st.number_input("Quantity Exited", min_value=1, max_value=current_qty, value=current_qty)
                    exit_price = st.number_input("Exit Price", value=0.0)
                    brokerage = st.number_input("Total Brokerage & Charges (₹)", min_value=0.0, step=10.0)
                    if st.button("Confirm Manual Exit"):
                        backend_engine.confirm_exit_action(trade_id, "TIME_EXIT", exit_price, exit_qty, brokerage)
                        st.info("Closed Manually.")
                        st.rerun()

# ========================================================
# VIEW 4: PERFORMANCE & DISCIPLINE
# ========================================================
def render_performance(all_trades, trade_actions, account_df):
    st.title("🏆 Performance & Discipline")
    st.markdown("### The Truth Screen")
    st.markdown("---")
    if all_trades.empty or trade_actions.empty:
        st.info("Insufficient data to generate performance metrics. Execute some trades first.")
        return

    st.subheader("1. System Performance Snapshot")
    closed_actions = trade_actions[trade_actions['action_type'].isin(['TP1', 'TP2', 'SL', 'TIME_EXIT'])]
    winning_trades = closed_actions[closed_actions['net_pnl'] > 0]
    losing_trades = closed_actions[closed_actions['net_pnl'] <= 0]
    
    win_count = len(winning_trades)
    total_closed = len(closed_actions)
    win_rate = (win_count / total_closed * 100) if total_closed > 0 else 0.0
    avg_win = winning_trades['net_pnl'].mean() if not winning_trades.empty else 0.0
    avg_loss = losing_trades['net_pnl'].mean() if not losing_trades.empty else 0.0
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    
    current_equity = account_df.iloc[0]['total_capital'] if not account_df.empty else 0.0

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Total Trades", len(all_trades))
    with m2: st.metric("Win Rate", f"{win_rate:.1f}%")
    with m3: st.metric("Avg Win", format_currency(avg_win))
    with m4: st.metric("Avg Loss", format_currency(avg_loss))
    
    m5, m6, m7, m8 = st.columns(4)
    with m5: st.metric("Expectancy (₹/Trade)", format_currency(expectancy))
    with m6: st.metric("Current Equity", format_currency(current_equity))
    with m7: st.metric("Net Profit", format_currency(account_df.iloc[0]['realized_pnl']))
    with m8: st.metric("Profit Factor", f"{abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "N/A")

    st.markdown("---")
    st.subheader("2. Equity Curve")
    start_cap = current_equity - account_df.iloc[0]['realized_pnl']
    equity_data = [{'Date': 'Start', 'Equity': start_cap}]
    running_equity = start_cap
    sorted_actions = trade_actions.sort_values(by='action_date')
    
    for _, row in sorted_actions.iterrows():
        pnl = row['net_pnl'] if row['net_pnl'] else 0
        if pnl != 0:
            running_equity += pnl
            equity_data.append({'Date': row['action_date'], 'Equity': running_equity})
            
    eq_df = pd.DataFrame(equity_data)
    if len(eq_df) > 1:
        fig = px.line(eq_df, x='Date', y='Equity', title='Portfolio Growth', markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Equity curve will appear after first closed trade.")

    st.markdown("---")
    st.subheader("3. Discipline Scoreboard")
    runner_count = len(trade_actions[trade_actions['action_type'] == 'TP1'])
    runner_pct = (runner_count / total_closed * 100) if total_closed > 0 else 0
    d1, d2, d3 = st.columns(3)
    with d1: st.metric("Trades Converted to Runners", f"{runner_count} ({runner_pct:.1f}%)")
    with d2: st.metric("Brokerage Paid (Total)", format_currency(trade_actions['brokerage_charges'].sum()))
    with d3: 
        week_start, week_end = get_current_week_dates()
        trades_this_week = all_trades[pd.to_datetime(all_trades['entry_date']).dt.date >= week_start]
        st.metric("Trades This Week", f"{len(trades_this_week)} / 3 (Max)")

# ========================================================
# MAIN APP CONTROLLER
# ========================================================
if __name__ == "__main__":
    ensure_tables_exist()

    if not is_account_initialized():
        render_initialization_screen()
    else:
        # Load and Sync Data
        account_state, active_trades, weekly_setups, all_trades, trade_actions = load_data()

        with st.sidebar:
            st.header("🧭 Fund Desk")
            page = st.radio("Navigate", ["Dashboard", "Weekly Opportunities", "Trade Execution", "Performance & Analytics"])
            st.divider()
            st.caption("Virtual AI Fund Manager v1.7 (Robust)")

        if page == "Dashboard":
            render_dashboard(account_state, active_trades)
        elif page == "Weekly Opportunities":
            render_opportunity_book(weekly_setups, account_state, active_trades)
        elif page == "Trade Execution":
            render_trade_execution(weekly_setups, active_trades)
        elif page == "Performance & Analytics":
            render_performance(all_trades, trade_actions, account_state)