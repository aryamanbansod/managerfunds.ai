import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date, timedelta
import backend_engine  # Phase 1 Logic
import sqlite3
import os

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

# ========================================================
# DECISION & CAPITAL INTELLIGENCE LOGIC (PHASE 3A + 3B)
# ========================================================
def enrich_setup_data(df, total_capital):
    """
    Adds Decision Clarity fields (Grade, Confidence, Logic) 
    AND Capital Intelligence fields (Risk ₹, Risk %, Risk Tag).
    UI ONLY.
    """
    if df.empty: return df

    # --- 3A: Decision Clarity Mappings ---
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

    # --- 3B: Capital Intelligence Mappings ---
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

    # Apply 3A Mappings
    df['Grade'] = df['confidence_score'].apply(get_grade)
    df['Confidence'] = df['Grade'].apply(get_confidence)
    df['Logic'] = df['setup_type'].apply(get_reason)
    
    # Apply 3B Mappings
    # Apply calculation row-wise
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

    # --- 3B: Capital Deployment Snapshot ---
    pct_deployed = (deployed_cap / total_cap) * 100 if total_cap > 0 else 0
    
    if pct_deployed < 50:
        exposure_status = "🟢 Comfortable"
    elif pct_deployed <= 70:
        exposure_status = "🟡 Moderate Exposure"
    else:
        exposure_status = "🔴 High Exposure (Size Down)"

    # Top Row Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("💰 Total Capital", format_currency(total_cap))
    with col2: st.metric("🔴 Deployed Capital", format_currency(deployed_cap), delta=f"{pct_deployed:.1f}% Used", delta_color="inverse")
    with col3: st.metric("🟢 Free Capital", format_currency(free_cap))
    with col4: st.metric("Exposure Status", exposure_status)

    st.markdown("---")
    
    # Second Row Metrics
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
        final_view = display_df[['ticker', 'trade_status', 'entry_price', 'stop_loss', 'target1', 'target2', 'initial_quantity', 'remaining_quantity', 'Days in Trade']]
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
# VIEW 2: WEEKLY OPPORTUNITY BOOK (DECISION CLARITY + RISK)
# ========================================================
def render_opportunity_book(setups_df, account_df, trades_df):
    st.title("📅 Weekly Opportunity Book")
    start_date, end_date = get_current_week_dates()
    today_date = date.today()
    
    # Get total capital for risk calc
    total_capital = account_df.iloc[0]['total_capital'] if not account_df.empty else 100000

    st.markdown(f"#### **Trading Week:** {start_date.strftime('%d %b')} — {end_date.strftime('%d %b %Y')}")
    
    # 1. Filter for Current Week Only (Monday to Friday)
    current_week_setups = pd.DataFrame()
    if not setups_df.empty:
        setups_df['week_start_dt'] = pd.to_datetime(setups_df['week_start_date']).dt.date
        current_week_setups = setups_df[setups_df['week_start_dt'] == start_date].copy()
        
        # Enrich data with Decision Clarity (3A) AND Capital Intelligence (3B)
        current_week_setups['confidence_score'] = pd.to_numeric(current_week_setups['confidence_score'], errors='coerce').fillna(0)
        current_week_setups = enrich_setup_data(current_week_setups, total_capital)

    # 2. Status Banner Determination
    today_signals = pd.DataFrame()
    if not current_week_setups.empty:
        current_week_setups['detected_dt'] = pd.to_datetime(current_week_setups['first_detected_date']).dt.date
        today_signals = current_week_setups[current_week_setups['detected_dt'] == today_date]

    # --- DECISION CLARITY BANNER ---
    if not today_signals.empty:
        st.success("🟢 **UPL-GRADE SETUP ACTIVE — EXECUTION WINDOW OPEN**")
        st.markdown(f"**Action Required:** Review {len(today_signals)} new high-confidence setups below.")
    elif not current_week_setups.empty:
        st.warning("🟡 **NO NEW SETUP TODAY — WEEKLY SCAN ACTIVE**")
        st.markdown("**Action:** Monitor existing watchlist. No new aggressive entries today.")
    else:
        st.info("🔵 **NO HIGH-CONVICTION SETUPS THIS WEEK — CAPITAL PRESERVED**")
        st.markdown("**Action:** Stand down. Wait for next scan cycle (Daily 8:30 AM).")
    # -------------------------------

    st.markdown("---")

    if current_week_setups.empty:
        return

    # 3. Categorize & Display
    # Separate Lists based on Grade
    tradr_grade = current_week_setups[current_week_setups['Grade'].isin(["A+", "A"])]
    alternatives = current_week_setups[current_week_setups['Grade'] == "B"]

    # --- SECTION A: HIGH CONVICTION ---
    st.subheader("🚀 High-Confidence Setups (A+ / A)")
    if tradr_grade.empty:
        st.caption("No high-confidence setups found this week.")
    else:
        # Reorder columns for Decision Clarity + Risk Transparency
        display_cols = ['ticker', 'Grade', 'Risk_Tag', 'Logic', 'Risk_Pct', 'Risk_Rupees', 'entry_price', 'stop_loss', 'target1', 'expected_move_pct', 'status']
        st.dataframe(
            tradr_grade[display_cols],
            use_container_width=True, hide_index=True,
            column_config={
                "ticker": "Ticker",
                "Grade": "Grade",
                "Risk_Tag": "Risk Level",
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

    # --- SECTION B: WATCHLIST ---
    st.subheader("⚠️ Alternative Setups (Grade B)")
    if alternatives.empty:
        st.caption("No alternative setups found.")
    else:
        display_cols = ['ticker', 'Grade', 'Risk_Tag', 'Logic', 'Risk_Pct', 'entry_price', 'stop_loss', 'confidence_score', 'status']
        st.dataframe(
            alternatives[display_cols],
            use_container_width=True, hide_index=True,
            column_config={
                "Logic": st.column_config.TextColumn("Why This Exists", width="medium"),
                "Risk_Pct": st.column_config.NumberColumn("Risk %", format="%.2f%%"),
                "entry_price": st.column_config.NumberColumn("Entry", format="₹%.2f"),
                "stop_loss": st.column_config.NumberColumn("SL", format="₹%.2f"),
                "confidence_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
            }
        )

# ========================================================
# VIEW 3: TRADE EXECUTION (WITH BUFFER GUARDRAIL)
# ========================================================
def render_trade_execution(setups_df, trades_df):
    st.title("⚡ Trade Execution Desk")
    st.caption("Execute trades based on this week's opportunity book.")
    
    # --- 3B: Weekly Risk Buffer Context ---
    # Fetch latest account state for buffer calc
    account_df = backend_engine.get_account_state()
    if not account_df.empty:
        total_cap = account_df.iloc[0]['total_capital']
        loss_limit_pct = account_df.iloc[0]['weekly_loss_limit_pct']
        weekly_pnl = account_df.iloc[0]['weekly_pnl']
        max_loss_amt = total_cap * (loss_limit_pct / 100.0)
        # Adding because PnL is usually negative when losing
        remaining_buffer = max_loss_amt + weekly_pnl 
        if remaining_buffer < 0: remaining_buffer = 0
        
        st.info(f"🛡️ **Weekly Risk Guardrail:** You have **{format_currency(remaining_buffer)}** of loss allowance remaining this week.")
    else:
        total_cap = 100000 # Fallback
        remaining_buffer = 2500 # Fallback

    st.markdown("---")

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
            candidates = setups_df[
                (setups_df['week_start_dt'] == start_date) & 
                (setups_df['status'] == 'AWAITING_ENTRY')
            ]

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
                st.caption(f"Status: {selected_trade['trade_status']}")
                st.metric("Entry Price", format_currency(selected_trade['entry_price']))
                st.metric("Current SL", format_currency(selected_trade['stop_loss']))
                st.metric("Remaining Qty", selected_trade['remaining_quantity'])
        
        with c2:
            st.subheader("⚙️ Execution Console")
            
            if trade_source == 'setup':
                # --- 3B: Risk Warning Check ---
                # Calculate risk for this specific trade
                risk_per_share = selected_trade['entry_price'] - selected_trade['stop_loss']
                trade_risk = risk_per_share * selected_trade['quantity']
                
                st.info(f"💡 **Confirm Entry:** Current Price should be near {format_currency(selected_trade['entry_price'])}")
                st.write(f"📉 **Trade Risk:** {format_currency(trade_risk)}")

                if trade_risk > remaining_buffer:
                    st.warning(f"⚠️ **WARNING:** This trade's risk ({format_currency(trade_risk)}) exceeds your remaining weekly buffer ({format_currency(remaining_buffer)}). Executing this may breach your weekly loss limit.")

                col_ex1, col_ex2 = st.columns(2)
                with col_ex1:
                    if st.button("✅ CONFIRM ENTRY TAKEN", type="primary", use_container_width=True):
                        success = backend_engine.confirm_trade_entry(int(selected_trade['id']))
                        if success:
                            st.success(f"Trade for {selected_trade['ticker']} ACTIVATED!")
                            st.rerun()
                        else:
                            st.error("Entry failed. Check capital or logic.")
                with col_ex2:
                    if st.button("❌ SKIP TRADE", use_container_width=True):
                        skip_setup(int(selected_trade['id']))
                        st.warning(f"Trade for {selected_trade['ticker']} marked as SKIPPED.")
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
                        st.success("TP1 Booked. SL moved to Breakeven.")
                        st.rerun()

                elif action == "🎯 Book Target 2":
                    st.markdown("#### TP2 Execution (Final)")
                    exit_qty = st.number_input("Quantity Exited", min_value=1, max_value=current_qty, value=current_qty)
                    exit_price = st.number_input("Exit Price", value=float(selected_trade['target2']))
                    brokerage = st.number_input("Total Brokerage & Charges (₹)", min_value=0.0, step=10.0)
                    if st.button("Confirm TP2 Booking"):
                        backend_engine.confirm_exit_action(trade_id, "TP2", exit_price, exit_qty, brokerage)
                        st.success("TP2 Booked. Trade Closed.")
                        st.rerun()

                elif action == "❌ Stop Loss Exit":
                    st.markdown("#### Stop Loss Execution")
                    exit_qty = st.number_input("Quantity Exited", min_value=1, max_value=current_qty, value=current_qty)
                    exit_price = st.number_input("Exit Price", value=float(selected_trade['stop_loss']))
                    brokerage = st.number_input("Total Brokerage & Charges (₹)", min_value=0.0, step=10.0)
                    if st.button("Confirm SL Exit", type="primary"):
                        backend_engine.confirm_exit_action(trade_id, "SL", exit_price, exit_qty, brokerage)
                        st.error("Stop Loss Hit. Trade Closed.")
                        st.rerun()
                        
                elif action == "⏳ Time Exit":
                    st.markdown("#### Manual/Time Exit")
                    exit_qty = st.number_input("Quantity Exited", min_value=1, max_value=current_qty, value=current_qty)
                    exit_price = st.number_input("Exit Price", value=0.0)
                    brokerage = st.number_input("Total Brokerage & Charges (₹)", min_value=0.0, step=10.0)
                    if st.button("Confirm Manual Exit"):
                        backend_engine.confirm_exit_action(trade_id, "TIME_EXIT", exit_price, exit_qty, brokerage)
                        st.info("Trade Closed Manually.")
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
            st.caption("Virtual AI Fund Manager v1.4 (Capital Intel)")

        if page == "Dashboard":
            render_dashboard(account_state, active_trades)
        elif page == "Weekly Opportunities":
            render_opportunity_book(weekly_setups, account_state, active_trades)
        elif page == "Trade Execution":
            render_trade_execution(weekly_setups, active_trades)
        elif page == "Performance & Analytics":
            render_performance(all_trades, trade_actions, account_state)
