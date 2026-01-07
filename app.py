import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date, timedelta
import backend_engine  # Phase 1 Logic
import sqlite3

# ========================================================
# CONFIGURATION & CONSTANTS
# ========================================================
st.set_page_config(
    page_title="AI Fund Manager | Pro Desk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

CONFIDENCE_THRESHOLD = 80.0 

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
# DATA LOADING
# ========================================================
def load_data():
    try:
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
# VIEW 1: DASHBOARD (Phase 2A)
# ========================================================
def render_dashboard(account_df, trades_df):
    st.title("🏦 Capital & Portfolio Overview")
    st.markdown("---")

    if account_df.empty:
        st.warning("⚠️ Account not initialized.")
        return

    account = account_df.iloc[0]
    total_cap = account['total_capital']
    free_cap = account['free_capital']
    deployed_cap = account['deployed_capital']
    realized_pnl = account['realized_pnl']
    weekly_pnl = account['weekly_pnl']
    is_halted = account['trading_halted']
    loss_limit_pct = account['weekly_loss_limit_pct']

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("💰 Total Capital", format_currency(total_cap))
    with col2: st.metric("🟢 Free Capital", format_currency(free_cap))
    with col3: st.metric("🔴 Deployed Capital", format_currency(deployed_cap))
    with col4:
        status_text = "HALTED 🛑" if is_halted else "ENABLED ✅"
        st.metric("Trading Status", status_text)

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
# VIEW 2: WEEKLY OPPORTUNITY BOOK (Phase 2B)
# ========================================================
def render_opportunity_book(setups_df, account_df, trades_df):
    st.title("📅 Weekly Opportunity Book")
    start_date, end_date = get_current_week_dates()
    st.markdown(f"#### **Current Trading Week:** {start_date.strftime('%d %b')} — {end_date.strftime('%d %b %Y')}")
    st.markdown("---")

    if setups_df.empty:
        st.info("No setups detected for this week yet.")
        return

    setups_df['confidence_score'] = pd.to_numeric(setups_df['confidence_score'], errors='coerce').fillna(0)
    
    def determine_grade(score):
        return "🚀 TRADR-GRADE" if score >= CONFIDENCE_THRESHOLD else "⚠️ ALTERNATIVE"

    setups_df['Grade'] = setups_df['confidence_score'].apply(determine_grade)

    tradr_count = len(setups_df[setups_df['Grade'] == "🚀 TRADR-GRADE"])
    alt_count = len(setups_df[setups_df['Grade'] == "⚠️ ALTERNATIVE"])
    active_trade_count = len(trades_df)
    is_halted = account_df.iloc[0]['trading_halted'] if not account_df.empty else False

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1: st.metric("🚀 TRADR-Grade Setups", tradr_count)
    with kpi2: st.metric("⚠️ Alternative Setups", alt_count)
    with kpi3: st.metric("🟡 Active Trades", active_trade_count)
    with kpi4: 
        status_label = "⛔ HALTED" if is_halted else "✅ ENABLED"
        st.metric("Trading Status", status_label)

    st.markdown("---")
    st.subheader("🔍 Identified Opportunities (Mon–Fri)")

    cols_to_show = ['ticker', 'Grade', 'setup_type', 'entry_price', 'stop_loss', 'target1', 'target2', 'quantity', 'risk_amount', 'expected_move_pct', 'confidence_score', 'status']
    available_cols = [c for c in cols_to_show if c in setups_df.columns]
    
    st.dataframe(
        setups_df[available_cols], use_container_width=True, hide_index=True,
        column_config={
            "entry_price": st.column_config.NumberColumn("Entry", format="₹%.2f"),
            "stop_loss": st.column_config.NumberColumn("SL", format="₹%.2f"),
            "target1": st.column_config.NumberColumn("TP1", format="₹%.2f"),
            "target2": st.column_config.NumberColumn("TP2", format="₹%.2f"),
            "confidence_score": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%d"),
        }
    )

# ========================================================
# VIEW 3: TRADE EXECUTION & MANAGEMENT (Phase 2C)
# ========================================================
def render_trade_execution(setups_df, trades_df):
    st.title("⚡ Trade Execution Desk")
    st.caption("Confirm entries, book profits, and manage stops. Strict discipline required.")
    st.markdown("---")

    mode = st.radio("Select List:", ["New Setups (Awaiting Entry)", "Active Trades (Open)"], horizontal=True)
    selected_trade = None
    trade_source = None 

    if mode == "New Setups (Awaiting Entry)":
        if setups_df.empty:
            st.info("No setups available.")
        else:
            candidates = setups_df[setups_df['status'] == 'AWAITING_ENTRY']
            if candidates.empty:
                st.info("No setups currently awaiting entry.")
            else:
                options = candidates['ticker'].tolist()
                selected_ticker = st.selectbox("Select Ticker to Act On:", options)
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
                st.caption(f"Setup: {selected_trade['setup_type']}")
                st.metric("Entry Trigger", format_currency(selected_trade['entry_price']))
                st.metric("Stop Loss", format_currency(selected_trade['stop_loss']))
                st.metric("Target 1", format_currency(selected_trade['target1']))
                st.metric("Quantity", selected_trade['quantity'])
            else:
                st.caption(f"Status: {selected_trade['trade_status']}")
                st.metric("Entry Price", format_currency(selected_trade['entry_price']))
                st.metric("Current SL", format_currency(selected_trade['stop_loss']))
                st.metric("Remaining Qty", selected_trade['remaining_quantity'])
        
        with c2:
            st.subheader("⚙️ Execution Console")
            
            if trade_source == 'setup':
                st.info("💡 **Action Required:** Has the price touched the entry level?")
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
                    st.warning("Ensure you have exited positions in your broker app first.")
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
# VIEW 4: PERFORMANCE & DISCIPLINE (Phase 2D)
# ========================================================
def render_performance(all_trades, trade_actions, account_df):
    st.title("🏆 Performance & Discipline")
    st.markdown("### The Truth Screen")
    st.caption("Metrics derived from executed trade data only. No theoretical paper trades.")
    st.markdown("---")

    if all_trades.empty or trade_actions.empty:
        st.info("Insufficient data to generate performance metrics. Execute some trades first.")
        return

    # --- 1. PERFORMANCE SNAPSHOT ---
    st.subheader("1. System Performance Snapshot")
    
    # Filter closed trades/actions for stats
    closed_actions = trade_actions[trade_actions['action_type'].isin(['TP1', 'TP2', 'SL', 'TIME_EXIT'])]
    
    total_trades = len(all_trades)
    winning_trades = closed_actions[closed_actions['net_pnl'] > 0]
    losing_trades = closed_actions[closed_actions['net_pnl'] <= 0]
    
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    total_closed = win_count + loss_count
    
    win_rate = (win_count / total_closed * 100) if total_closed > 0 else 0.0
    avg_win = winning_trades['net_pnl'].mean() if not winning_trades.empty else 0.0
    avg_loss = losing_trades['net_pnl'].mean() if not losing_trades.empty else 0.0
    
    # Expectancy (R-Multiple approximation or simpler PnL expectancy)
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    
    # Equity & Drawdown
    initial_capital = 100000.0 # Default fallback, ideally stored
    if not account_df.empty:
        # Reconstruct initial from current total - realized pnl (approx)
        # Better: Assume user input correct initial cap in init_db
        # For display, we use total_capital from DB
        current_equity = account_df.iloc[0]['total_capital']
    else:
        current_equity = 0.0

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Total Trades", total_trades)
    with m2: st.metric("Win Rate", f"{win_rate:.1f}%")
    with m3: st.metric("Avg Win", format_currency(avg_win))
    with m4: st.metric("Avg Loss", format_currency(avg_loss))
    
    m5, m6, m7, m8 = st.columns(4)
    with m5: st.metric("Expectancy (₹/Trade)", format_currency(expectancy))
    with m6: st.metric("Current Equity", format_currency(current_equity))
    with m7: st.metric("Net Profit", format_currency(account_df.iloc[0]['realized_pnl']))
    with m8: st.metric("Profit Factor", f"{abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "N/A")

    st.markdown("---")

    # --- 2. EQUITY CURVE ---
    st.subheader("2. Equity Curve")
    
    # Construct Equity Series
    # Start with initial capital (Current - Realized PnL)
    start_cap = current_equity - account_df.iloc[0]['realized_pnl']
    
    equity_data = []
    equity_data.append({'Date': 'Start', 'Equity': start_cap})
    
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

    # --- 3. DISCIPLINE SCOREBOARD ---
    st.subheader("3. Discipline Scoreboard")
    
    # Logic to approximate discipline
    # SL Respected: Did we exit near SL price? (Approximate check)
    # Runners: Trades with TP1 actions
    
    tp1_actions = trade_actions[trade_actions['action_type'] == 'TP1']
    runner_count = len(tp1_actions)
    runner_pct = (runner_count / total_closed * 100) if total_closed > 0 else 0
    
    d1, d2, d3 = st.columns(3)
    with d1: st.metric("Trades Converted to Runners", f"{runner_count} ({runner_pct:.1f}%)")
    with d2: st.metric("Brokerage Paid (Total)", format_currency(trade_actions['brokerage_charges'].sum()))
    with d3: 
        # Weekly Overtrade check
        week_start, week_end = get_current_week_dates()
        trades_this_week = all_trades[pd.to_datetime(all_trades['entry_date']).dt.date >= week_start]
        count_week = len(trades_this_week)
        st.metric("Trades This Week", f"{count_week} / 3 (Max)")

# ========================================================
# MAIN APP CONTROLLER
# ========================================================

# Load Global Data
account_state, active_trades, weekly_setups, all_trades, trade_actions = load_data()

# Sidebar Navigation
with st.sidebar:
    st.header("🧭 Fund Desk")
    page = st.radio("Navigate", ["Dashboard", "Weekly Opportunities", "Trade Execution", "Performance & Analytics"])
    
    st.divider()
    st.caption("Virtual AI Fund Manager v1.0")

# Route to Page
if page == "Dashboard":
    render_dashboard(account_state, active_trades)
elif page == "Weekly Opportunities":
    render_opportunity_book(weekly_setups, account_state, active_trades)
elif page == "Trade Execution":
    render_trade_execution(weekly_setups, active_trades)
elif page == "Performance & Analytics":
    render_performance(all_trades, trade_actions, account_state)