# ==============================================================================
# 🎯 DAILY TRADR FUND MANAGER ENGINE - WEEKLY MEMORY EDITION (FINAL)
# ==============================================================================
#
# ARCHITECTURE:
# 1. 🛑 SPEED GATE: Kills slow/heavy stocks.
# 2. ⚡️ UNIFIED EXPLOSIVE ENGINE: Scans 5 patterns.
# 3. 🧱 CMP EXECUTION GATE: Forces "Set & Forget" entries.
# 4. 🏦 SMART MONEY CONFIRMATION: Optional Fundamental Veto.
# 5. 📅 WEEKLY MEMORY (NEW): Tracks setups Mon-Fri.
# 6. ⏰ TIME-AWARE EXECUTION: Market Hours vs Off-Hours Logic.
#
# ==============================================================================

import pandas as pd
import numpy as np
import os
import datetime
import math
import shutil
import yfinance as yf

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# Configure Pandas Display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

# --- USER CONFIGURATION ---
CONFIG = {
    "base_drive_path": ".", 
    "history_years": 2,
    "min_liquidity_volume": 500000,

    # Trend Context
    "ema_fast": 21,
    "ema_slow": 50,
    "ma_institutional": 44,

    # Volatility Gate
    "atr_lookback": 14,
    "atr_reference": 60,
    
    # --- CALIBRATION TUNING (UPDATED) ---
    "vcp_tightness_threshold": 0.8,       # Relaxed from 0.7 (Allow 80% of avg width)
    "flag_consolidation_threshold": 0.025, # Relaxed from 0.02 (Allow 2.5% daily range)
    "accumulation_vol_threshold": 1.25,    # Sensitized from 1.5 (Catch 1.25x vol spikes)
    "thrust_impulse_min": 0.15,            # Keep 15% thrust requirement

    # Unified Engine Logic
    "avg_volume_window": 20,

    # FINANCIAL FILTERS (Base)
    "min_expected_move_pct": 15.0,
    "max_base_risk_pct": 8.0,
    "min_rr_ratio": 2.5,
    "max_holding_days": 20,

    # SPEED GATE
    "min_daily_atr_pct": 2.5,
    "min_60d_move_pct": 12.0,
    "max_market_cap_crore": 300000,

    # CMP EXECUTION GATE
    "max_entry_slippage_pct": 1.5,
    "max_execution_risk_pct": 4.0,
    "min_execution_rr": 2.5,

    # Capital Management
    "initial_capital_fallback": 100000.0,
    "risk_per_trade_pct": 2.0,
    "max_positions": 5,
}

# --- UNIVERSE ---
UNIVERSE_TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
    "AXISBANK.NS", "HINDUNILVR.NS", "TATAMOTORS.NS", "BAJFINANCE.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "M&M.NS", "ADANIENT.NS", "TITAN.NS",
    "ULTRACEMCO.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS", "TATASTEEL.NS",
    "JSWSTEEL.NS", "COALINDIA.NS", "ADANIPORTS.NS", "BPCL.NS", "BAJAJFINSV.NS",
    "EICHERMOT.NS", "DRREDDY.NS", "HCLTECH.NS", "ASIANPAINT.NS", "GRASIM.NS",
    "WIPRO.NS", "CIPLA.NS", "TECHM.NS", "HINDALCO.NS", "SBILIFE.NS",
    "BRITANNIA.NS", "INDUSINDBK.NS", "TATACONSUM.NS", "DIVISLAB.NS",
    "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "UPL.NS", "VEDL.NS"
]

# ==============================================================================
# 2. TIME & ENVIRONMENT UTILS
# ==============================================================================

def get_market_status():
    """
    Determines if the Indian Equity Market is currently open.
    IST Timezone: UTC + 5:30
    Market Hours: 09:15 to 15:30
    Weekdays: Mon(0) to Fri(4)
    """
    # Define IST Timezone
    IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    now_ist = datetime.datetime.now(IST)
    
    # Define Market Start and End times for today
    market_start = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    
    is_weekday = now_ist.weekday() < 5  # 0=Mon, 4=Fri
    is_time_open = market_start <= now_ist <= market_end
    
    if is_weekday and is_time_open:
        return "MARKET_OPEN", now_ist
    else:
        return "MARKET_CLOSED", now_ist

def setup_environment():
    """Sets up folders and weekly context."""
    print("\nStep 2: Setting up environment...")
    
    base_path = CONFIG["base_drive_path"]
    today_date = datetime.date.today()
    today_str = today_date.strftime("%Y-%m-%d")

    # --- WEEKLY LOGIC ---
    week_start_date = today_date - datetime.timedelta(days=today_date.weekday())
    week_end_date = week_start_date + datetime.timedelta(days=4)
    week_start_str = week_start_date.strftime("%Y-%m-%d")
    week_end_str = week_end_date.strftime("%Y-%m-%d")

    # Dynamic Weekly Folder
    year_str = str(today_date.year)
    month_str = today_date.strftime("%m")
    week_folder_name = f"Week_{week_start_str}_to_{week_end_str}"
    weekly_path = os.path.join(base_path, "weekly_dashboard", year_str, month_str, week_folder_name)

    dirs = {
        "prices":    os.path.join(base_path, "data", "raw_prices"),
        "signals":   os.path.join(base_path, "data", "signals"),
        "inst":      os.path.join(base_path, "data", "institutional"),
        "combined":  os.path.join(base_path, "data", "combined"),
        "backtest":  os.path.join(base_path, "backtests"),
        "dashboard": os.path.join(base_path, "dashboard", today_str),
        "weekly":    weekly_path
    }

    for k, path in dirs.items():
        os.makedirs(path, exist_ok=True)

    print(f"✅ Folder structure verified.")
    print(f"   📅 Current Week: {week_start_str} to {week_end_str}")
    
    return dirs, today_str, week_start_str, week_end_str

def update_data(save_dir):
    """Downloads fresh market data."""
    print("\nStep 3: Updating Market Data...")
    
    all_dfs = []
    # Fetch ample history for indicators
    start_date = (datetime.datetime.today() - datetime.timedelta(days=CONFIG['history_years']*365)).strftime('%Y-%m-%d')
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')

    print(f"   ⏳ Downloading {len(UNIVERSE_TICKERS)} tickers via yfinance...")

    try:
        data = yf.download(UNIVERSE_TICKERS, start=start_date, end=end_date, group_by='ticker', progress=False, auto_adjust=True)

        for ticker in UNIVERSE_TICKERS:
            try:
                if len(UNIVERSE_TICKERS) == 1: 
                    df = data.copy()
                else: 
                    try:
                        df = data[ticker].copy()
                    except KeyError:
                        continue

                if df.empty: continue
                df.dropna(how='all', inplace=True)
                if df.empty: continue

                df['Ticker'] = ticker
                df.reset_index(inplace=True)
                all_dfs.append(df)
            except Exception: pass
    except Exception as e:
        print(f"   ❌ Critical Download Error: {e}")
        return pd.DataFrame()

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        
        save_path = os.path.join(save_dir, "all_prices_combined.csv")
        combined_df.to_csv(save_path, index=False)
        print(f"   ✅ Data Updated: {len(combined_df)} rows.")
        return combined_df
    else:
        print("   ⚠️ No data downloaded.")
        return pd.DataFrame()

# ==============================================================================
# 3. CORE STRATEGY LOGIC (CALIBRATED)
# ==============================================================================

def run_unified_engine(df_prices, dirs):
    """Runs the 5-pattern scanner logic with calibrated sensitivity."""
    print("\nStep 4: Running Unified Explosive Compression Engine (Calibrated)...")

    if df_prices.empty:
        print("   ❌ No price data.")
        return pd.DataFrame()

    df = df_prices.copy()
    df.sort_values(['Ticker', 'Date'], inplace=True)
    g = df.groupby('Ticker')

    # --- A. INDICATORS ---
    df['ema_fast'] = g['Close'].transform(lambda x: x.ewm(span=CONFIG['ema_fast']).mean())
    df['ema_inst'] = g['Close'].transform(lambda x: x.ewm(span=CONFIG['ma_institutional']).mean())
    df['avg_vol'] = g['Volume'].transform(lambda x: x.rolling(CONFIG['avg_volume_window']).mean())

    df['prev_close'] = g['Close'].shift(1)
    df['tr'] = df[['High', 'Low', 'prev_close']].apply(
        lambda x: max(x['High']-x['Low'], abs(x['High']-x['prev_close']), abs(x['Low']-x['prev_close'])), 
        axis=1
    )
    df['atr_short'] = g['tr'].transform(lambda x: x.rolling(CONFIG['atr_lookback']).mean())

    df['roi_10d'] = g['Close'].transform(lambda x: x.pct_change(10))
    df['roi_20d'] = g['Close'].transform(lambda x: x.pct_change(20))
    df['range_pct'] = (df['High'] - df['Low']) / df['prev_close']

    # --- B. UNIFIED DETECTION (5 PATTERNS) ---
    
    # 1. VCP (Slightly Relaxed)
    df['rolling_std'] = g['Close'].transform(lambda x: x.rolling(20).std())
    df['bb_width'] = (4 * df['rolling_std']) / df['ema_fast']
    df['bbw_avg'] = g['bb_width'].transform(lambda x: x.rolling(40).mean())
    
    # Use calibrated threshold from CONFIG
    is_tight = df['bb_width'] < (df['bbw_avg'] * CONFIG['vcp_tightness_threshold'])
    is_dry = df['Volume'] < df['avg_vol']
    mask_vcp = is_tight & is_dry & (df['Close'] > df['ema_inst'])

    # 2. HIGH TIGHT FLAG (Consolidation Relaxed)
    has_power = df['roi_20d'].shift(5) > CONFIG['thrust_impulse_min']
    
    # Use calibrated threshold from CONFIG
    is_consolidating = df['range_pct'].rolling(5).mean() < CONFIG['flag_consolidation_threshold']
    mask_flag = has_power & is_consolidating & (df['Close'] > df['ema_fast'])

    # 3. NR7 (Standard)
    df['daily_range'] = df['High'] - df['Low']
    min_range_7 = g['daily_range'].transform(lambda x: x.rolling(7).min())
    is_nr7 = df['daily_range'] == min_range_7
    mask_nr7 = is_nr7 & (df['Close'] > df['ema_inst']) & (df['Volume'] < df['avg_vol'])

    # 4. 44-MA RIDE (Standard)
    touched_ma = (df['Low'] <= df['ema_inst'] * 1.01) & (df['Low'] >= df['ema_inst'] * 0.99)
    held_ma = (df['Close'] > df['ema_inst']) & (df['Close'] > df['Open'])
    mask_44ma = touched_ma & held_ma & (df['Volume'] > df['avg_vol'])

    # 5. DELIVERY ACCUMULATION (Sensitized Volume)
    price_flat = df['Close'].pct_change().abs() < 0.01
    
    # Use calibrated threshold from CONFIG (Catching 1.25x vs 1.5x)
    vol_spike = df['Volume'] > (df['avg_vol'] * CONFIG['accumulation_vol_threshold'])
    mask_accum = price_flat & vol_spike & (df['Close'] > df['ema_inst'])

    # --- C. SPEED GATE ---
    df['daily_atr_pct'] = (df['atr_short'] / df['Close']) * 100
    df['move_60d'] = ((df['Close'].rolling(60).max() - df['Close'].rolling(60).min()) / df['Close'].rolling(60).min()) * 100
    mask_speed_gate = (
        (df['daily_atr_pct'] >= CONFIG['min_daily_atr_pct']) &
        (df['move_60d'] >= CONFIG['min_60d_move_pct'])
    )
    mask_liq = g['Volume'].transform(lambda x: x.rolling(20).mean()) >= CONFIG['min_liquidity_volume']

    # --- D. SELECTION ---
    df['SETUP_TYPE'] = "NONE"
    df.loc[mask_vcp, 'SETUP_TYPE'] = "VCP_COMPRESSION"
    df.loc[mask_flag, 'SETUP_TYPE'] = "HIGH_TIGHT_FLAG"
    df.loc[mask_nr7, 'SETUP_TYPE'] = "NR7_CONTRACTION"
    df.loc[mask_44ma, 'SETUP_TYPE'] = "44MA_SUPPORT_BUY"
    df.loc[mask_accum, 'SETUP_TYPE'] = "ACCUMULATION_SPIKE"

    df['swing_low'] = g['Low'].transform(lambda x: x.rolling(5).min())
    df['impulse_height'] = g['High'].transform(lambda x: x.rolling(20).max()) - g['Low'].transform(lambda x: x.rolling(20).min())
    df['projected_target'] = df['Close'] + df['impulse_height']
    df['expected_move_pct'] = ((df['projected_target'] - df['Close']) / df['Close']) * 100
    df['base_risk_pct'] = ((df['Close'] - df['swing_low']) / df['Close']) * 100

    mask_setup = df['SETUP_TYPE'] != "NONE"
    mask_upside = df['expected_move_pct'] >= CONFIG['min_expected_move_pct']
    mask_risk = df['base_risk_pct'] <= CONFIG['max_base_risk_pct']

    valid_signal = mask_liq & mask_speed_gate & mask_setup & mask_upside & mask_risk

    signals = df[valid_signal].copy()

    # Determine "Days to Target" (Estimation)
    signals['days_to_target'] = CONFIG['max_holding_days']

    signals['super_score'] = 80
    signals.loc[signals['SETUP_TYPE'] == "HIGH_TIGHT_FLAG", 'super_score'] += 15
    signals.loc[signals['SETUP_TYPE'] == "VCP_COMPRESSION", 'super_score'] += 10

    signals_path = os.path.join(dirs['signals'], "smart_money_tradr_signals.csv")
    signals.to_csv(signals_path, index=False)
    print(f"   ✅ Engine Complete. Found {len(signals)} Explosive Candidates.")
    
    return signals

def smart_money_fundamental_check(ticker):
    """Optional fundamental check using yfinance info."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        debt_eq = info.get('debtToEquity', 0)
        margins = info.get('profitMargins', 0)
        inst_held = info.get('heldPercentInstitutions', 0) * 100
        
        # Simple veto logic
        if debt_eq > 500: return "WEAK", f"Debt {debt_eq}%"
        if margins < -0.20: return "WEAK", "Losses"
        if inst_held > 20: return "STRONG", f"Inst {inst_held:.1f}%"
    except: 
        return "NEUTRAL", "No Data"
    return "NEUTRAL", "No Red Flags"

def process_execution_gates(signals, df_prices, dirs, today_str, week_start_str, week_end_str, market_status):
    """
    Filters signals through execution gates and updates weekly memory.
    """
    print(f"\nStep 5: Running Execution Gates in [{market_status}] Mode...")
    
    todays_candidates = []
    
    if not signals.empty:
        latest_date = signals['Date'].max()
        fresh_signals = signals[signals['Date'] >= (latest_date - datetime.timedelta(days=2))].copy()
        fresh_signals = fresh_signals.sort_values('super_score', ascending=False).drop_duplicates('Ticker')

        latest_prices = df_prices.sort_values('Date').groupby('Ticker').tail(1).set_index('Ticker')['Close']

        for idx, row in fresh_signals.iterrows():
            ticker = row['Ticker']

            # Market Cap Check
            try:
                t = yf.Ticker(ticker)
                mcap = t.info.get('marketCap', 0)
                if (mcap / 10000000) > CONFIG['max_market_cap_crore']: 
                    continue
            except: 
                pass

            # Fundamental Check
            fund_status, fund_reason = smart_money_fundamental_check(ticker)
            if fund_status == "WEAK": 
                continue

            # CMP Gate
            if ticker not in latest_prices: 
                continue
            
            cmp = latest_prices[ticker]
            signal_entry = row['Close']

            if (abs(cmp - signal_entry) / signal_entry * 100) > CONFIG['max_entry_slippage_pct']: 
                continue

            stop_loss = row['swing_low']
            exec_risk_pct = ((cmp - stop_loss) / cmp) * 100
            target2 = row['projected_target']
            
            # Risk/Reward Check
            denominator = (cmp - stop_loss)
            exec_rr = (target2 - cmp) / denominator if denominator > 0 else 0

            if exec_risk_pct > CONFIG['max_execution_risk_pct']: 
                continue
            if exec_rr < CONFIG['min_execution_rr']: 
                continue

            target1 = cmp + (denominator * 2.0)

            todays_candidates.append({
                'Week_Start': week_start_str,
                'Week_End': week_end_str,
                'Ticker': ticker,
                'Setup': row['SETUP_TYPE'],
                'Entry': cmp,
                'StopLoss': stop_loss,
                'Target1': target1,
                'Target2': target2,
                'Exp_Move_Pct': ((target2 - cmp)/cmp)*100,
                'Risk_Pct': exec_risk_pct,
                'Est_Days': row['days_to_target'],
                'Fund_Status': fund_status,
                'Detected_Date': today_str,
                'Status': "AWAITING_ENTRY"
            })

    # --- MODE SPECIFIC LOGIC ---
    
    # Load existing book
    weekly_csv_path = os.path.join(dirs['weekly'], "weekly_setups.csv")
    if os.path.exists(weekly_csv_path):
        weekly_book_df = pd.read_csv(weekly_csv_path)
    else:
        weekly_book_df = pd.DataFrame()

    if market_status == "MARKET_CLOSED":
        print("🌙 OFF-MARKET MODE: Scanning for opportunities (Watchlist Update Only).")
        print(f"   found {len(todays_candidates)} potential candidates (Not Saving to CSV).")
        return todays_candidates, weekly_book_df

    elif market_status == "MARKET_OPEN":
        print("🚀 MARKET OPEN: Full Execution Mode Active.")
        
        # Append new ones if not already in book for this week
        if todays_candidates:
            new_df = pd.DataFrame(todays_candidates)
            if not weekly_book_df.empty:
                # Avoid duplicates based on Ticker
                existing_tickers = weekly_book_df['Ticker'].tolist()
                new_df = new_df[~new_df['Ticker'].isin(existing_tickers)]

            if not new_df.empty:
                weekly_book_df = pd.concat([weekly_book_df, new_df], ignore_index=True)
                print(f"   ✅ Added {len(new_df)} NEW setups to Weekly Book.")

        # Save Master Book
        if not weekly_book_df.empty:
            weekly_book_df.to_csv(weekly_csv_path, index=False)
            print(f"   📘 Weekly Book Persisted: {len(weekly_book_df)} active setups.")
        
        return todays_candidates, weekly_book_df

def print_dashboard(todays_candidates, weekly_book_df, dirs, today_str, week_start_str, week_end_str, market_status):
    """Prints the final summary to console."""
    print("\nStep 6: Generating Weekly Fund Manager Dashboard...")

    current_capital = CONFIG['initial_capital_fallback']
    state_path = os.path.join(dirs['combined'], "account_state.csv")
    
    if os.path.exists(state_path):
        try: 
            current_capital = float(pd.read_csv(state_path).iloc[-1]['total_capital'])
        except: 
            pass

    print("\n" + "="*60)
    print(f"🎯 MORNING DASHBOARD (WEEKLY VIEW) - {today_str}")
    print(f"⏰ STATUS: {market_status}")
    print("="*60)
    print(f"💰 Total Capital:  ₹{current_capital:,.2f}")
    print(f"📅 Trading Week:   {week_start_str} to {week_end_str}")

    # 1. Show Today's Fresh Finds
    print("\n✨ NEW SETUPS DETECTED TODAY:")
    if todays_candidates:
        fresh_df = pd.DataFrame(todays_candidates)
        cols = ['Ticker', 'Setup', 'Entry', 'StopLoss', 'Target1', 'Target2', 'Fund_Status']

        disp = fresh_df.copy()
        disp['Entry'] = disp['Entry'].map('{:,.2f}'.format)
        disp['StopLoss'] = disp['StopLoss'].map('{:,.2f}'.format)
        disp['Target1'] = disp['Target1'].map('{:,.2f}'.format)
        disp['Target2'] = disp['Target2'].map('{:,.2f}'.format)
        print(disp[cols].to_string(index=False))
        
        if market_status == "MARKET_CLOSED":
            print("\n   ⚠️ (OFF-MARKET: These were NOT saved to DB to prevent noise)")
    else:
        print("   (No new setups today)")

    # 2. Show Active Weekly Book
    print("\n📚 CURRENT WEEK ACTIVE BOOK (MON-FRI):")
    if not weekly_book_df.empty:
        cols = ['Ticker', 'Setup', 'Entry', 'StopLoss', 'Target1', 'Target2', 'Status', 'Detected_Date']
        # Check if columns exist
        available_cols = [c for c in cols if c in weekly_book_df.columns]
        
        disp = weekly_book_df.copy()
        if 'Entry' in disp.columns: disp['Entry'] = disp['Entry'].map('{:,.2f}'.format)
        if 'StopLoss' in disp.columns: disp['StopLoss'] = disp['StopLoss'].map('{:,.2f}'.format)
        if 'Target1' in disp.columns: disp['Target1'] = disp['Target1'].map('{:,.2f}'.format)
        if 'Target2' in disp.columns: disp['Target2'] = disp['Target2'].map('{:,.2f}'.format)
        print(disp[available_cols].to_string(index=False))
    else:
        print("   (Book is empty for this week)")

    print("\n" + "="*60)
    print("🏁 EXECUTION COMPLETE")

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    print("✅ Libraries loaded & Configuration set.")
    
    # 0. Check Time & Mode
    status, ist_time = get_market_status()
    print(f"🕒 Current IST Time: {ist_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🚦 Execution Mode: {status}")

    # 1. Setup
    dirs, today_str, w_start, w_end = setup_environment()
    
    # 2. Update Data
    # We download data regardless of mode to keep cache fresh
    df_prices = update_data(save_dir=dirs['prices'])
    
    # 3. Run Strategy (Calibrated)
    signals = run_unified_engine(df_prices, dirs)
    
    # 4. Process Gates & Memory (Mode Dependent)
    todays_candidates, weekly_book_df = process_execution_gates(
        signals, df_prices, dirs, today_str, w_start, w_end, market_status=status
    )
    
    # 5. Dashboard Output
    print_dashboard(todays_candidates, weekly_book_df, dirs, today_str, w_start, w_end, market_status=status)
