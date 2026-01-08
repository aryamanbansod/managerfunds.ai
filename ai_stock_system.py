# ==============================================================================
# 🎯 DAILY TRADR FUND MANAGER ENGINE - PRODUCTION EDITION (FINAL)
# ==============================================================================
# ARCHITECTURE:
# 1. 🛡️ SELF-HEALING GUARD: Prevents redundant scans (20 min cooldown).
# 2. 🌌 UNIVERSE v2: Nifty Mid/Small/Next50 + Strict Hygiene (Price > 50, ADTV > 10Cr).
# 3. 🛑 SPEED GATE: Kills slow/heavy stocks.
# 4. ⚡️ UNIFIED EXPLOSIVE ENGINE: 5 Patterns (VCP, Flag, NR7, 44MA, Accum).
# 5. 🧱 CMP EXECUTION GATE: Set & Forget entries.
# 6. 📅 WEEKLY MEMORY: Tracks book Mon-Fri.
# ==============================================================================

import os
import sys
import datetime
import math
import shutil
import requests
import io
import pandas as pd
import numpy as np
import yfinance as yf

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Using relative paths for production safety (Works on GitHub Actions / Local)
BASE_PATH = "data"

CONFIG = {
    # System Paths
    "dirs": {
        "prices":    f"{BASE_PATH}/raw_prices",
        "signals":   f"{BASE_PATH}/signals",
        "combined":  f"{BASE_PATH}/combined",
        "dashboard": f"{BASE_PATH}/dashboard",
        "weekly":    f"{BASE_PATH}/weekly",
        "guard":     f"{BASE_PATH}/system"  # Corrected system path for guard file
    },

    # Self-Healing Guard Settings
    "scan_cooldown_minutes": 20,

    # Strategy Settings
    "history_years": 2,              
    "min_liquidity_volume": 500000,
    
    # Trend Context
    "ema_fast": 21,                  
    "ema_slow": 50,
    "ma_institutional": 44, 

    # Volatility Gate
    "atr_lookback": 14,
    "atr_reference": 60,             
    "vol_expansion_threshold": 1.2,  

    # Unified Engine Logic
    "avg_volume_window": 20,         
    "thrust_impulse_min": 0.15,      
    
    # FINANCIAL FILTERS
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

# Ensure folders exist
for p in CONFIG["dirs"].values():
    os.makedirs(p, exist_ok=True)

# ==============================================================================
# 2. SELF-HEALING GUARD
# ==============================================================================
# FIXED PATH: Stores system metadata in data/system/ to keep hygiene
GUARD_FILE_PATH = f"{CONFIG['dirs']['guard']}/last_successful_scan.txt"

def run_scan_guard():
    """
    Checks if a scan is needed based on the last successful timestamp.
    Returns: True (Run Scan), False (Skip Scan)
    """
    if not os.path.exists(GUARD_FILE_PATH):
        print("   🆕 First run detected. Starting scan...")
        return True

    try:
        with open(GUARD_FILE_PATH, 'r') as f:
            last_ts_str = f.read().strip()
        
        if not last_ts_str: return True

        last_scan_time = datetime.datetime.fromisoformat(last_ts_str)
        # Assuming system time is UTC
        now_utc = datetime.datetime.utcnow()
        age_minutes = (now_utc - last_scan_time).total_seconds() / 60.0
        
        if age_minutes < CONFIG["scan_cooldown_minutes"]:
            print(f"   🛑 SCAN SKIPPED: Last scan was {age_minutes:.1f} min ago.")
            print(f"      (Cooldown is {CONFIG['scan_cooldown_minutes']} min. Market data likely unchanged.)")
            return False
        else:
            print(f"   🟢 SCAN APPROVED: Last scan was {age_minutes:.1f} min ago (Stale).")
            return True
    except Exception as e:
        print(f"   ⚠️ Guard error ({e}). Running for safety.")
        return True

def update_scan_timestamp():
    """
    Updates the guard file with current UTC time.
    Call this ONLY at the very end of a successful script run.
    """
    try:
        with open(GUARD_FILE_PATH, 'w') as f:
            f.write(datetime.datetime.utcnow().isoformat())
        print("   ✅ Guard Timestamp Updated.")
    except Exception as e:
        print(f"   ❌ Failed to update guard: {e}")

# EXECUTE GUARD
if not run_scan_guard():
    sys.exit(0)

# ==============================================================================
# 3. UNIVERSE v2 GENERATOR
# ==============================================================================
def fetch_nse_index_tickers(index_name):
    urls = {
        'next50': "https://nsearchives.nseindia.com/content/indices/ind_niftynext50list.csv",
        'midcap150': "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
        'smallcap250': "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv"
    }
    url = urls.get(index_name)
    if not url: return []
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
            return [f"{x}.NS" for x in df['Symbol'].tolist()]
    except: pass
    return []

def apply_hygiene_filters(tickers, min_adtv_cr=10, min_price=50):
    print(f"   🧹 Hygiene Check on {len(tickers)} stocks...")
    clean_tickers = []
    try:
        # Download 1mo data for ADTV
        data = yf.download(tickers, period="1mo", group_by='ticker', progress=False, auto_adjust=True)
        for ticker in tickers:
            try:
                if len(tickers) == 1: df = data
                else: df = data[ticker]
                
                if df.empty: continue
                last_price = df['Close'].iloc[-1]
                if last_price < min_price: continue
                
                # ADTV Check
                turnover = df['Close'] * df['Volume']
                adtv_cr = turnover.rolling(20).mean().iloc[-1] / 10000000
                
                if adtv_cr >= min_adtv_cr:
                    clean_tickers.append((ticker, adtv_cr))
            except: continue
    except Exception as e: 
        print(f"   ⚠️ Hygiene Error: {e}")
        return []
        
    return clean_tickers

def get_stock_universe():
    print("\n🌍 GENERATING UNIVERSE v2...")
    next50 = fetch_nse_index_tickers('next50')
    midcap150 = fetch_nse_index_tickers('midcap150')
    smallcap250 = fetch_nse_index_tickers('smallcap250')
    
    if not next50 and not midcap150:
        print("   ⚠️ NSE fetch failed. Using fallback list.")
        return ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "SBIN.NS"]

    # Core Universe
    core_raw = list(set(next50 + midcap150))
    core_clean = [x[0] for x in apply_hygiene_filters(core_raw)]
    
    # Smallcap Top 100
    small_clean_tuples = apply_hygiene_filters(smallcap250)
    small_clean_tuples.sort(key=lambda x: x[1], reverse=True)
    small_final = [x[0] for x in small_clean_tuples[:100]]
    
    final_univ = list(set(core_clean + small_final))
    print(f"   🚀 Universe Ready: {len(final_univ)} Tickers")
    return final_univ

UNIVERSE_TICKERS = get_stock_universe()

# ==============================================================================
# 4. MARKET DATA UPDATE
# ==============================================================================
def update_data(tickers):
    print("\nStep 4: Updating Market Data...")
    if not tickers: return pd.DataFrame()
    all_dfs = []
    start_date = (datetime.datetime.today() - datetime.timedelta(days=CONFIG['history_years']*365)).strftime('%Y-%m-%d')
    
    try:
        # Batch download
        data = yf.download(tickers, start=start_date, group_by='ticker', progress=False, auto_adjust=True)
        for ticker in tickers:
            try:
                df = data[ticker].copy() if len(tickers) > 1 else data.copy()
                if df.empty: continue
                df.dropna(how='all', inplace=True)
                df['Ticker'] = ticker
                df.reset_index(inplace=True)
                all_dfs.append(df)
            except: pass
    except: pass
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined['Date'] = pd.to_datetime(combined['Date'])
        return combined
    return pd.DataFrame()

df_prices = update_data(UNIVERSE_TICKERS)

# ==============================================================================
# 5. UNIFIED EXPLOSIVE ENGINE SCANNER
# ==============================================================================
print("\nStep 5: Running Scanner...")
signals = pd.DataFrame()

if not df_prices.empty:
    df = df_prices.copy()
    df.sort_values(['Ticker', 'Date'], inplace=True)
    g = df.groupby('Ticker')

    # Indicators
    df['ema_fast'] = g['Close'].transform(lambda x: x.ewm(span=CONFIG['ema_fast']).mean())
    df['ema_inst'] = g['Close'].transform(lambda x: x.ewm(span=CONFIG['ma_institutional']).mean())
    df['avg_vol'] = g['Volume'].transform(lambda x: x.rolling(CONFIG['avg_volume_window']).mean())
    
    df['prev_close'] = g['Close'].shift(1)
    df['tr'] = df[['High','Low','prev_close']].apply(lambda x: max(x['High']-x['Low'], abs(x['High']-x['prev_close']), abs(x['Low']-x['prev_close'])), axis=1)
    df['atr_short'] = g['tr'].transform(lambda x: x.rolling(CONFIG['atr_lookback']).mean())
    
    # Speed Checks
    df['roi_20d'] = g['Close'].transform(lambda x: x.pct_change(20))
    df['range_pct'] = (df['High'] - df['Low']) / df['prev_close']
    
    # --- PATTERNS ---
    # 1. VCP
    df['rolling_std'] = g['Close'].transform(lambda x: x.rolling(20).std())
    df['bb_width'] = (4 * df['rolling_std']) / df['ema_fast']
    df['bbw_avg'] = g['bb_width'].transform(lambda x: x.rolling(40).mean())
    mask_vcp = (df['bb_width'] < df['bbw_avg']*0.7) & (df['Volume'] < df['avg_vol']) & (df['Close'] > df['ema_inst'])
    
    # 2. Flag
    mask_flag = (df['roi_20d'].shift(5) > CONFIG['thrust_impulse_min']) & (df['range_pct'].rolling(5).mean() < 0.02) & (df['Close'] > df['ema_fast'])
    
    # 3. NR7
    min_range_7 = g['range_pct'].transform(lambda x: x.rolling(7).min())
    mask_nr7 = (df['range_pct'] == min_range_7) & (df['Close'] > df['ema_inst'])
    
    # 4. 44MA
    mask_44ma = (df['Low'] <= df['ema_inst']*1.01) & (df['Close'] > df['ema_inst']) & (df['Close'] > df['Open'])
    
    # 5. Accumulation
    mask_accum = (df['Close'].pct_change().abs() < 0.01) & (df['Volume'] > df['avg_vol']*1.5) & (df['Close'] > df['ema_inst'])

    # --- GATES ---
    # Speed Gate
    df['daily_atr_pct'] = (df['atr_short']/df['Close'])*100
    df['move_60d'] = ((df['Close'].rolling(60).max()-df['Close'].rolling(60).min())/df['Close'].rolling(60).min())*100
    mask_speed = (df['daily_atr_pct'] >= CONFIG['min_daily_atr_pct']) & (df['move_60d'] >= CONFIG['min_60d_move_pct'])
    mask_liq = g['Volume'].transform(lambda x: x.rolling(20).mean()) >= CONFIG['min_liquidity_volume']

    # Assign Setup
    df['SETUP_TYPE'] = "NONE"
    df.loc[mask_vcp, 'SETUP_TYPE'] = "VCP_COMPRESSION"
    df.loc[mask_flag, 'SETUP_TYPE'] = "HIGH_TIGHT_FLAG"
    df.loc[mask_nr7, 'SETUP_TYPE'] = "NR7_CONTRACTION"
    df.loc[mask_44ma, 'SETUP_TYPE'] = "44MA_SUPPORT_BUY"
    df.loc[mask_accum, 'SETUP_TYPE'] = "ACCUMULATION_SPIKE"

    # Financials
    df['swing_low'] = g['Low'].transform(lambda x: x.rolling(5).min())
    df['impulse_height'] = g['High'].transform(lambda x: x.rolling(20).max()) - g['Low'].transform(lambda x: x.rolling(20).min())
    df['projected_target'] = df['Close'] + df['impulse_height']
    df['base_risk_pct'] = ((df['Close'] - df['swing_low'])/df['Close'])*100
    df['expected_move_pct'] = ((df['projected_target'] - df['Close'])/df['Close'])*100
    
    mask_setup = df['SETUP_TYPE'] != "NONE"
    mask_upside = df['expected_move_pct'] >= CONFIG['min_expected_move_pct']
    mask_risk = df['base_risk_pct'] <= CONFIG['max_base_risk_pct']
    
    signals = df[mask_liq & mask_speed & mask_setup & mask_upside & mask_risk].copy()
    
    # Scoring
    signals['super_score'] = 80
    signals.loc[signals['SETUP_TYPE'] == "HIGH_TIGHT_FLAG", 'super_score'] += 15
    signals.loc[signals['SETUP_TYPE'] == "VCP_COMPRESSION", 'super_score'] += 10
    
    print(f"   ✅ Candidates Found: {len(signals)}")

# ==============================================================================
# 6. EXECUTION & DASHBOARD
# ==============================================================================
print("\nStep 6: Processing Execution & Dashboard...")

TODAY_DATE = datetime.date.today()
TODAY_STR = TODAY_DATE.strftime("%Y-%m-%d")

# Weekly Logic
idx = (TODAY_DATE.weekday() + 1) % 7
week_start_date = TODAY_DATE - datetime.timedelta(days=TODAY_DATE.weekday())
week_end_date = week_start_date + datetime.timedelta(days=4)
week_start_str = week_start_date.strftime("%Y-%m-%d")
week_end_str = week_end_date.strftime("%Y-%m-%d")

todays_candidates = []

if not signals.empty:
    latest_date = signals['Date'].max()
    fresh = signals[signals['Date'] >= (latest_date - datetime.timedelta(days=2))].copy()
    fresh = fresh.sort_values('super_score', ascending=False).drop_duplicates('Ticker')
    
    latest_prices = df_prices.sort_values('Date').groupby('Ticker').tail(1).set_index('Ticker')['Close']
    
    for idx, row in fresh.iterrows():
        ticker = row['Ticker']
        if ticker not in latest_prices: continue
        cmp = latest_prices[ticker]
        
        # 1. Market Cap Gate
        try:
            t = yf.Ticker(ticker)
            mcap = t.info.get('marketCap', 0)
            if (mcap / 10000000) > CONFIG['max_market_cap_crore']: continue
        except: pass

        # 2. CMP Gate (1.5% Proximity)
        signal_entry = row['Close']
        if abs(cmp - signal_entry)/signal_entry*100 > CONFIG['max_entry_slippage_pct']: continue
        
        # 3. Execution Risk Gate
        stop_loss = row['swing_low']
        exec_risk_pct = ((cmp - stop_loss)/cmp)*100
        target2 = row['projected_target']
        exec_rr = (target2 - cmp)/(cmp - stop_loss) if (cmp - stop_loss) > 0 else 0
        
        if exec_risk_pct > CONFIG['max_execution_risk_pct']: continue
        if exec_rr < CONFIG['min_execution_rr']: continue
        
        target1 = cmp + ((cmp - stop_loss)*2.0)
        
        # 4. Fundamental Check (Smart Money)
        fund_status = "NEUTRAL"
        try:
            t_info = yf.Ticker(ticker).info
            if t_info.get('debtToEquity', 0) > 500: fund_status = "WEAK"
            if t_info.get('profitMargins', 0) < -0.20: fund_status = "WEAK"
            if t_info.get('heldPercentInstitutions', 0) * 100 > 20: fund_status = "STRONG"
        except: pass
        
        if fund_status == "WEAK": continue

        todays_candidates.append({
            'Week_Start': week_start_str, 'Week_End': week_end_str,
            'Ticker': ticker, 'Setup': row['SETUP_TYPE'],
            'Entry': cmp, 'StopLoss': stop_loss, 'Target1': target1, 'Target2': target2,
            'Risk_Pct': exec_risk_pct, 'Fund_Status': fund_status,
            'Status': "ACTIVE", 'Detected_Date': TODAY_STR
        })

# --- WEEKLY BOOK UPDATE ---
weekly_csv = f"{CONFIG['dirs']['weekly']}/weekly_setups.csv"
weekly_df = pd.read_csv(weekly_csv) if os.path.exists(weekly_csv) else pd.DataFrame()

if todays_candidates:
    new_df = pd.DataFrame(todays_candidates)
    if not weekly_df.empty:
        existing = weekly_df['Ticker'].tolist()
        new_df = new_df[~new_df['Ticker'].isin(existing)]
    weekly_df = pd.concat([weekly_df, new_df], ignore_index=True)
    weekly_df.to_csv(weekly_csv, index=False)
    
    # Save Daily Snapshot
    pd.DataFrame(todays_candidates).to_csv(f"{CONFIG['dirs']['dashboard']}/daily_plan_{TODAY_STR}.csv", index=False)

# --- DISPLAY ---
print("\n" + "="*60)
print(f"🎯 MORNING DASHBOARD (WEEKLY VIEW) - {TODAY_STR}")
print("="*60)
print(f"📅 Week: {week_start_str} to {week_end_str}")

print("\n✨ TODAY'S FRESH SETUPS:")
if todays_candidates:
    print(pd.DataFrame(todays_candidates)[['Ticker', 'Setup', 'Entry', 'StopLoss', 'Target1']].to_string(index=False))
else:
    print("   (No new setups matching Operator criteria today)")

print("\n📚 ACTIVE WEEKLY BOOK:")
if not weekly_df.empty:
    print(weekly_df[['Ticker', 'Setup', 'Entry', 'Status']].to_string(index=False))
else:
    print("   (Book empty for this week)")
print("="*60)

# ==============================================================================
# 7. SUCCESS MARKER
# ==============================================================================
update_scan_timestamp()
print("🏁 SCRIPT COMPLETE")