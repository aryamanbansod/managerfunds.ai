# ==============================================================================
# 🌌 UNIVERSE v2: SHADOW MODE IMPLEMENTATION
# ==============================================================================
# OBJECTIVE:
# Define a targeted universe of Midcaps, Next 50, and Liquid Smallcaps.
# STRICTLY enforce hygiene (Price > 50, ADTV > 10Cr) to remove junk.
# ==============================================================================

import requests
import pandas as pd
import io
import yfinance as yf
import time

def fetch_nse_index_tickers(index_name):
    """
    Fetches official ticker list for a given NSE index.
    index_name examples: 'niftymidcap150', 'niftynext50', 'niftysmallcap250'
    """
    # URLs for NSE Indices (Standard Archival Links)
    urls = {
        'next50': "https://nsearchives.nseindia.com/content/indices/ind_niftynext50list.csv",
        'midcap150': "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
        'smallcap250': "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv"
    }
    
    url = urls.get(index_name)
    if not url: return []

    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
            # NSE CSVs usually have a 'Symbol' column
            return [f"{x}.NS" for x in df['Symbol'].tolist()]
        else:
            print(f"   ⚠️ Failed to fetch {index_name} (Status: {response.status_code})")
            return []
    except Exception as e:
        print(f"   ⚠️ Error fetching {index_name}: {e}")
        return []

def apply_hygiene_filters(tickers, min_adtv_cr=10, min_price=50):
    """
    Downloads recent data to check Liquidity (ADTV) and Price.
    Returns a CLEAN list of tickers.
    """
    print(f"   🧹 Applying Hygiene Filters to {len(tickers)} stocks...")
    print(f"      Criteria: ADTV ≥ {min_adtv_cr} Cr, Price ≥ {min_price}")
    
    clean_tickers = []
    
    # Download last 1 month data (sufficient for ADTV)
    # Batch download is faster
    try:
        data = yf.download(tickers, period="1mo", group_by='ticker', progress=False, auto_adjust=True)
    except Exception as e:
        print(f"   ❌ Critical Error in Hygiene Check: {e}")
        return []

    for ticker in tickers:
        try:
            # Handle single vs multi-index DataFrame structure
            if len(tickers) == 1:
                df = data
            else:
                df = data[ticker]
            
            if df.empty: continue
            
            # 1. Price Check (Last Close)
            last_price = df['Close'].iloc[-1]
            if last_price < min_price: continue 

            # 2. ADTV Check (Average Daily Traded Value - 20 Days)
            # ADTV = Average(Close * Volume)
            df['Turnover'] = df['Close'] * df['Volume']
            # Convert to Crores
            adtv_cr = df['Turnover'].rolling(20).mean().iloc[-1] / 10000000 
            
            if adtv_cr < min_adtv_cr: continue
            
            # 3. Circuit/Illiquidity Check (Proxy)
            # If High == Low consistently (Circuit locked), usually Volume is tiny, so ADTV catches it.
            # We can also check if ADTV is valid.
            if pd.isna(adtv_cr): continue

            # If passed, add tuple (Ticker, ADTV) for sorting later
            clean_tickers.append((ticker, adtv_cr))
            
        except Exception:
            continue
            
    return clean_tickers

def get_stock_universe():
    print("\n🌍 GENERATING UNIVERSE v2 (Shadow Mode)...")
    
    # 1. Fetch Raw Constituents
    next50 = fetch_nse_index_tickers('next50')
    midcap150 = fetch_nse_index_tickers('midcap150')
    smallcap250 = fetch_nse_index_tickers('smallcap250')
    
    if not next50 and not midcap150:
        print("   ❌ Failed to fetch indices. Using fallback list.")
        # FALLBACK: Use your old manual list if NSE fetch fails entirely
        return [
            "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
            "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS"
        ]

    print(f"   📥 Raw Counts: Next50={len(next50)}, Mid150={len(midcap150)}, Small250={len(smallcap250)}")

    # 2. Process CORE Universe (Next 50 + Mid 150)
    # We want ALL of these, provided they pass basic hygiene
    core_raw = list(set(next50 + midcap150))
    core_clean_tuples = apply_hygiene_filters(core_raw, min_adtv_cr=10, min_price=50)
    core_final = [x[0] for x in core_clean_tuples]
    
    print(f"   ✅ Core Universe (Mid+Next50) Cleaned: {len(core_final)} stocks")

    # 3. Process BETA BOOSTER (Smallcap 250)
    # We only want the TOP 100 by Liquidity (ADTV)
    small_clean_tuples = apply_hygiene_filters(smallcap250, min_adtv_cr=10, min_price=50)
    
    # Sort by ADTV (descending) and take Top 100
    small_clean_tuples.sort(key=lambda x: x[1], reverse=True)
    small_final = [x[0] for x in small_clean_tuples[:100]]
    
    print(f"   ✅ Smallcap Universe Filtered (Top 100 by Liquidity): {len(small_final)} stocks")

    # 4. Combine & Deduplicate
    full_universe = list(set(core_final + small_final))
    print(f"   🚀 UNIVERSE v2 READY: {len(full_universe)} Total Tickers")
    print("   ------------------------------------------------")
    
    return full_universe

# ==============================================================================
# USAGE INSTRUCTION:
# Replace: UNIVERSE_TICKERS = [...] 
# With:    UNIVERSE_TICKERS = get_stock_universe()
# ==============================================================================