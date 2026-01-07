import sqlite3
import pandas as pd
from datetime import datetime, date
import os

# ========================================================
# CONFIGURATION
# ========================================================
DB_NAME = "fund_manager.db"

# ========================================================
# DATABASE INITIALIZATION
# ========================================================

def init_db():
    """Initializes the SQLite database and creates tables if they don't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 1. Account State (Singleton)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS account_state (
        id INTEGER PRIMARY KEY,
        total_capital REAL,
        free_capital REAL,
        deployed_capital REAL,
        realized_pnl REAL,
        unrealized_pnl REAL,
        weekly_pnl REAL,
        weekly_loss_limit_pct REAL,
        trading_halted BOOLEAN,
        last_updated TEXT
    )
    ''')

    # 2. Weekly Setups (AI Brain Output)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS weekly_setups (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        week_start_date TEXT,
        week_end_date TEXT,
        ticker TEXT,
        setup_type TEXT,
        entry_price REAL,
        stop_loss REAL,
        target1 REAL,
        target2 REAL,
        quantity INTEGER,
        risk_amount REAL,
        expected_move_pct REAL,
        est_days_tp1 INTEGER,
        est_days_tp2 INTEGER,
        confidence_score REAL,
        first_detected_date TEXT,
        status TEXT,
        UNIQUE(ticker, week_start_date)
    )
    ''')

    # 3. Trades (Active & Closed)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        setup_id INTEGER,
        ticker TEXT,
        entry_price REAL,
        stop_loss REAL,
        target1 REAL,
        target2 REAL,
        initial_quantity INTEGER,
        remaining_quantity INTEGER,
        trade_status TEXT, 
        entry_date TEXT,
        exit_date TEXT,
        FOREIGN KEY(setup_id) REFERENCES weekly_setups(id)
    )
    ''')

    # 4. Trade Actions (Audit Trail & Reconciliation)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trade_actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_id INTEGER,
        action_type TEXT,
        quantity INTEGER,
        execution_price REAL,
        gross_pnl REAL,
        brokerage_charges REAL,
        net_pnl REAL,
        action_date TEXT,
        FOREIGN KEY(trade_id) REFERENCES trades(id)
    )
    ''')

    conn.commit()
    conn.close()
    print(f"Database {DB_NAME} initialized successfully.")

# ========================================================
# CORE FUNCTIONS
# ========================================================

def get_db_connection():
    """Returns a connection to the SQLite database."""
    return sqlite3.connect(DB_NAME)

def initialize_account(initial_capital: float, weekly_loss_limit_pct: float = 2.5):
    """Sets up the initial capital and risk parameters. Resets state."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if account already exists
    cursor.execute("SELECT COUNT(*) FROM account_state")
    exists = cursor.fetchone()[0]
    
    today_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if exists == 0:
        cursor.execute('''
        INSERT INTO account_state (
            id, total_capital, free_capital, deployed_capital, 
            realized_pnl, unrealized_pnl, weekly_pnl, 
            weekly_loss_limit_pct, trading_halted, last_updated
        ) VALUES (1, ?, ?, 0.0, 0.0, 0.0, 0.0, ?, 0, ?)
        ''', (initial_capital, initial_capital, weekly_loss_limit_pct, today_str))
        print(f"Account initialized with ₹{initial_capital}")
    else:
        # Optional: Logic to hard reset if needed, for now we skip if exists
        print("Account state already exists. Skipping initialization.")
    
    conn.commit()
    conn.close()

def load_ai_weekly_setups(csv_path: str):
    """
    Loads AI Brain output CSV into the weekly_setups table.
    Expects CSV columns to match database schema or mapping logic provided here.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # Assuming CSV has headers matching these variables. 
    # Adjust column names in the df[...] calls if your AI brain uses different headers.
    count = 0
    for _, row in df.iterrows():
        try:
            # Upsert logic (Ignore if exists to prevent duplicates for same week)
            cursor.execute('''
            INSERT OR IGNORE INTO weekly_setups (
                week_start_date, week_end_date, ticker, setup_type,
                entry_price, stop_loss, target1, target2,
                quantity, risk_amount, expected_move_pct,
                est_days_tp1, est_days_tp2, confidence_score,
                first_detected_date, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row.get('week_start_date', today_str), # Fallback if missing
                row.get('week_end_date', '2099-12-31'),
                row['ticker'],
                row['setup_type'],
                row['entry'],
                row['stop_loss'],
                row['target1'],
                row['target2'],
                int(row['quantity']),
                row['risk_amount'],
                row.get('expected_move_pct', 0),
                int(row.get('est_days_tp1', 0)),
                int(row.get('est_days_tp2', 0)),
                row.get('confidence_score', 0),
                today_str,
                "AWAITING_ENTRY"
            ))
            count += 1
        except Exception as e:
            print(f"Error inserting row for {row.get('ticker')}: {e}")

    conn.commit()
    conn.close()
    print(f"Processed {count} setups from CSV.")

def confirm_trade_entry(setup_id: int):
    """
    Confirms a trade entry based on a setup.
    Moves money from Free Capital to Deployed Capital.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Check if trading is halted
    cursor.execute("SELECT trading_halted FROM account_state WHERE id=1")
    halted = cursor.fetchone()[0]
    if halted:
        print("TRADING HALTED: Weekly loss limit reached.")
        conn.close()
        return False

    # 2. Get Setup Details
    cursor.execute("SELECT * FROM weekly_setups WHERE id=?", (setup_id,))
    setup = cursor.fetchone()
    if not setup:
        print("Setup not found.")
        conn.close()
        return False
        
    # Column mapping for tuple 'setup'
    # 0:id, 3:ticker, 5:entry, 6:sl, 7:tp1, 8:tp2, 9:qty
    ticker = setup[3]
    entry_price = setup[5]
    stop_loss = setup[6]
    tp1 = setup[7]
    tp2 = setup[8]
    qty = setup[9]
    cost = entry_price * qty

    # 3. Check Capital
    cursor.execute("SELECT free_capital FROM account_state WHERE id=1")
    free_cap = cursor.fetchone()[0]
    
    if free_cap < cost:
        print("Insufficient Free Capital.")
        conn.close()
        return False

    # 4. Insert into Trades
    today_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
    INSERT INTO trades (
        setup_id, ticker, entry_price, stop_loss, target1, target2, 
        initial_quantity, remaining_quantity, trade_status, entry_date
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (setup_id, ticker, entry_price, stop_loss, tp1, tp2, qty, qty, "ACTIVE", today_str))
    
    trade_id = cursor.lastrowid

    # 5. Log Action
    cursor.execute('''
    INSERT INTO trade_actions (
        trade_id, action_type, quantity, execution_price, 
        gross_pnl, brokerage_charges, net_pnl, action_date
    ) VALUES (?, 'ENTRY', ?, ?, 0, 0, 0, ?)
    ''', (trade_id, qty, entry_price, today_str))

    # 6. Update Setup Status
    cursor.execute("UPDATE weekly_setups SET status='ACTIVE' WHERE id=?", (setup_id,))

    # 7. Update Account State
    cursor.execute('''
    UPDATE account_state 
    SET free_capital = free_capital - ?,
        deployed_capital = deployed_capital + ?
    WHERE id=1
    ''', (cost, cost))

    conn.commit()
    conn.close()
    print(f"Trade Activated: {ticker} @ {entry_price}")
    return True

def confirm_exit_action(trade_id: int, action_type: str, exit_price: float, exited_qty: int, brokerage: float):
    """
    Handles TP1, TP2, SL, or TIME_EXIT.
    Reconciles PnL and updates Capital.
    
    action_type options: 'TP1', 'TP2', 'SL', 'TIME_EXIT'
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get Trade Details
    cursor.execute("SELECT ticker, entry_price, remaining_quantity, stop_loss FROM trades WHERE id=?", (trade_id,))
    trade = cursor.fetchone()
    if not trade:
        print("Trade not found.")
        conn.close()
        return

    ticker, entry_price, current_qty, current_sl = trade
    
    if exited_qty > current_qty:
        print("Error: Exited quantity cannot be greater than remaining quantity.")
        conn.close()
        return

    # Calculate PnL
    gross_pnl = (exit_price - entry_price) * exited_qty
    net_pnl = gross_pnl - brokerage
    capital_released = (entry_price * exited_qty) # The original capital put in for these shares

    today_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log Action
    cursor.execute('''
    INSERT INTO trade_actions (
        trade_id, action_type, quantity, execution_price, 
        gross_pnl, brokerage_charges, net_pnl, action_date
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (trade_id, action_type, exited_qty, exit_price, gross_pnl, brokerage, net_pnl, today_str))

    # Update Trade Status & Quantity
    new_qty = current_qty - exited_qty
    new_status = "ACTIVE"
    if new_qty == 0:
        new_status = "CLOSED"
        cursor.execute("UPDATE trades SET exit_date=? WHERE id=?", (today_str, trade_id))

    # SPECIAL LOGIC FOR TP1: Move SL to Cost
    if action_type == 'TP1' and new_qty > 0:
        cursor.execute("UPDATE trades SET stop_loss=? WHERE id=?", (entry_price, trade_id))
        print(f"TP1 Hit. Stop Loss for {ticker} moved to Breakeven ({entry_price}).")

    cursor.execute('''
    UPDATE trades 
    SET remaining_quantity=?, trade_status=? 
    WHERE id=?
    ''', (new_qty, new_status, trade_id))

    # Update Account State
    # Free Capital increases by (Capital Released + Net PnL)
    # Deployed Capital decreases by Capital Released
    # Realized PnL increases by Net PnL
    # Weekly PnL increases by Net PnL
    
    cursor.execute('''
    UPDATE account_state 
    SET free_capital = free_capital + ? + ?,
        deployed_capital = deployed_capital - ?,
        realized_pnl = realized_pnl + ?,
        weekly_pnl = weekly_pnl + ?,
        last_updated = ?
    WHERE id=1
    ''', (capital_released, net_pnl, capital_released, net_pnl, net_pnl, today_str))

    conn.commit()
    
    # Check Weekly Loss Cap
    enforce_weekly_loss_cap(conn)
    
    conn.close()
    print(f"Processed {action_type} for {ticker}. Net PnL: {net_pnl}")

def enforce_weekly_loss_cap(conn):
    """Checks if weekly PnL exceeds loss limit and halts trading if so."""
    cursor = conn.cursor()
    cursor.execute("SELECT total_capital, weekly_pnl, weekly_loss_limit_pct FROM account_state WHERE id=1")
    row = cursor.fetchone()
    if not row: return

    total_cap, weekly_pnl, limit_pct = row
    max_loss_amount = total_cap * (limit_pct / 100.0) * -1 # Make negative

    if weekly_pnl <= max_loss_amount:
        cursor.execute("UPDATE account_state SET trading_halted=1 WHERE id=1")
        conn.commit()
        print(f"⚠️ WEEKLY LOSS LIMIT HIT ({weekly_pnl}). TRADING HALTED.")

# ========================================================
# GETTERS FOR UI
# ========================================================

def get_account_state():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM account_state", conn)
    conn.close()
    return df

def get_active_trades():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM trades WHERE trade_status='ACTIVE'", conn)
    conn.close()
    return df

def get_weekly_setups():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM weekly_setups", conn)
    conn.close()
    return df

# ========================================================
# EXECUTION ENTRY POINT
# ========================================================
if __name__ == "__main__":
    init_db()
    # Example initialization (Uncomment to test)
    # initialize_account(100000)
    print("Backend Engine is Ready.")