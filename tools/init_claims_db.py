# tools/init_claims_db.py
import sqlite3
import os

DB_PATH = "knowledge_base/claims_history.db"

def init_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Create Table (Added vehicle_vin)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_name TEXT,
            vehicle_vin TEXT,
            claim_date TEXT,
            claim_type TEXT,
            fault_status TEXT,
            payout_amount REAL
        )
    ''')

    # 2. Seed with Data matching your CSV and Policy
    # VINs are encrypted in database, ... means encrypted bits.
    data = [
        ("John Doe", "...456", "2024-05-15", "Rear-End", "At Fault", 1500.00),
        ("John Doe", "...456", "2025-10-10", "Rear-End", "At Fault", 500.50),
        ("John Doe", "...456", "2025-11-10", "T-Bone", "At Fault", 2200.50),        
        ("Jane Smith", "...123", "2024-01-20", "Sideswipe", "Not At Fault", 0.00),
        ("Alice Wonderland", "...789", "2023-08-05", "T-Bone", "At Fault", 12000.00),
        ("Bob Builder", "...321", "2024-03-15", "Front-End", "Not At Fault", 0.00),
    ]

    cursor.executemany('''
        INSERT INTO claims (driver_name, vehicle_vin, claim_date, claim_type, fault_status, payout_amount)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', data)

    conn.commit()
    print(f"Database initialized at {DB_PATH} with {len(data)} records.")
    conn.close()

if __name__ == "__main__":
    os.makedirs("knowledge_base", exist_ok=True)
    init_db()