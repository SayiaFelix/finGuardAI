-- Migration: create customers table and backfill customer_id
-- Usage: sqlite3 database/fraudsentinel.db < database/fraudsentinel_migration.sql
-- Backup your DB before running: cp database/fraudsentinel.db database/fraudsentinel.db.bak

PRAGMA foreign_keys = OFF;
BEGIN TRANSACTION;

ALTER TABLE transactions ADD COLUMN channel VARCHAR DEFAULT 'MOBILE_BANKING';
ALTER TABLE transactions ADD COLUMN direction VARCHAR DEFAULT 'OUTGOING';

-- Create customers table if missing
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    full_name TEXT NOT NULL,
    email TEXT UNIQUE,
    phone_number TEXT,
    national_id TEXT UNIQUE,
    date_of_birth TEXT,
    registration_date TEXT,
    customer_segment TEXT,
    risk_profile TEXT,
    account_status TEXT,
    branch_code TEXT,
    home_city TEXT,
    currency TEXT
);

-- Backfill transactions.customer_id from JSON customer_info
UPDATE transactions
SET customer_id = json_extract(customer_info, '$.customer_id')
WHERE (customer_id IS NULL OR TRIM(customer_id) = '')
  AND customer_info IS NOT NULL
  AND json_valid(customer_info) = 1;

-- Insert discovered customer_ids into customers (placeholders)
INSERT OR IGNORE INTO customers (customer_id, first_name, last_name, full_name)
SELECT DISTINCT json_extract(customer_info, '$.customer_id') AS cid, cid, cid, cid
FROM transactions
WHERE json_valid(customer_info) = 1
  AND json_extract(customer_info, '$.customer_id') IS NOT NULL
  AND TRIM(json_extract(customer_info, '$.customer_id')) <> '';

-- Backfill alerts.customer_id from linked transaction
UPDATE alerts
SET customer_id = (SELECT t.customer_id FROM transactions t WHERE t.id = alerts.transaction_id)
WHERE (customer_id IS NULL OR TRIM(customer_id) = '')
  AND transaction_id IS NOT NULL;

-- Backfill cases.customer_id from linked alert
UPDATE cases
SET customer_id = (SELECT a.customer_id FROM alerts a WHERE a.id = cases.alert_id)
WHERE (customer_id IS NULL OR TRIM(customer_id) = '')
  AND alert_id IS NOT NULL;

-- Indexes used by the application
CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_alerts_customer_id ON alerts(customer_id);
CREATE INDEX IF NOT EXISTS idx_cases_customer_id ON cases(customer_id);
CREATE INDEX IF NOT EXISTS idx_customers_home_city ON customers(home_city);
CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);
CREATE INDEX IF NOT EXISTS idx_customers_national_id ON customers(national_id);

COMMIT;
PRAGMA foreign_keys = ON;

-- Optional: enrich `customers` from external source using a separate script.
