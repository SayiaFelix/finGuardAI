# FINCA Customer 360 Backend Additions

Copy the contents of this package into the root of `Finca_backend`.

It adds:

- `modules/customer360/repository.py`
- `modules/customer360/service.py`
- `modules/customer360/routes.py`
- `data/finca360/*.json`
- `fin_guard_ai_customer360_patch.txt`

## Existing fraud module

No fraud scoring, ML, transaction, alert, case, SQLite or authentication code is replaced.

## New API routes

### Customer table
GET `/finca/v1/customers?page=1&size=20`

Optional filters:

- `search`
- `segment`
- `risk_profile`
- `status`

### Customer detail
GET `/finca/v1/customers/C00001/360`

This returns:

- customer identity/header
- KPI values
- transaction summary
- 30-day transaction trend
- behaviour profile
- account list
- realistic device list
- recent transactions

## Currency

JSON uses ISO code `KES`. The frontend should format `KES` as `Ksh`.

## Local startup

Continue using the existing command:

`py .\fin_guard_ai.py`

and the existing Flask port 5001.
