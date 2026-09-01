import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "finca360"


def _load_json(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Customer 360 dataset not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _group_by_customer(records):
    grouped = defaultdict(list)

    for record in records:
        customer_id = record.get("customer_id")
        if customer_id:
            grouped[customer_id].append(record)

    return dict(grouped)


# ------------------------------------------------------------------
# Load once at application startup
# ------------------------------------------------------------------

CUSTOMERS = _load_json("customers.json")
ACCOUNTS = _load_json("accounts.json")
TRANSACTIONS = _load_json("transactions.json")
DEVICES = _load_json("devices.json")
DEVICE_RELATIONSHIPS = _load_json("device_customer_relationships.json")
BENEFICIARIES = _load_json("beneficiaries.json")
AUTH_EVENTS = _load_json("authentication_events.json")
BEHAVIORAL_PROFILES = _load_json("behavioral_profiles.json")
RISK_FEATURES = _load_json("risk_features.json")


# ------------------------------------------------------------------
# Indexes
# ------------------------------------------------------------------

CUSTOMERS_BY_ID = {
    row["customer_id"]: row
    for row in CUSTOMERS
}

ACCOUNTS_BY_CUSTOMER = _group_by_customer(ACCOUNTS)
TRANSACTIONS_BY_CUSTOMER = _group_by_customer(TRANSACTIONS)
DEVICES_BY_CUSTOMER = _group_by_customer(DEVICES)
DEVICE_RELATIONSHIPS_BY_CUSTOMER = _group_by_customer(DEVICE_RELATIONSHIPS)
BENEFICIARIES_BY_CUSTOMER = _group_by_customer(BENEFICIARIES)
AUTH_BY_CUSTOMER = _group_by_customer(AUTH_EVENTS)
RISK_BY_CUSTOMER = _group_by_customer(RISK_FEATURES)

BEHAVIOR_BY_CUSTOMER = {
    row["customer_id"]: row
    for row in BEHAVIORAL_PROFILES
}

_device_customers = defaultdict(set)

for relationship in DEVICE_RELATIONSHIPS:
    device_id = relationship.get("device_id")
    customer_id = relationship.get("customer_id")

    if device_id and customer_id:
        _device_customers[device_id].add(customer_id)

DEVICE_CUSTOMER_COUNTS = {
    device_id: len(customer_ids)
    for device_id, customer_ids in _device_customers.items()
}


# ------------------------------------------------------------------
# Repository API
# ------------------------------------------------------------------

def get_all_customers():
    return CUSTOMERS


def get_customer(customer_id):
    return CUSTOMERS_BY_ID.get(customer_id)


def get_customer_accounts(customer_id):
    return ACCOUNTS_BY_CUSTOMER.get(customer_id, [])


def get_customer_transactions(customer_id):
    return TRANSACTIONS_BY_CUSTOMER.get(customer_id, [])


def get_customer_devices(customer_id):
    return DEVICES_BY_CUSTOMER.get(customer_id, [])


def get_customer_device_relationships(customer_id):
    return DEVICE_RELATIONSHIPS_BY_CUSTOMER.get(customer_id, [])


def get_customer_beneficiaries(customer_id):
    return BENEFICIARIES_BY_CUSTOMER.get(customer_id, [])


def get_customer_auth_events(customer_id):
    return AUTH_BY_CUSTOMER.get(customer_id, [])


def get_customer_behavior_profile(customer_id):
    return BEHAVIOR_BY_CUSTOMER.get(customer_id)


def get_customer_risk_features(customer_id):
    return RISK_BY_CUSTOMER.get(customer_id, [])


def get_device_customer_count(device_id):
    return DEVICE_CUSTOMER_COUNTS.get(device_id, 0)
