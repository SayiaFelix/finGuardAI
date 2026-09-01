from collections import Counter, defaultdict
from datetime import datetime, timedelta

from . import repository


def _parse_datetime(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return None


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_average(values):
    clean = [
        _safe_float(value)
        for value in values
        if value is not None
    ]

    return sum(clean) / len(clean) if clean else 0.0


def _display_risk(customer):
    """
    Customer-master risk profile for the Customer 360 list/header.
    This is intentionally separate from transaction fraud scores.
    """
    return customer.get("risk_profile", "LOW")


# ------------------------------------------------------------------
# Customer table
# ------------------------------------------------------------------

def list_customers(
    page=1,
    size=20,
    search=None,
    segment=None,
    risk_profile=None,
    status=None
):
    customers = repository.get_all_customers()

    search_value = search.strip().lower() if search else None
    results = []

    for customer in customers:
        if segment and customer.get("customer_segment") != segment:
            continue

        if risk_profile and customer.get("risk_profile") != risk_profile:
            continue

        if status and customer.get("account_status") != status:
            continue

        if search_value:
            searchable = " ".join(
                [
                    str(customer.get("customer_id", "")),
                    str(customer.get("first_name", "")),
                    str(customer.get("last_name", "")),
                    str(customer.get("full_name", "")),
                    str(customer.get("phone_number", "")),
                    str(customer.get("email", "")),
                ]
            ).lower()

            if search_value not in searchable:
                continue

        results.append(
            {
                "customer_id": customer.get("customer_id"),
                "full_name": customer.get("full_name")
                    or f'{customer.get("first_name", "")} {customer.get("last_name", "")}'.strip(),
                "customer_segment": customer.get("customer_segment"),
                "home_city": customer.get("home_city"),
                "account_status": customer.get("account_status"),
                "risk_profile": _display_risk(customer),
                "branch_code": customer.get("branch_code"),
                "currency": customer.get("currency", "KES"),
            }
        )

    total = len(results)
    start = (page - 1) * size
    end = start + size
    total_pages = (total + size - 1) // size if size else 0

    return {
        "customers": results[start:end],
        "pagination": {
            "page": page,
            "size": size,
            "total": total,
            "total_pages": total_pages,
            "has_more": end < total,
        },
    }


# ------------------------------------------------------------------
# Customer detail helpers
# ------------------------------------------------------------------

def _build_transaction_trend(transactions, reference_date, days=30):
    if not reference_date:
        return []

    start_date = (reference_date - timedelta(days=days - 1)).date()

    daily = defaultdict(
        lambda: {
            "transaction_count": 0,
            "total_amount": 0.0,
        }
    )

    for transaction in transactions:
        tx_date = _parse_datetime(
            transaction.get("transaction_date")
        )

        if not tx_date or tx_date.date() < start_date:
            continue

        day = tx_date.date().isoformat()
        daily[day]["transaction_count"] += 1
        daily[day]["total_amount"] += _safe_float(
            transaction.get("amount")
        )

    result = []

    for offset in range(days):
        current_date = (
            start_date + timedelta(days=offset)
        ).isoformat()

        row = daily[current_date]

        result.append(
            {
                "date": current_date,
                "transaction_count":
                    row["transaction_count"],
                "total_amount":
                    round(row["total_amount"], 2),
            }
        )

    return result


def _clean_recent_transaction(transaction):
    return {
        "transaction_id":
            transaction.get("transaction_id"),
        "transaction_date":
            transaction.get("transaction_date"),
        "channel":
            transaction.get("channel"),
        "transaction_type":
            transaction.get("transaction_type"),
        "amount":
            _safe_float(transaction.get("amount")),
        "currency":
            transaction.get("currency", "KES"),
        "location_city":
            transaction.get("location_city"),
        "status":
            transaction.get("status"),
    }


# ------------------------------------------------------------------
# Customer 360 details
# ------------------------------------------------------------------

def build_customer_360(customer_id):
    customer = repository.get_customer(customer_id)

    if not customer:
        return None

    accounts = repository.get_customer_accounts(
        customer_id
    )
    transactions = repository.get_customer_transactions(
        customer_id
    )
    devices = repository.get_customer_devices(
        customer_id
    )
    auth_events = repository.get_customer_auth_events(
        customer_id
    )
    behavior = repository.get_customer_behavior_profile(
        customer_id
    ) or {}

    transaction_dates = [
        _parse_datetime(tx.get("transaction_date"))
        for tx in transactions
    ]
    transaction_dates = [
        value for value in transaction_dates
        if value is not None
    ]

    latest_transaction_date = (
        max(transaction_dates)
        if transaction_dates
        else None
    )

    cutoff_30d = (
        latest_transaction_date - timedelta(days=29)
        if latest_transaction_date
        else None
    )

    transactions_30d = []

    if cutoff_30d:
        transactions_30d = [
            transaction
            for transaction in transactions
            if (
                _parse_datetime(
                    transaction.get("transaction_date")
                )
                and _parse_datetime(
                    transaction.get("transaction_date")
                ) >= cutoff_30d
            )
        ]

    all_amounts = [
        _safe_float(tx.get("amount"))
        for tx in transactions
    ]

    account_balance = sum(
        _safe_float(account.get("current_balance"))
        for account in accounts
    )

    known_devices = len(
        {
            device.get("device_id")
            for device in devices
            if device.get("device_id")
        }
    )

    failed_logins = sum(
        1
        for event in auth_events
        if event.get("event_type") == "LOGIN_FAILED"
    )

    channel_counter = Counter(
        tx.get("channel")
        for tx in transactions
        if tx.get("channel")
    )

    preferred_channel = (
        channel_counter.most_common(1)[0][0]
        if channel_counter
        else None
    )

    # Details shown on the page
    account_details = [
        {
            "account_id":
                account.get("account_id"),
            "account_type":
                account.get("account_type"),
            "masked_account_number":
                account.get("masked_account_number")
                or account.get("account_number"),
            "current_balance":
                _safe_float(
                    account.get("current_balance")
                ),
            "currency":
                account.get("currency", "KES"),
            "status":
                account.get("status"),
        }
        for account in accounts
    ]

    device_details = [
        {
            "device_id":
                device.get("device_id"),
            "device_name":
                device.get("device_name"),
            "device_brand":
                device.get("device_brand"),
            "device_model":
                device.get("device_model"),
            "device_type":
                device.get("device_type"),
            "os_version":
                device.get("os_version"),
            "trust_score":
                device.get("trust_score"),
            "is_trusted":
                device.get("is_trusted"),
            "last_seen":
                device.get("last_seen"),
        }
        for device in devices
    ]

    recent_transactions = sorted(
        transactions,
        key=lambda row:
            row.get("transaction_date", ""),
        reverse=True,
    )[:10]

    return {
        "customer": {
            "customer_id":
                customer.get("customer_id"),
            "full_name":
                customer.get("full_name")
                or f'{customer.get("first_name", "")} {customer.get("last_name", "")}'.strip(),
            "first_name":
                customer.get("first_name"),
            "last_name":
                customer.get("last_name"),
            "email":
                customer.get("email"),
            "phone_number":
                customer.get("phone_number"),
            "customer_segment":
                customer.get("customer_segment"),
            "branch_code":
                customer.get("branch_code"),
            "home_city":
                customer.get("home_city"),
            "account_status":
                customer.get("account_status"),
            "risk_profile":
                customer.get("risk_profile"),
            "currency":
                customer.get("currency", "KES"),
        },

        "kpis": {
            "total_balance":
                round(account_balance, 2),
            "transactions_30d":
                len(transactions_30d),
            "average_transaction_amount":
                round(_safe_average(all_amounts), 2),
            "account_count":
                len(accounts),
            "known_devices":
                known_devices,
            "failed_logins":
                failed_logins,
        },

        "transaction_summary": {
            "total_transactions":
                len(transactions),
            "preferred_channel":
                preferred_channel,
            "maximum_transaction_amount":
                round(
                    max(all_amounts)
                    if all_amounts
                    else 0,
                    2,
                ),
            "last_transaction":
                latest_transaction_date.isoformat()
                if latest_transaction_date
                else None,
        },

        "behavior_profile": {
            "currency":
                behavior.get("currency", "KES"),
            "average_transaction_amount":
                behavior.get(
                    "avg_transaction_amount"
                ),
            "median_transaction_amount":
                behavior.get(
                    "median_transaction_amount"
                ),
            "typical_amount_range":
                behavior.get(
                    "typical_amount_range", {}
                ),
            "average_transaction_frequency":
                behavior.get(
                    "avg_transaction_frequency"
                ),
            "typical_hours":
                behavior.get("typical_hours", []),
            "typical_locations":
                behavior.get(
                    "typical_locations", []
                ),
            "typical_channels":
                behavior.get(
                    "typical_channels", []
                ),
            "typical_device_names":
                behavior.get(
                    "typical_device_names", []
                ),
        },

        "transaction_trend":
            _build_transaction_trend(
                transactions,
                latest_transaction_date,
                days=30,
            ),

        "accounts":
            account_details,

        "devices":
            device_details,

        "recent_transactions": [
            _clean_recent_transaction(tx)
            for tx in recent_transactions
        ],
    }
