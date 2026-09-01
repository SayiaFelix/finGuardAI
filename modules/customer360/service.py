
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from functools import lru_cache

from . import repository


def _parse_datetime(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_average(values):
    clean = [_safe_float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else 0.0


def _display_risk(customer):
    return customer.get("risk_profile", "LOW")


def _profile_level(score):
    if score >= 85:
        return "CRITICAL"
    if score >= 65:
        return "HIGH"
    if score >= 35:
        return "MEDIUM"
    return "LOW"


@lru_cache(maxsize=2048)
def build_customer_profile(customer_id):
    customer = repository.get_customer(customer_id)
    if not customer:
        return None

    transactions = repository.get_customer_transactions(customer_id)
    risk_features = repository.get_customer_risk_features(customer_id)
    auth_events = repository.get_customer_auth_events(customer_id)
    devices = repository.get_customer_devices(customer_id)
    beneficiaries = repository.get_customer_beneficiaries(customer_id)
    behavior = repository.get_customer_behavior_profile(customer_id) or {}

    risk_by_tx = {
        row.get("transaction_id"): row
        for row in risk_features
        if row.get("transaction_id")
    }

    known_device_ids = {
        row.get("device_id")
        for row in devices
        if row.get("device_id")
    }

    known_beneficiary_ids = {
        row.get("beneficiary_id")
        for row in beneficiaries
        if row.get("beneficiary_id")
    }

    typical_locations = set(behavior.get("typical_locations") or [])
    if not typical_locations and customer.get("home_city"):
        typical_locations.add(customer.get("home_city"))

    typical_hours = [
        int(hour)
        for hour in (behavior.get("typical_hours") or [])
        if isinstance(hour, (int, float)) or str(hour).isdigit()
    ]
    min_typical_hour = min(typical_hours) if typical_hours else 6
    max_typical_hour = max(typical_hours) if typical_hours else 22

    dated_transactions = []
    for tx in transactions:
        tx_date = _parse_datetime(tx.get("transaction_date"))
        if tx_date:
            dated_transactions.append((tx, tx_date))

    latest_date = (
        max(tx_date for _, tx_date in dated_transactions)
        if dated_transactions
        else None
    )

    critical_count = sum(
        1 for row in risk_features
        if str(row.get("risk_level", "")).upper() == "CRITICAL"
    )
    high_count = sum(
        1 for row in risk_features
        if str(row.get("risk_level", "")).upper() == "HIGH"
    )
    medium_count = sum(
        1 for row in risk_features
        if str(row.get("risk_level", "")).upper() == "MEDIUM"
    )

    blocked_count = sum(
        1 for tx in transactions
        if str(tx.get("status", "")).upper() == "BLOCKED"
    )
    challenged_count = sum(
        1 for tx in transactions
        if str(tx.get("status", "")).upper() == "CHALLENGED"
    )

    fraud_type_counts = Counter(
        str(tx.get("fraud_type", "NORMAL")).upper()
        for tx in transactions
    )

    account_takeover_count = fraud_type_counts.get("ACCOUNT_TAKEOVER", 0)
    high_value_count = fraud_type_counts.get("HIGH_VALUE_ANOMALY", 0)
    velocity_count = fraud_type_counts.get("VELOCITY_FRAUD", 0)

    suspicious_transactions = []
    unknown_device_count = 0
    unusual_location_count = 0
    new_beneficiary_count = 0
    off_hours_count = 0

    for tx, tx_date in dated_transactions:
        risk = risk_by_tx.get(tx.get("transaction_id"), {})
        risk_score = _safe_float(risk.get("final_risk_score"))
        is_suspicious = (
            bool(tx.get("is_fraud"))
            or str(tx.get("status", "")).upper() in {"BLOCKED", "CHALLENGED"}
            or risk_score >= 40
        )

        if not is_suspicious:
            continue

        suspicious_transactions.append((tx, tx_date))

        device_id = tx.get("device_id")
        if device_id and device_id not in known_device_ids:
            unknown_device_count += 1

        location = tx.get("location_city")
        if location and typical_locations and location not in typical_locations:
            unusual_location_count += 1

        beneficiary_id = tx.get("beneficiary_id")
        if beneficiary_id and beneficiary_id not in known_beneficiary_ids:
            new_beneficiary_count += 1

        if tx_date.hour < min_typical_hour or tx_date.hour > max_typical_hour:
            off_hours_count += 1

    failed_logins = sum(
        1
        for event in auth_events
        if str(event.get("event_type", "")).upper() == "LOGIN_FAILED"
        or event.get("success") is False
    )

    recent_suspicious_count = 0
    if latest_date:
        recent_cutoff = latest_date - timedelta(days=30)
        recent_suspicious_count = sum(
            1
            for _, tx_date in suspicious_transactions
            if tx_date >= recent_cutoff
        )

    raw_factors = [
        (
            "Transaction Risk History",
            min(24, critical_count * 12 + high_count * 8 + medium_count * 4),
            f"{critical_count} critical, {high_count} high and {medium_count} medium risk transactions",
        ),
        (
            "Blocked / Challenged Activity",
            min(18, blocked_count * 10 + challenged_count * 3),
            f"{blocked_count} blocked and {challenged_count} challenged transactions",
        ),
        (
            "Account Takeover",
            min(18, account_takeover_count * 12),
            f"{account_takeover_count} account takeover pattern(s) detected",
        ),
        (
            "High-Value Anomaly",
            min(10, high_value_count * 6),
            f"{high_value_count} high-value anomaly/anomalies detected",
        ),
        (
            "New Beneficiary",
            min(8, new_beneficiary_count * 4),
            f"{new_beneficiary_count} suspicious transaction(s) used a previously unseen beneficiary",
        ),
        (
            "New / Unknown Device",
            min(10, unknown_device_count * 4),
            f"{unknown_device_count} suspicious transaction(s) used a previously unseen device",
        ),
        (
            "Unusual Location",
            min(10, unusual_location_count * 4),
            f"{unusual_location_count} suspicious transaction(s) occurred outside the behavioural baseline location",
        ),
        (
            "Off-Hours Activity",
            min(8, off_hours_count * 4),
            f"{off_hours_count} suspicious transaction(s) occurred outside typical activity hours",
        ),
        (
            "Velocity Anomaly",
            min(10, velocity_count * 6),
            f"{velocity_count} velocity fraud pattern(s) detected",
        ),
        (
            "Failed Login Activity",
            min(8, failed_logins * 2),
            f"{failed_logins} failed login event(s) detected",
        ),
        (
            "Recent Suspicious Activity",
            6 if recent_suspicious_count > 0 else 0,
            f"{recent_suspicious_count} suspicious transaction(s) occurred in the latest 30-day activity window",
        ),
    ]

    factors = [
        {
            "name": name,
            "contribution": contribution,
            "evidence": evidence,
        }
        for name, contribution, evidence in raw_factors
        if contribution > 0
    ]
    factors.sort(key=lambda row: row["contribution"], reverse=True)

    score = min(100, sum(row["contribution"] for row in factors))
    level = _profile_level(score)

    reasons = [row["evidence"] for row in factors[:6]]

    return {
        "score": score,
        "level": level,
        "reasons": reasons,
        "factors": factors,
        "statistics": {
            "total_transactions": len(transactions),
            "critical_risk_transactions": critical_count,
            "high_risk_transactions": high_count,
            "medium_risk_transactions": medium_count,
            "blocked_transactions": blocked_count,
            "challenged_transactions": challenged_count,
            "account_takeover_events": account_takeover_count,
            "high_value_anomalies": high_value_count,
            "velocity_anomalies": velocity_count,
            "new_or_unknown_device_events": unknown_device_count,
            "unusual_location_events": unusual_location_count,
            "new_beneficiary_events": new_beneficiary_count,
            "off_hours_events": off_hours_count,
            "failed_logins": failed_logins,
            "recent_suspicious_transactions": recent_suspicious_count,
        },
        "legacy_risk_profile": customer.get("risk_profile"),
    }


def list_customers(page=1, size=20, search=None, segment=None, risk_profile=None, status=None):
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
            searchable = " ".join([
                str(customer.get("customer_id", "")),
                str(customer.get("first_name", "")),
                str(customer.get("last_name", "")),
                str(customer.get("full_name", "")),
                str(customer.get("phone_number", "")),
                str(customer.get("email", "")),
            ]).lower()
            if search_value not in searchable:
                continue

        results.append({
            "customer_id": customer.get("customer_id"),
            "full_name": customer.get("full_name")
                or f'{customer.get("first_name", "")} {customer.get("last_name", "")}'.strip(),
            "customer_segment": customer.get("customer_segment"),
            "home_city": customer.get("home_city"),
            "account_status": customer.get("account_status"),
            "risk_profile": _display_risk(customer),
            "branch_code": customer.get("branch_code"),
            "currency": customer.get("currency", "KES"),
        })

    total = len(results)
    start = (page - 1) * size
    end = start + size
    total_pages = (total + size - 1) // size if size else 0

    page_results = results[start:end]
    for row in page_results:
        profile = build_customer_profile(row["customer_id"]) or {}
        row["customer_profile"] = {
            "score": profile.get("score", 0),
            "level": profile.get("level", "LOW"),
        }

    return {
        "customers": page_results,
        "pagination": {
            "page": page,
            "size": size,
            "total": total,
            "total_pages": total_pages,
            "has_more": end < total,
        },
    }


def _build_transaction_trend(transactions, reference_date, days=30):
    if not reference_date:
        return []

    start_date = (reference_date - timedelta(days=days - 1)).date()
    daily = defaultdict(lambda: {"transaction_count": 0, "total_amount": 0.0})

    for transaction in transactions:
        tx_date = _parse_datetime(transaction.get("transaction_date"))
        if not tx_date or tx_date.date() < start_date:
            continue
        day = tx_date.date().isoformat()
        daily[day]["transaction_count"] += 1
        daily[day]["total_amount"] += _safe_float(transaction.get("amount"))

    result = []
    for offset in range(days):
        current_date = (start_date + timedelta(days=offset)).isoformat()
        row = daily[current_date]
        result.append({
            "date": current_date,
            "transaction_count": row["transaction_count"],
            "total_amount": round(row["total_amount"], 2),
        })
    return result


def _clean_recent_transaction(transaction):
    risk = repository.get_transaction_risk(
        transaction.get("transaction_id")
    )

    return {
        "transaction_id": transaction.get("transaction_id"),
        "transaction_date": transaction.get("transaction_date"),
        "channel": transaction.get("channel"),
        "transaction_type": transaction.get("transaction_type"),
        "amount": _safe_float(transaction.get("amount")),
        "currency": transaction.get("currency", "KES"),
        "location_city": transaction.get("location_city"),
        "risk_score": _safe_float(risk.get("final_risk_score")),
        "risk_level": risk.get("risk_level", "LOW"),
        "fraud_type": risk.get("fraud_type", transaction.get("fraud_type", "NORMAL")),
        "status": transaction.get("status"),
    }


def build_customer_360(customer_id):
    customer = repository.get_customer(customer_id)
    if not customer:
        return None

    accounts = repository.get_customer_accounts(customer_id)
    transactions = repository.get_customer_transactions(customer_id)
    devices = repository.get_customer_devices(customer_id)
    auth_events = repository.get_customer_auth_events(customer_id)
    behavior = repository.get_customer_behavior_profile(customer_id) or {}
    customer_profile = build_customer_profile(customer_id) or {}

    transaction_dates = [
        _parse_datetime(tx.get("transaction_date"))
        for tx in transactions
    ]
    transaction_dates = [value for value in transaction_dates if value is not None]

    latest_transaction_date = max(transaction_dates) if transaction_dates else None
    cutoff_30d = latest_transaction_date - timedelta(days=29) if latest_transaction_date else None

    transactions_30d = []
    if cutoff_30d:
        transactions_30d = [
            transaction
            for transaction in transactions
            if (
                _parse_datetime(transaction.get("transaction_date"))
                and _parse_datetime(transaction.get("transaction_date")) >= cutoff_30d
            )
        ]

    all_amounts = [_safe_float(tx.get("amount")) for tx in transactions]
    account_balance = sum(_safe_float(account.get("current_balance")) for account in accounts)
    known_devices = len({device.get("device_id") for device in devices if device.get("device_id")})
    failed_logins = sum(
        1 for event in auth_events
        if event.get("event_type") == "LOGIN_FAILED"
    )

    channel_counter = Counter(
        tx.get("channel")
        for tx in transactions
        if tx.get("channel")
    )
    preferred_channel = channel_counter.most_common(1)[0][0] if channel_counter else None

    account_details = [
        {
            "account_id": account.get("account_id"),
            "account_type": account.get("account_type"),
            "masked_account_number": account.get("masked_account_number") or account.get("account_number"),
            "current_balance": _safe_float(account.get("current_balance")),
            "currency": account.get("currency", "KES"),
            "status": account.get("status"),
        }
        for account in accounts
    ]

    device_details = [
        {
            "device_id": device.get("device_id"),
            "device_name": device.get("device_name"),
            "device_brand": device.get("device_brand"),
            "device_model": device.get("device_model"),
            "device_type": device.get("device_type"),
            "os_version": device.get("os_version"),
            "trust_score": device.get("trust_score"),
            "is_trusted": device.get("is_trusted"),
            "last_seen": device.get("last_seen"),
        }
        for device in devices
    ]

    recent_transactions = sorted(
        transactions,
        key=lambda row: row.get("transaction_date", ""),
        reverse=True,
    )[:15]

    return {
        "customer": {
            "customer_id": customer.get("customer_id"),
            "full_name": customer.get("full_name")
                or f'{customer.get("first_name", "")} {customer.get("last_name", "")}'.strip(),
            "first_name": customer.get("first_name"),
            "last_name": customer.get("last_name"),
            "email": customer.get("email"),
            "phone_number": customer.get("phone_number"),
            "customer_segment": customer.get("customer_segment"),
            "branch_code": customer.get("branch_code"),
            "home_city": customer.get("home_city"),
            "account_status": customer.get("account_status"),
            "risk_profile": customer.get("risk_profile"),
            "currency": customer.get("currency", "KES"),
        },
        "customer_profile": customer_profile,
        "kpis": {
            "total_balance": round(account_balance, 2),
            "transactions_30d": len(transactions_30d),
            "average_transaction_amount": round(_safe_average(all_amounts), 2),
            "account_count": len(accounts),
            "known_devices": known_devices,
            "failed_logins": failed_logins,
        },
        "transaction_summary": {
            "total_transactions": len(transactions),
            "preferred_channel": preferred_channel,
            "maximum_transaction_amount": round(max(all_amounts) if all_amounts else 0, 2),
            "last_transaction": latest_transaction_date.isoformat() if latest_transaction_date else None,
        },
        "behavior_profile": {
            "currency": behavior.get("currency", "KES"),
            "average_transaction_amount": behavior.get("avg_transaction_amount"),
            "median_transaction_amount": behavior.get("median_transaction_amount"),
            "typical_amount_range": behavior.get("typical_amount_range", {}),
            "average_transaction_frequency": behavior.get("avg_transaction_frequency"),
            "typical_hours": behavior.get("typical_hours", []),
            "typical_locations": behavior.get("typical_locations", []),
            "typical_channels": behavior.get("typical_channels", []),
            "typical_device_names": behavior.get("typical_device_names", []),
        },
        "transaction_trend": _build_transaction_trend(
            transactions,
            latest_transaction_date,
            days=30,
        ),
        "accounts": account_details,
        "devices": device_details,
        "recent_transactions": [
            _clean_recent_transaction(tx)
            for tx in recent_transactions
        ],
    }
