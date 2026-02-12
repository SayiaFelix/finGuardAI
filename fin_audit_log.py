import json
import os
from datetime import datetime

AUDIT_FILE = "data/adaptation_audit.json"

def log_adaptation(before, after, reason):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "reason": reason,
        "before": before,
        "after": after
    }

    if os.path.exists(AUDIT_FILE):
        with open(AUDIT_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(log_entry)

    with open(AUDIT_FILE, "w") as f:
        json.dump(data, f, indent=4)
