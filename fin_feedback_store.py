import json
import os
from datetime import datetime

FEEDBACK_FILE = "data/feedback.json"


def store_feedback(transaction_id, outcome, signals):
    feedback = {
        "transaction_id": transaction_id,
        "outcome": outcome,  # "false_positive" or "confirmed_fraud"
        "signals": signals,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Loading existing feedback safely
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                data = json.load(f)  
        except (json.JSONDecodeError, ValueError):
            data = [] 
    else:
        data = []

    data.append(feedback)

    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=4)