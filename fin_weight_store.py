import json
import os

WEIGHTS_FILE = "data/signal_weights.json"

def load_weights():
    if not os.path.exists(WEIGHTS_FILE):
        return {}
    with open(WEIGHTS_FILE, "r") as f:
        return json.load(f)

def save_weights(weights):
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(weights, f, indent=4)
