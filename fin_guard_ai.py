# Standard library imports
import json
import os
import logging
import pickle
from fin_feedback_store import FEEDBACK_FILE, store_feedback
from fin_weight_store import load_weights
import joblib
import random
import string
import hashlib
import ipaddress
from datetime import datetime
from collections import Counter

# Third-party library imports
import numpy as np
import pandas as pd
import logging
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LassoCV
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from flask_cors import CORS

from dotenv import load_dotenv
from openai import OpenAI
import os


weights_map = load_weights()


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) 
random.seed = 42

load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if OPENAI_API_KEY and client:
    logger.info(f"OpenAI API Key (first 8): {OPENAI_API_KEY[:8]}")
    # logger.info(f"OpenAI client initialized: {client}")
    logger.info(f"OpenAI Status: CONNECTED ✓")
else:
    logger.warning("OpenAI Status: DISCONNECTED ✗")
    

# Cache location
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)
logger.info(f"Cache directory set at {DATA_DIR}")


# pickle file
DATA_WRANGLE_PKL = os.path.join(CACHE_DIR, "data_wrangle.pkl")
IMPORTANT_FEATURES_WEIGHTS_PKL = os.path.join(CACHE_DIR, 'important_features_weights.pkl')
IMPORTANT_FEATURES_PKL = os.path.join(CACHE_DIR, 'important_features.pkl')
NORMALIZED_RISK_SCORES_PKL = os.path.join(CACHE_DIR, 'normalized_risk_score.pkl')
REAL_TIME_RISK_SCORES_PKL = os.path.join(CACHE_DIR, 'real_time_risk_score.pkl')
RISK_MODELS_JOBLIB = os.path.join(CACHE_DIR, 'risk_models.joblib')

SCALER_DATA = os.path.join(CACHE_DIR, 'scaler.pkl')
file_path = os.path.join(DATA_DIR, "fraud_detection_data.csv")



@app.route('/v2/api/test', methods=['GET'])
def test():
    return "Testing endpoint, fraud detection apis working effectively !!!!!!!!!!!!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
