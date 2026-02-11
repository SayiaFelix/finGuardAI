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

# Saving to pickle 
def save_to_pickle(data, filename):
    """ Save only the selected important features to a pickle file """
    try:
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        logger.info(f"{file} Pickle File saved successfully to {filename} !!!!!!!")
    except Exception as e:
        logger.error(f"Error saving {file} Pickle File !!!!!: {str(e)}")
        raise

# Loading from pickle
def load_from_pickle(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            logger.info(f"Important features saved successfully to {filename}")
        return data
    else:
        return {} 

# Saving the trained model
def save_model_to_JobLib(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")
    
#### Loading the saved model from a file
def load_model_from_JobLib(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def load_or_initialize_pickle(filename, data):
    """Load pickle file if it exists, or initialize it with default_data."""
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        # Initialize with default data and save
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        return data
    
def load_feedback():
    """Safely load feedback from JSON file."""
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            data = []
    else:
        data = []
    return data


models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.9, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
        'Bagging': BaggingClassifier(n_estimators=50, max_samples=0.8, max_features=0.8, random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=-1, min_data_in_leaf=20, class_weight='balanced', verbose=-1),
        'XGBoost': XGBClassifier(scale_pos_weight=3, n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42),
        'CatBoost': CatBoostClassifier(n_estimators=100, learning_rate=0.1, depth=6, class_weights=[1, 5], verbose=0, random_state=42),
    }

##############  Helper functions

# Data preparation
def prepare_data(file_path):

    logger.info("Loading and preprocessing data !!!!!!!!!!!!!!!!!!!!!!!!!!!")
   
    # file_path = os.path.join(DATA_DIR, "fraud_detection_data (1).csv")
    data = pd.read_csv(file_path)
    cols_to_check = ['Transaction_Amount', 'Device_Type', 'Transaction_Type', 'IP_Address']
    data.dropna(subset=cols_to_check, inplace=True)

    # Convert Transaction_Date to datetime and extract hour, day of week, and weekend info
    data['Transaction_Date'] = pd.to_datetime(data['Transaction_Date'], errors='coerce')
    data['Transaction_Hour'] = data['Transaction_Date'].dt.hour
    data['Day_of_Week'] = data['Transaction_Date'].dt.dayofweek
    data['Is_Weekend'] = (data['Day_of_Week'] >= 5).astype(int)

    # Binning Transaction_Hour into periods
    data['Transaction_Period'] = pd.cut(
        data['Transaction_Hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening'],
        right=False,
        include_lowest=True
    )

    # Binning Transaction_Amount into categories
    data['Amount_Category'] = pd.cut(
        data['Transaction_Amount'],
        bins=[0, 5000, 10000, 15000, float('inf')],
        labels=['Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
    )

    # Dropping unnecessary columns
    data = data.drop(['Transaction_ID', 'Account_ID', 'Transaction_Date'], axis=1)

    # Converting IP addresses and other IDs to integer values
    def convert_to_integer(value):
        try:
            value = str(value)
            if ':' in value or '.' in value:
                return int(ipaddress.ip_address(value))
            elif len(value.split(':')) == 6:
                return int(value.replace(':', ''), 16)
            else:
                return int(hashlib.sha256(value.encode()).hexdigest(), 16) % (10 ** 10)
        except (ValueError, TypeError):
            return None

    data['IP_Address'] = data['IP_Address'].apply(convert_to_integer)

    # Scaling Transaction_Hour
    robust_scaler = RobustScaler()
    data['Transaction_Hour'] = robust_scaler.fit_transform(data['Transaction_Hour'].values.reshape(-1, 1))

    # One-hot encoding categorical columns
    categorical_columns = ['Transaction_Type', 'Device_Type', 'Transaction_Period', 'Amount_Category', 'Transaction_Location']
    onehot_encoder = OneHotEncoder()
    encoded_columns = onehot_encoder.fit_transform(data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_columns.toarray(), columns=onehot_encoder.get_feature_names_out(categorical_columns))
    encoded_df.index = data.index

    # Dropping original categorical columns and concatenate encoded columns
    data = data.drop(categorical_columns, axis=1)
    data = pd.concat([data, encoded_df], axis=1)

    data['Class'] = data['Class'].astype(int)
    cols = data.columns.tolist()
    cols.remove('Class')
    cols.append('Class')
    data = data[cols]
    # logger.info('Data', data)
    return data

def feature_selection_rf(X_train, y_train, original_columns):
    """ Feature selection using RandomForestClassifier """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    sel_rf = SelectFromModel(rf, threshold='median')
    sel_rf.fit(X_train, y_train)
    rf_selected_feat = [original_columns[i] for i in sel_rf.get_support(indices=True)]
    return rf_selected_feat

def feature_selection_lasso(X_train, y_train, original_columns):
    """ Feature selection using LassoCV """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_train_scaled, y_train)
    lasso_selected_feat = [original_columns[i] for i in np.where(lasso.coef_ != 0)[0]]
    return lasso_selected_feat

def feature_selection_xgb(X_train, y_train, original_columns):
    """ Feature selection using XGBoost """
    count = Counter(y_train)
    scale_pos_weight = count[0] / count[1]
    
    xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
    sel_xgb = SelectFromModel(xgb, threshold='median')
    sel_xgb.fit(X_train, y_train)
    xgb_selected_feat = [original_columns[i] for i in sel_xgb.get_support(indices=True)]
    return xgb_selected_feat

def combine_selected_features(rf_selected_feat, lasso_selected_feat, xgb_selected_feat):
    """ Combine features selected by RF, LassoCV, and XGBoost """
    overall_selected_features = set(rf_selected_feat) | set(lasso_selected_feat) | set(xgb_selected_feat)
    return list(overall_selected_features)

def prepare_and_split_data():
    """ Function to load, prepare data, and split it into training and testing sets """
    
    try:
        # Load data from pickle
        data = prepare_data(file_path)
        logger.info('Data loaded', extra={'columns': data.columns.tolist()})
        
        # Extract target variable
        y = data['Class']
        
        # Load selected features from pickle
        overall_selected_features = load_from_pickle(IMPORTANT_FEATURES_PKL)
        logger.info('Loaded selected features from pickle', extra={'features': overall_selected_features})
        
        # Prepare feature set
        X = data[overall_selected_features]
        logger.info('Prepared feature set', extra={'features': X.columns.tolist()})
        
        # Splitting data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        logger.info(f'Training set size: {X_train.shape}, Test set size: {X_test.shape}')

        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error in preparing and splitting data: {str(e)}")
        raise








@app.route('/v2/api/test', methods=['GET'])
def test():
    return "Testing endpoint, fraud detection apis working effectively !!!!!!!!!!!!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
