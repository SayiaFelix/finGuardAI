# Standard library imports
import json
import os
import logging
import pickle
from fin_feedback_store import FEEDBACK_FILE, store_feedback
from fin_weight_store import load_weights
from finca_rules import evaluate_all_rules, register_rules_routes
import joblib
import random
import string
import hashlib
import ipaddress
from datetime import datetime
from collections import Counter

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid

# Third-party library imports
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LassoCV
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from auth.jwt_auth import token_required, admin_required, analyst_required, api_key_required
from auth.auth_routes import register_auth_routes

from database.db_manager import save_transaction_to_db, save_feedback_to_db
from database.db_manager import SessionLocal, Transaction

from database.db_manager import (
    save_case_to_db,
    get_cases_from_db,
    update_case_in_db,
    delete_case_from_db,
    assign_case_to_analyst,
    add_case_note,
    resolve_case
)

from database.db_manager import SessionLocal, Transaction
from sqlalchemy import desc

from datetime import datetime
import pytz 

from flask_cors import CORS

from dotenv import load_dotenv
from openai import OpenAI

weights_map = load_weights()

from modules.customer360.routes import customer360_bp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



app = Flask(__name__)

CORS(app, origins=['*'])

app.register_blueprint(
    customer360_bp,
    url_prefix="/finca/v1"
)

random.seed(42)

load_dotenv() 

GROQ_API_KEY = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")

if GROQ_API_KEY:
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url=base_url
    )
    logger.info(f"Groq API Key (first 8): {GROQ_API_KEY[:8]}")
    logger.info(f"Groq Status: CONNECTED ✓")
else:
    client = None
    logger.warning("Groq Status: DISCONNECTED ✗ - No API key found")    
     
ml_executor = ThreadPoolExecutor(max_workers=10)
file_lock = threading.Lock() 

# Cache location
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)
logger.info(f"Cache directory set at {DATA_DIR}")

DATA_WRANGLE_PKL = os.path.join(CACHE_DIR, "data_wrangle.pkl")
IMPORTANT_FEATURES_WEIGHTS_PKL = os.path.join(CACHE_DIR, 'important_features_weights.pkl')
IMPORTANT_FEATURES_PKL = os.path.join(CACHE_DIR, 'important_features.pkl')
NORMALIZED_RISK_SCORES_PKL = os.path.join(CACHE_DIR, 'normalized_risk_score.pkl')
REAL_TIME_RISK_SCORES_PKL = os.path.join(CACHE_DIR, 'real_time_risk_score.pkl')
RISK_MODELS_JOBLIB = os.path.join(CACHE_DIR, 'risk_models.joblib')
MODEL_METRICS_PKL = os.path.join(CACHE_DIR, 'model_metrics.pkl')

SCALER_DATA = os.path.join(CACHE_DIR, 'scaler.pkl')
file_path = os.path.join(DATA_DIR, "fraud_detection_data.csv")

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Setting the custom encoder
app.json_encoder = CustomJSONEncoder

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
  
def get_nairobi_time():
    """Returns current time in Africa/Nairobi timezone"""
    nairobi_tz = pytz.timezone('Africa/Nairobi')
    utc_now = datetime.utcnow()
    utc_now = utc_now.replace(tzinfo=pytz.UTC)
    nairobi_time = utc_now.astimezone(nairobi_tz)
    return nairobi_time.isoformat()

MODEL_VERSION = "v1.0.0"

SOVEREIGN_MODE = True 
NATIONAL_ALERT_MODE = False

DEFAULT_THRESHOLD = 6.0
ALERT_THRESHOLD = 4.0

# In-memory storage for FINCA demo
finca_transactions = {}
finca_alerts = {}
finca_cases = {}

def get_active_threshold():
    return ALERT_THRESHOLD if NATIONAL_ALERT_MODE else DEFAULT_THRESHOLD

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

#### Loading the saved model
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
        'XGBoost': XGBClassifier(scale_pos_weight=3, n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=42),
        'CatBoost': CatBoostClassifier(n_estimators=100, learning_rate=0.1, depth=6, class_weights=[1, 5], verbose=0, random_state=42),
    }

def prepare_data(file_path):

    logger.info("Loading and preprocessing data !!!!!!!!!!!!!!!!!!!!!!!!!!!")
   
    #file_path = os.path.join(DATA_DIR, "fraud_detection_data (1).csv")
    data = pd.read_csv(file_path)
    cols_to_check = ['Transaction_Amount', 'Device_Type', 'Transaction_Type', 'IP_Address']
    data.dropna(subset=cols_to_check, inplace=True)

    #Converting Transaction_Date to datetime and extract hour, day of week, and weekend info
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

    #Binning Transaction_Amount into categories
    data['Amount_Category'] = pd.cut(
        data['Transaction_Amount'],
        bins=[0, 5000, 10000, 15000, float('inf')],
        labels=['Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
    )

    data = data.drop(['Transaction_ID', 'Account_ID', 'Transaction_Date'], axis=1)

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

    robust_scaler = RobustScaler()
    data['Transaction_Hour'] = robust_scaler.fit_transform(data['Transaction_Hour'].values.reshape(-1, 1))

    #One-hot encode categorical columns
    categorical_columns = ['Transaction_Type', 'Device_Type', 'Transaction_Period', 'Amount_Category', 'Transaction_Location']
    onehot_encoder = OneHotEncoder()
    encoded_columns = onehot_encoder.fit_transform(data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_columns.toarray(), columns=onehot_encoder.get_feature_names_out(categorical_columns))
    encoded_df.index = data.index

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
    
    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
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
        data = prepare_data(file_path)
        logger.info('Data loaded', extra={'columns': data.columns.tolist()})

        y = data['Class']
        
        overall_selected_features = load_from_pickle(IMPORTANT_FEATURES_PKL)
        logger.info('Loaded selected features from pickle', extra={'features': overall_selected_features})
        
        X = data[overall_selected_features]
        logger.info('Prepared feature set', extra={'features': X.columns.tolist()})
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        logger.info(f'Training set size: {X_train.shape}, Test set size: {X_test.shape}')

        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error in preparing and splitting data: {str(e)}")
        raise

#Function to calculate feature importance weight
def calculate_feature_importance_weights():
    """ Calculate feature importance weights using RandomForest, Lasso, and XGBoost models """

    X_train, X_test, y_train, y_test = prepare_and_split_data()
  
    # Fit the models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_train, y_train)

    xgb = XGBClassifier(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)

    # Extract feature importances
    rf_importances = rf.feature_importances_
    lasso_coefficients = lasso.coef_
    xgb_importances = xgb.feature_importances_

    feature_names = X_train.columns
    weights_df = pd.DataFrame({
        'Feature': feature_names,
        'RF_Importance': rf_importances,
        # 'Lasso_Coefficients': np.where(lasso_coefficients != 0, lasso_coefficients, 0),
        'XGB_Importance': xgb_importances
    })
    print('\n================================ Calculated Weights ===============================================')
    print(weights_df)
    print('==========================================================================================')

    weights_df.set_index('Feature', inplace=True)

    weights_df['RF_Importance'] = weights_df['RF_Importance'] / weights_df['RF_Importance'].sum()
    # weights_df['Lasso_Coefficients'] = weights_df['Lasso_Coefficients'] / np.abs(weights_df['Lasso_Coefficients']).sum()
    weights_df['XGB_Importance'] = weights_df['XGB_Importance'] / weights_df['XGB_Importance'].sum()

    weights_df['Combined_Weight'] = (weights_df['RF_Importance'] +
                                      weights_df['XGB_Importance']) / 2

    sorted_weights = weights_df.sort_values(by='Combined_Weight', ascending=False)
    print('\n\n================================ Sorted Weights ===========================================')
    print(sorted_weights)
    print('================================================================================================')
    
    save_to_pickle(sorted_weights, IMPORTANT_FEATURES_WEIGHTS_PKL)
    logger.info('IMPORTANT_FEATURES_WEIGHTS pickle Saved Successfully !!!!!!')
    
    return sorted_weights

# normalize_and_categorize_risk_scores
def normalize_and_categorize_risk_scores():
  
    X_train, X_test, y_train, y_test = prepare_and_split_data()

    print('Training models, Saving and Calculating risk scores...')
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f'{name} model saved as {name}_model.joblib successfully!!!!!!!!!')

    save_model_to_JobLib(models, RISK_MODELS_JOBLIB)
    print(f"All models saved in '{RISK_MODELS_JOBLIB}' successfully!!!!!")
    
    # Initialize DF
    results_df = pd.DataFrame(X_test)
    results_df['True_Label'] = y_test

    aggregated_scores = np.zeros(len(X_test))

    for name, model in models.items():
        # Predict probabilities (risk scores)
        risk_scores = model.predict_proba(X_test)[:, 1]  
        print('Model', name,'risk scores ===========>', risk_scores)
        aggregated_scores += risk_scores
        
    aggregated_scores /= len(models)

    # Normalize aggregated scores to a 0–100 scale
    min_score = aggregated_scores.min()
    max_score = aggregated_scores.max()
    
    # normalized_scores = 100 * (aggregated_scores - min_score) / (max_score - min_score)
  
    normalized_scores = np.clip(aggregated_scores * 100, 0, 10)

    print('Risk score =========>', normalized_scores)
    print("Max Score:", max_score)
    print("Min Score:", min_score)

    results_df['Average_Risk_Score'] = aggregated_scores
    results_df['Normalized_Risk_Score'] = normalized_scores
    results_df['Fraud_Prediction'] = (normalized_scores >= 5).astype(int)

    #Risk categories using binning
    bins = [0, 5, 10]
    labels = ['Low Potential Fraud', 'High Fraud Potential']
    results_df['Risk_Category'] = pd.cut(results_df['Normalized_Risk_Score'], bins=bins, labels=labels, include_lowest=True)
    
    print("\n Results has been calculated and updated Successfully !!!!!")
    print("="*100)
    print(results_df.columns.to_list())
    print("="*100, '\n')
    
    return results_df[['Transaction_Amount', 'Average_Risk_Score', 'Normalized_Risk_Score',  'True_Label', 'Fraud_Prediction','Risk_Category']]

def real_time_risk_scoring(transaction, models, weights_map):
    """
    Improved function to calculate risk score with better weighting.
    Hybrid approach combining 7 ML models with rule-based detection.
    """
    
    overall_selected_features = load_from_pickle(IMPORTANT_FEATURES_PKL)
    
    ## Ensuring transaction has all required features
    for feature in overall_selected_features:
        if feature not in transaction:
            transaction[feature] = 0
    
    transaction_features = transaction[overall_selected_features]

    predictions = []
    probabilities = []
    
    for name, model in models.items():
        prob = model.predict_proba(transaction_features.values.reshape(1, -1))[:, 1][0]
        probabilities.append(prob)

        pred = model.predict(transaction_features.values.reshape(1, -1))[0]
        predictions.append(pred)
    
    avg_probability = np.mean(probabilities)
    fraud_votes = sum(predictions)
    total_models = len(models)
    
    ## RULE-BASED DETECTION ENGINE
    rule_flagged = False
    rule_reasons = []
    rule_severity = 0  
    
    ## Rule 1: Unknown device
    if transaction.get('Device_Type_Unknown_Device', 0) == 1:
        rule_flagged = True
        rule_reasons.append("Unknown device")
        rule_severity += 2
    
    ## Rule 2: International transaction
    if transaction.get('Transaction_Location_International', 0) == 1:
        rule_flagged = True
        rule_reasons.append("International location")
        rule_severity += 2
    
    ## Rule 3: Weekend night transaction
    if transaction.get('Is_Weekend', 0) == 1 and transaction.get('Transaction_Period_Evening', 0) == 1:
        rule_flagged = True
        rule_reasons.append("Weekend evening transaction")
        rule_severity += 1
    
    ## Rule 4: Amount exceeds threshold
    if transaction.get('Transaction_Amount', 0) > 100000:
        rule_flagged = True
        rule_reasons.append("Amount exceeds threshold")
        rule_severity += 1
    
    ## Rule 5: High transaction frequency
    if transaction.get('Transaction_Frequency', 0) > 5:
        rule_flagged = True
        rule_reasons.append("High transaction frequency")
        rule_severity += 2
    
    ## Rule 6: Unusual transaction hour
    if transaction.get('Transaction_Hour', 0) < 5 or transaction.get('Transaction_Hour', 0) > 23:
        rule_flagged = True
        rule_reasons.append("Unusual transaction hour")
        rule_severity += 1
    
    total_flags = fraud_votes  # 
    
    if rule_flagged:
        # Add rule severity boost
        if rule_severity >= 4:
            total_flags = max(total_flags, 2)
        elif rule_severity >= 6:
            total_flags = max(total_flags, 3)
        
        # Ensure at least 1 flag if rule triggered
        total_flags = max(total_flags, 1)
    
    if rule_flagged:
        print(f"\nRULE ENGINE TRIGGERED:")
        print(f"   Rules: {', '.join(rule_reasons)}")
        print(f"   Severity: {rule_severity}")
        print(f"   ML Votes: {fraud_votes}/{total_models} → Total Flags: {total_flags}/{total_models}")
    
    high_risk_features = [
        'Amount_Category_Very High',
        'Transaction_Location_International', 
        'Device_Type_Unknown_Device',
        'Transaction_Type_Online',
        'Transaction_Period_Evening'
    ]
    
    feature_score = 0
    for feature in high_risk_features:
        if feature in transaction and transaction[feature] == 1:
            feature_score += 2 
    
    normalized_feature_score = min(feature_score / 10, 1.0)
    
    if rule_flagged:
        rule_boost = min(rule_severity / 10, 0.3)  
        normalized_feature_score = min(normalized_feature_score + rule_boost, 1.0)
    
    final_score = (0.6 * avg_probability + 
                   0.2 * (total_flags / total_models) + 
                   0.2 * normalized_feature_score)
    
    risk_score = round(final_score * 10, 2)
    
    threshold = get_active_threshold()

    if risk_score >= 8.0:
        risk_category = "Critical Fraud Risk"
        recommended_action = "Block transaction immediately and notify authorities."
    elif risk_score >= threshold:
        risk_category = "High Potential Fraud"
        recommended_action = "Flag for review and escalate to fraud investigation team."
    elif risk_score >= 3.0:
        risk_category = "Medium Risk"
        recommended_action = "Require additional verification (2FA)."
    else:
        risk_category = "Low Potential Fraud"
        recommended_action = "Approve transaction with monitoring."
    
    transaction_details = {
        'Transaction_Amount': transaction.get('Transaction_Amount', 0),
        'Risk_Score': risk_score,
        'Model_Agreement': f"{total_flags}/{total_models} models flagged as fraud",  # ← total_flags
        'ML_Votes': f"{total_flags}/{total_models}",  # ← FIXED: now same as Model_Agreement
        'Rule_Engine': {
            'triggered': rule_flagged,
            'rules': rule_reasons,
            'severity': rule_severity
        },
        'Rule_Flags': rule_reasons,  
        'Rule_Triggered': rule_flagged,
        'Hybrid_Score': True 
    }
    
    print(f"\n{'='*60}")
    print(f"TRANSACTION RISK ASSESSMENT (Hybrid ML + Rules)")
    print(f"{'='*60}")
    print(f"Risk Score: {risk_score}/10")
    print(f"Risk Category: {risk_category}")
    print(f"ML Model Agreement: {fraud_votes}/{total_models} models")
    print(f"Rule Engine: {' Triggered' if rule_flagged else ' Not Triggered'}")
    if rule_flagged:
        print(f"Rules Triggered: {', '.join(rule_reasons)}")
    print(f"Final Flags: {total_flags}/{total_models}")
    print(f"Recommended Action: {recommended_action}")
    print(f"{'='*60}")
    
    return risk_score, risk_category, transaction_details, recommended_action

def generate_transaction_id():
    unique_id = uuid.uuid4().hex[:8].upper()
    date_str = datetime.now().strftime("%Y%m%d")
    random_letter = random.choice(string.ascii_uppercase)
    random_digits = f"{random.randint(0, 999999):06d}" 
    micro = datetime.now().strftime("%f")[:3] 
    return f"T{random_letter}{date_str}{unique_id}{random_digits}{micro}I"

def generate_llm_explanation(
    risk_score,
    risk_category,
    transaction_details,
    recommended_action
):

    global SOVEREIGN_MODE
    
    if not SOVEREIGN_MODE:
        logger.info("Sovereign mode inactive - LLM disabled")
        return None

    if client is None:
        logger.info("LLM disabled: GROQ_API_KEY not set")
        return None
    
    prompt = build_llm_prompt(
        risk_score,
        risk_category,
        transaction_details,
        recommended_action
    )

    try:
        logger.info(f"🔄 Calling Groq API with model: openai/gpt-oss-120b")
        
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a financial fraud analyst explaining risk decisions to banking customers. Be clear, concise, and reassuring in a customer-friendly way. NB: Do NOT mention machine learning or models explicitly. Always use KES instead of $ and complete your paragraph with a clear recommendation at the end."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5, 
            max_tokens=200 
        )
        
        # Extract the explanation properly
        if response and response.choices and len(response.choices) > 0:
            explanation = response.choices[0].message.content.strip()
            if explanation:
                logger.info(f"✅ Groq LLM explanation generated successfully: {explanation[:100]}...")
                return explanation
            else:
                logger.warning("⚠️ LLM returned empty explanation")
                return None
        else:
            logger.warning("⚠️ LLM response had no choices")
            return None

    except Exception as e:
        logger.error(f"❌ LLM explanation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def build_llm_prompt(
    risk_score,
    risk_category,
    transaction_details,
    recommended_action
):
    #rule information
    rule_info = ""
    if transaction_details.get('Rule_Triggered', False):
        rules = transaction_details.get('Rule_Flags', [])
        rule_info = f"\n- Risk Patterns Detected: {', '.join(rules)}"
    
    #Mapping risk category to user-friendly terms
    if "Critical" in risk_category:
        risk_level = "critical"
        action_urgency = "immediately"
    elif "High" in risk_category:
        risk_level = "high"
        action_urgency = "as soon as possible"
    elif "Medium" in risk_category:
        risk_level = "medium"
        action_urgency = "as a precaution"
    else:
        risk_level = "low"
        action_urgency = "for your information"
    
    return f"""
        You are a financial fraud explanation assistant for a bank. Your responses MUST be complete sentences and ALWAYS end with a clear recommendation.

        Transaction summary:
        - Risk Score: {risk_score}/10 ({risk_level} risk)
        - Risk Category: {risk_category}
        - Transaction Amount: KES {transaction_details.get("Transaction_Amount"):,.0f}
        - Risk Indicators: {rule_info if rule_info else 'No specific risk patterns detected'}
        - Recommended Action from System: {recommended_action}

        Write a SINGLE flowing paragraph that:
        1. Starts by stating the transaction amount and risk assessment
        2. Explains WHY this risk level makes sense (mention specific risk factors if any)
        3. ENDS with a clear, complete recommendation (use the recommended action provided) with full sentences.

        IMPORTANT GUIDELINES:
        - Write in complete sentences only
        - Do NOT use numbered lists (1., 2., 3.)
        - Do NOT use bullet points
        - Do NOT mention machine learning or models explicitly
        - ALWAYS end with a complete sentence that states the recommendation with full stop at the end
        - Keep it concise but complete (1-2 sentences max)
        - Use KES when mentioning amounts, and format with commas for thousands
        - Use a clear, trustworthy, and customer-friendly tone throughout the explanation.

        Tone: Clear, Trustworthy, Customer-friendly
        """
 
def generate_fraud_explanation(risk_score, risk_category, transaction_details):
    """
    Generates a human-readable explanation for ALL risk categories
    
    NOTE: risk_score should be in the SAME scale as the response
    - For Real-time API: 0-10 scale
    - For FINCA API: 0-100 scale
    """
    
    amount = transaction_details.get("Transaction_Amount", 0)
    model_agreement = transaction_details.get(
        "Model_Agreement", "multiple models evaluated this transaction"
    )

    signals = []

    if amount >= 100000:
        signals.append("a relatively high transaction amount")
    elif amount >= 50000:
        signals.append("a moderately high transaction amount")

    flagged_models = model_agreement.split("/")[0]
    try:
        flagged_models = int(flagged_models)
    except:
        flagged_models = 0

    if flagged_models >= 4:
        signals.append("strong agreement across multiple fraud detection models")
    elif flagged_models >= 2:
        signals.append("partial agreement across fraud detection models")
    else:
        signals.append("minimal agreement across fraud detection models")

    if not signals:
        signals.append("normal transaction behavior patterns")

    signals_text = ", ".join(signals)

    # Determine explanation based on risk_category string
    risk_category_lower = risk_category.lower()
    
    if "critical" in risk_category_lower:
        explanation = (
            f"This transaction was identified as Critical Fraud Risk with a risk score of "
            f"{round(risk_score, 1)}. Strong risk signals were detected, including {signals_text}, "
            f"and a high level of consensus among fraud detection models. "
            f"The observed patterns closely resemble confirmed fraud cases, posing a significant threat "
            f"of financial loss. As a result, the transaction was blocked automatically and escalated "
            f"for immediate investigation. ({model_agreement})."
        )
    elif "high" in risk_category_lower:
        explanation = (
            f"This transaction was flagged as High Potential Fraud with a risk score of "
            f"{round(risk_score, 1)}. The system detected {signals_text}, along with behavioral patterns "
            f"that differ significantly from the customer's historical activity. "
            f"These indicators are consistent with known fraud scenarios observed across similar accounts. "
            f"Immediate review by the fraud investigation team is recommended. "
            f"({model_agreement})."
        )
    elif "medium" in risk_category_lower:
        explanation = (
            f"This transaction was classified as Medium Risk with a risk score of "
            f"{round(risk_score, 1)}. While the transaction does not strongly indicate fraud, "
            f"the system detected {signals_text}, which slightly deviates from normal patterns. "
            f"As a precaution, additional verification is recommended to confirm transaction legitimacy."
        )
    else:
        # Low Potential Fraud
        explanation = (
            f"This transaction was assessed as Low Potential Fraud with a risk score of "
            f"{round(risk_score, 1)}. The transaction aligns closely with the "
            f"customer's typical behavior and historical transaction patterns. "
            f"Only minimal risk indicators were observed, including {signals_text}. "
            f"As a result, the transaction was approved while remaining under routine monitoring."
        )

    return explanation

def get_final_explanation(
    risk_score,
    risk_category,
    transaction_details,
    recommended_action
):
    llm_explanation = generate_llm_explanation(
        risk_score,
        risk_category,
        transaction_details,
        recommended_action
    )

    if llm_explanation:
        return llm_explanation

    return generate_fraud_explanation(
        risk_score,
        risk_category,
        transaction_details
    )

def adapt_weights(transaction_features, feedback, weights_file=IMPORTANT_FEATURES_WEIGHTS_PKL):
    """
    Adjusts feature weights based on user feedback.
    """
    try:
        weights_df = load_from_pickle(weights_file)
        
        ##Ensuring weights_df is not empty
        if weights_df.empty:
            logger.error(f"Weights DataFrame is empty from {weights_file}")
            weights_df = calculate_feature_importance_weights()
 
        weights_map = weights_df['Combined_Weight'].to_dict()
        
        selected_features = load_from_pickle(IMPORTANT_FEATURES_PKL)
        
        ##Increase step size for more visible effect
        step_size = 0.05  
        
        logger.info(f"Adapting weights for feedback: {feedback}")
        logger.info(f"Features in transaction: {list(transaction_features.keys())[:5]}...")
        
        updated_count = 0
        for feature in selected_features:
            if feature in transaction_features:
                current_weight = weights_map.get(feature, 0)
                
                if feedback == "confirmed_fraud":
                    # Increase weight for features that contributed to correct fraud detection
                    new_weight = min(current_weight + step_size, 1.0)
                    weights_map[feature] = new_weight
                    updated_count += 1
                    print(f"   Increased {feature}: {current_weight:.4f} → {new_weight:.4f}")
                    
                elif feedback == "false_positive":
                    # Decrease weight for features that led to false positive
                    new_weight = max(current_weight - step_size, 0.0)
                    weights_map[feature] = new_weight
                    updated_count += 1
                    print(f" Decreased {feature}: {current_weight:.4f} → {new_weight:.4f}")
        
        for feature in weights_df.index:
            if feature in weights_map:
                weights_df.loc[feature, 'Combined_Weight'] = weights_map[feature]
        
        # Normalize to ensure weights sum to 1
        weights_df['Combined_Weight'] = weights_df['Combined_Weight'] / weights_df['Combined_Weight'].sum()
        
        save_to_pickle(weights_df, weights_file)
        
        logger.info(f"Adaptive weights updated: {updated_count} features adjusted for {feedback}")
        logger.info(f"Updated weight range: [{weights_df['Combined_Weight'].min():.4f}, {weights_df['Combined_Weight'].max():.4f}]")
        
        return weights_map

    except Exception as e:
        logger.error(f"Error updating adaptive weights: {str(e)}")
       
        weights_df = load_from_pickle(IMPORTANT_FEATURES_WEIGHTS_PKL)
        return weights_df['Combined_Weight'].to_dict()

def layer3_lite_adjustment(
    base_risk_score,
    transaction_amount,
    avg_amount=None, 
    tx_count_last_hour=1
):
    """
    Layer 3 Lite with dynamic average based on transaction patterns
    """
 
    if avg_amount is None:
        if transaction_amount < 2000:
            avg_amount = 500  #Small transactions average
        elif transaction_amount < 20000:
            avg_amount = 25000  #Medium transactions average
        else:
            avg_amount = 50000 
    
    # Amount anomaly (0–1)
    amount_risk = min(abs(transaction_amount - avg_amount) / max(avg_amount, 1), 1)

    # Velocity risk (0–1)
    velocity_risk = min(tx_count_last_hour / 5, 1)

    adjusted_score = (
        0.6 * (base_risk_score / 10) +
        0.25 * amount_risk +
        0.15 * velocity_risk
    )

    final_score = round(min(adjusted_score * 10, 10), 2)

    signals = {
        "amount_risk": float(round(amount_risk, 3)),
        "velocity_risk": float(round(velocity_risk, 3)),
        "avg_amount_used": float(avg_amount) 
    }

    return float(final_score), signals

def log_decision(transaction_id, risk_score, risk_category, recommended_action):
    global NATIONAL_ALERT_MODE
    
    log_entry = {
        "timestamp": get_nairobi_time(),
        "transaction_id": transaction_id,
        "model_version": MODEL_VERSION,
        "risk_score": risk_score,
        "risk_category": risk_category,
        "recommended_action": recommended_action,
        "national_alert_mode": NATIONAL_ALERT_MODE
    }

    os.makedirs("data", exist_ok=True)
    
    with open("data/audit_log.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def make_json_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    
#########################################################################################################################################
######################################## -------------------- APIS End Points ------------------------###################################
#########################################################################################################################################

### authentication routes
register_auth_routes(app)

## rules routes
register_rules_routes(app) 

@app.route('/v1/api/data_preparation', methods=['POST'])
def prepare_data_endpoint():
    filename = request.json.get("filename")
    if not filename:
        return jsonify({"error": "filename not provided"}), 400

    file_path = os.path.join(DATA_DIR, filename)

    if os.path.exists(DATA_WRANGLE_PKL):
        try:
       
            processed_data = load_from_pickle(DATA_WRANGLE_PKL)
            return jsonify({
                "status": "success",
                "message": "Data loaded from pickle !!!!!!!!!!",
                "data_shape": processed_data.shape
            })
        except Exception as e:
            logger.error(f"Error loading processed data from pickle: {e}")
            return jsonify({"error": str(e)}), 500
    else:
   
        try:
            processed_data = prepare_data(file_path)
            save_to_pickle(processed_data, DATA_WRANGLE_PKL)
            return jsonify({
                "status": "success",
                "message": "Data processed and saved successfully !!!!!!!!!!!!!!",
                "data_shape": processed_data.shape
            })
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return jsonify({"error": str(e)}), 500

@app.route('/v1/api/feature_selection', methods=['GET'])
def feature_selection():
    """ Endpoint for performing feature selection """
    try:
        # features pickle file exists
        if os.path.exists(IMPORTANT_FEATURES_PKL):
           
            overall_selected_features = load_from_pickle(IMPORTANT_FEATURES_PKL)
            logger.info("Data Loaded from the pickle file !!!!!!!")
            
            return jsonify({
                'status': 'success',
                'message': 'Loaded selected features from pickle file !!!!!!!!!',
                'selected_features': overall_selected_features
            })
        else:
            logger.info("Important Feature pickle file not found. Running Feature Selection !!!!!!!!!!!!!!.")
            data = load_from_pickle(DATA_WRANGLE_PKL)

            if isinstance(data, pd.DataFrame):
               
                X = data.drop('Class', axis=1) 
                y = data['Class']
                
                original_columns = X.columns.tolist()
                # logger.info('Original Columns from our DataFrame !!!!!!!', original_columns)
                
                # Perform feature selection using RF, Lasso, and XGBoost
                rf_selected_feat = feature_selection_rf(X, y, original_columns)
                lasso_selected_feat = feature_selection_lasso(X, y, original_columns)
                xgb_selected_feat = feature_selection_xgb(X, y, original_columns)
                
                # all selected features
                overall_selected_features = combine_selected_features(rf_selected_feat, lasso_selected_feat, xgb_selected_feat)
                logger.info('Overall Selected Features from our Models !!!!!!!', overall_selected_features)
                
            
                save_to_pickle(overall_selected_features, IMPORTANT_FEATURES_PKL)
                logger.info("Overall Selected Features saved to pickle file successfully !!!!!!!!!!!!!")
                
                return jsonify({
                    'status': 'success',
                    'message': 'Feature selection completed successfully and selected features saved !!!!!!!!!!',
                    'selected_features': overall_selected_features
                })
            else:
                return jsonify({
                    'status': 'fail',
                    'message': 'Loaded data is not in the correct format (expected DataFrame)'
                }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500

@app.route('/v1/api/feature_importance_weight', methods=['GET'])
def feature_importance_weight_endpoint():
    """ Endpoint for loading or calculating feature importance weights """
    try:
        if os.path.exists(IMPORTANT_FEATURES_WEIGHTS_PKL):
       
            feature_importance_df = load_from_pickle(IMPORTANT_FEATURES_WEIGHTS_PKL)
            
            return jsonify({
                'status': 'success',
                'message': 'Loaded feature importance weights from pickle file.',
                'feature_importance': feature_importance_df.to_dict(orient='index')
            })
        
        weights_df = calculate_feature_importance_weights()

        return jsonify({
            'status': 'success',
            'message': 'Feature importance calculated and saved successfully !!!!!!!!!!!!!!!!!!!',
            'feature_importance': weights_df.to_dict(orient='index')
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500

@app.route('/v1/api/batch_risk_scores', methods=['POST'])
def normalized_scores_endpoint():
    """Endpoint for loading or calculating normalized scores sample"""
    try:
        # Parse JSON request body
        data = request.get_json()
        page = data.get('page', 1) 
        size = data.get('size', 10) 

        ##### Validate that page and size are positive integers
        if not isinstance(page, int) or not isinstance(size, int) or page < 1 or size < 1:
            return jsonify({
                'status': 'error',
                'message': 'Page and size must be positive integers.'
            }), 400

        if os.path.exists(NORMALIZED_RISK_SCORES_PKL):
            normalized_scores_df = load_from_pickle(NORMALIZED_RISK_SCORES_PKL)
        else:
            ### If the pickle file doesn't exist, calculate the normalized scores
            normalized_scores_df = normalize_and_categorize_risk_scores()
            save_to_pickle(normalized_scores_df, NORMALIZED_RISK_SCORES_PKL)

        normalized_scores_dict = normalized_scores_df.to_dict(orient='index')

        total_records = len(normalized_scores_dict)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated_data = dict(list(normalized_scores_dict.items())[start_idx:end_idx])

        return jsonify({
            'status': 'success',
            'message': 'Normalized scores retrieved successfully !!!!!!!!!!!!',
            'page': page,
            'size': size,
            'total_records': total_records,
            'normalized_scores': paginated_data
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500

@app.route('/v1/api/real_time_risk_score', methods=['POST'])
@token_required
def real_time_risk_score_endpoint(current_user):
    """
    ASYNC VERSION - SQLite as primary storage
    """
    try:
        data = request.json
        transaction_id = generate_transaction_id()
        
        def _process_transaction():
            """Process transaction - SQLite is thread-safe"""
            
            transaction_data = pd.Series(data, name=transaction_id)
            timestamp = get_nairobi_time()
            
            # Load models
            models = load_model_from_JobLib(RISK_MODELS_JOBLIB)
            weights_pickle = load_from_pickle(IMPORTANT_FEATURES_WEIGHTS_PKL)
            weights_map = weights_pickle['Combined_Weight']
            
            # Load feedback
            stored_feedback = load_feedback()
            
            ###Check for existing feedback
            existing_feedback = None
            for fb in stored_feedback:
                fb_signals = fb.get("signals", {})
                tx_amount = transaction_data.get("Transaction_Amount", 0)
                fb_amount = fb_signals.get("Transaction_Amount", 0)
                if fb_amount > 0 and abs(tx_amount - fb_amount) / max(fb_amount, 1) < 0.05:
                    existing_feedback = fb
                    break
            
            # Run risk scoring
            baseline_score, baseline_category, baseline_details, baseline_action = real_time_risk_scoring(
                transaction_data, models, weights_map
            )
            
            tx_count_last_hour = data.get("tx_count_last_hour", 1)
            
            adjusted_score, layer3_signals = layer3_lite_adjustment(
                base_risk_score=baseline_score,
                transaction_amount=transaction_data.get("Transaction_Amount", 0),
                tx_count_last_hour=tx_count_last_hour
            )
            
            transaction_details = baseline_details.copy() if baseline_details else {}
            recommended_action = baseline_action
            
            if adjusted_score > baseline_score:
                risk_score = adjusted_score
                threshold = get_active_threshold()
                if risk_score >= 8.0:
                    risk_category = "Critical Fraud Risk"
                elif risk_score >= threshold:
                    risk_category = "High Potential Fraud"
                elif risk_score >= 3.0:
                    risk_category = "Medium Risk"
                else:
                    risk_category = "Low Potential Fraud"
                transaction_details['Risk_Score'] = risk_score
            else:
                risk_score = baseline_score
                risk_category = baseline_category
            
            feedback_effect = None
            if existing_feedback:
                print(f"Similar transaction found. Adapting weights...")
                adapted_weights = adapt_weights(transaction_data, existing_feedback['outcome'], IMPORTANT_FEATURES_WEIGHTS_PKL)
                original_baseline_score = baseline_score
                original_baseline_category = baseline_category
                risk_score, risk_category, adapted_details, adapted_action = real_time_risk_scoring(
                    transaction_data, models, adapted_weights
                )
                transaction_details = adapted_details
                recommended_action = adapted_action
                feedback_effect = {
                    "original_score": original_baseline_score,
                    "adjusted_score": risk_score,
                    "original_category": original_baseline_category,
                    "adjusted_category": risk_category,
                    "difference": abs(risk_score - original_baseline_score),
                    "feedback_outcome": existing_feedback['outcome'],
                    "weights_adjusted": True
                }
            
            transaction_details["real_time_signals"] = layer3_signals
            
            # Generate explanations
            rule_based_explanation = generate_fraud_explanation(
                risk_score=risk_score,
                risk_category=risk_category,
                transaction_details=transaction_details
            )
            
            llm_explanation = generate_llm_explanation(
                risk_score=risk_score,
                risk_category=risk_category,
                transaction_details=transaction_details,
                recommended_action=recommended_action
            )
            
            final_explanation = llm_explanation if llm_explanation else rule_based_explanation
            
            customer_info = {
                'customer_id': data.get('customer_id', f"CUST-{transaction_id[1:9]}"),
                'customer_name': data.get('customer_name', f"Customer {transaction_id[1:9]}"),
                'customer_email': data.get('customer_email', ''),
                'customer_phone': data.get('customer_phone', ''),
                'account_age_days': data.get('account_age_days', 0),
                'avg_transaction_amount': data.get('avg_transaction_amount', 0)
            }
            
            # Prepare transaction data for SQLite
            db_transaction_data = {
                'transaction_id': transaction_id,
                'timestamp': timestamp,
                'risk_score': risk_score,
                'risk_category': risk_category,
                'transaction_details': transaction_details,
                'customer_info': customer_info,
                'recommended_action': recommended_action,
                'explanations': {
                    'rule_based': rule_based_explanation,
                    'llm': llm_explanation,
                    'final': final_explanation
                },
                'llm_status': 'connected' if client is not None else 'disconnected',
                'model_version': MODEL_VERSION,
                'threshold_used': get_active_threshold(),
                'national_alert_mode': NATIONAL_ALERT_MODE,
                'feedback_used': existing_feedback['transaction_id'] if existing_feedback else None,
                'feedback_effect': feedback_effect,
                'status': {'current': 'Open', 'history': []}
            }
            
            # ============ ONLY SAVE TO SQLITE ============
            # SQLite is thread-safe - no lock needed!
            save_transaction_to_db(db_transaction_data)
            logger.info(f"Transaction {transaction_id} saved to SQLite successfully!")
            
            # ============ OPTIONAL: Update pickle with LOCK ============
            # Only update pickle if you need it for backward compatibility
            with file_lock:
                stored_scores = load_or_initialize_pickle(REAL_TIME_RISK_SCORES_PKL, {})
                stored_scores[transaction_id] = db_transaction_data
                save_to_pickle(stored_scores, REAL_TIME_RISK_SCORES_PKL)
            
            log_decision(transaction_id, risk_score, risk_category, recommended_action)
            
            result = {
                'transaction_id': transaction_id,
                'timestamp': timestamp,
                'risk_score': risk_score,
                'risk_category': risk_category,
                'transaction_details': transaction_details,
                'customer_info': customer_info,
                'recommended_action': recommended_action,
                'explanations': {
                    'rule_based': rule_based_explanation,
                    'llm': llm_explanation if llm_explanation else 'LLM not available',
                    'final': final_explanation
                },
                'llm_status': 'connected' if client is not None else 'disconnected',
                'feedback_used': existing_feedback['transaction_id'] if existing_feedback else None,
                'feedback_effect': feedback_effect
            }
            
            return convert_numpy_types(result)
        
        # Submit to thread pool
        future = ml_executor.submit(_process_transaction)
        
        try:
            result = future.result(timeout=30)
        except concurrent.futures.TimeoutError:
            return jsonify({
                'status': 'error',
                'message': 'Processing timeout (30s)'
            }), 504
        
        return jsonify({
            'status': 'success',
            'message': 'Risk score calculated successfully (async mode) !!!!!!!',
            'async_mode': True,
            'async_processing': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Async error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
@app.route('/v1/api/batch_risk_scores_async', methods=['POST'])
@token_required
def batch_risk_scores_async(current_user):
    """
    Process multiple transactions in parallel
    Handles 100+ transactions per minute
    """
    try:
        data = request.json
        transactions = data.get('transactions', [])
        
        if not transactions:
            return jsonify({
                'status': 'error',
                'message': 'No transactions provided'
            }), 400
        
        # Limit to 100 per batch (safety)
        if len(transactions) > 100:
            transactions = transactions[:100]
        
        # Process in parallel
        def _process_single(tx):
            """Process single transaction"""
            try:
                # Create a request context for each transaction
                with app.test_request_context(json=tx):
                    # FIXED: Call the correct endpoint name
                    response = real_time_risk_score_endpoint(current_user)
                    return response.json
            except Exception as e:
                return {'error': str(e), 'status': 'error'}
        
        # Run in parallel using thread pool
        with ThreadPoolExecutor(max_workers=min(len(transactions), 20)) as executor:
            results = list(executor.map(_process_single, transactions))
        
        # Calculate stats
        success_count = sum(1 for r in results if r.get('status') == 'success')
        error_count = len(results) - success_count
        
        return jsonify({
            'status': 'success',
            'message': f'Processed {len(transactions)} transactions',
            'total_processed': len(transactions),
            'success_count': success_count,
            'error_count': error_count,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Batch error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
        
@app.route('/v1/api/transactions', methods=['POST'])
@token_required
def transactions_endpoint(current_user):
    """Get transactions from SQLite (primary)"""
    try:
        data = request.get_json() or {}
        transaction_id = data.get('transaction_id')
        
        if transaction_id:
            #SQLite first
            try:
                db = SessionLocal()
                tx = db.query(Transaction).filter(
                    Transaction.id == transaction_id
                ).first()
                db.close()
                
                if tx:
                    cleaned_transaction = {
                        'transaction_id': tx.id,
                        'timestamp': tx.timestamp.isoformat(),
                        'risk_score': tx.risk_score,
                        'risk_category': tx.risk_category,
                        'transaction_details': tx.transaction_details,
                        'customer_info': tx.customer_info,
                        'recommended_action': tx.recommended_action,
                        'explanations': tx.explanations,
                        'llm_status': tx.llm_status,
                        'feedback_effect': tx.feedback_effect,
                        'status_info': tx.status
                    }
                    
                    risk_category = cleaned_transaction.get('risk_category', 'Unknown')
                    risk_score = float(cleaned_transaction.get('risk_score', 0))
                    
                    if risk_category in ['Critical Fraud Risk', 'High Potential Fraud']:
                        risk_level = 'HIGH_RISK'
                    elif risk_category == 'Medium Risk':
                        risk_level = 'MEDIUM_RISK'
                    else:
                        risk_level = 'LOW_RISK'
                    
                    active_threshold = get_active_threshold()
                    
                    response_data = {
                        'status': 'success',
                        'message': 'Transaction found in SQLite',
                        'transaction_id': transaction_id,
                        'timestamp': cleaned_transaction.get('timestamp', ''),
                        'risk_assessment': {
                            'risk_score': float(risk_score),
                            'risk_category': str(risk_category),
                            'risk_alert_level': str(risk_level),
                            'threshold': active_threshold,
                            'is_high_risk': bool(risk_score >= active_threshold)
                        },
                        'transaction_details': cleaned_transaction.get('transaction_details', {}),
                        'customer_info': cleaned_transaction.get('customer_info', {}),
                        'recommended_action': cleaned_transaction.get('recommended_action', ''),
                        'explanations': cleaned_transaction.get('explanations', {}),
                        'llm_status': cleaned_transaction.get('llm_status', 'disconnected'),
                        'feedback_effect': cleaned_transaction.get('feedback_effect'),
                        'status_info': cleaned_transaction.get('status_info', {})
                    }
                    
                    final_response = convert_numpy_types(response_data)
                    return jsonify(final_response), 200
                else:
                    # Fallback to pickle
                    transactions = load_from_pickle(REAL_TIME_RISK_SCORES_PKL)
                    transaction = transactions.get(transaction_id)
                    if transaction:
                        cleaned_transaction = convert_numpy_types(transaction)
                        return jsonify({
                            'status': 'success',
                            'message': 'Transaction found (pickle fallback)',
                            **cleaned_transaction
                        }), 200
                    else:
                        return jsonify({
                            'status': 'error',
                            'message': f'Transaction {transaction_id} not found'
                        }), 404
                        
            except Exception as e:
                logger.warning(f"SQLite read failed: {e}")
                # Fallback to pickle
                transactions = load_from_pickle(REAL_TIME_RISK_SCORES_PKL)
                transaction = transactions.get(transaction_id)
                if transaction:
                    cleaned_transaction = convert_numpy_types(transaction)
                    return jsonify({
                        'status': 'success',
                        'message': 'Transaction found (pickle fallback)',
                        **cleaned_transaction
                    }), 200
                else:
                    return jsonify({
                        'status': 'error',
                        'message': f'Transaction {transaction_id} not found'
                    }), 404
            
        else:
            ##all transactions - use SQLite
            page = data.get('page', 1)
            size = data.get('size', 1000)
            
            if not isinstance(page, int) or page < 1:
                return jsonify({
                    'status': 'error',
                    'message': 'Page must be an integer greater than 0.'
                }), 400
            
            if not isinstance(size, int) or size < 1 or size > 1000:
                return jsonify({
                    'status': 'error',
                    'message': 'Size must be an integer between 1 and 1000.'
                }), 400
            
            try:
                db = SessionLocal()
                
                # Get total count
                total = db.query(Transaction).count()
                
                # Get paginated results
                transactions = db.query(Transaction)\
                    .order_by(desc(Transaction.timestamp))\
                    .offset((page - 1) * size)\
                    .limit(size)\
                    .all()
                
                db.close()
                
                tx_list = []
                for tx in transactions:
                    tx_list.append({
                        'transaction_id': tx.id,
                        'timestamp': tx.timestamp.isoformat(),
                        'risk_score': tx.risk_score,
                        'risk_category': tx.risk_category,
                        'transaction_details': tx.transaction_details,
                        'customer_info': tx.customer_info,
                        'recommended_action': tx.recommended_action,
                        'explanations': tx.explanations,
                        'llm_status': tx.llm_status,
                        'model_version': tx.model_version,
                        'threshold_used': tx.threshold_used,
                        'feedback_used': tx.feedback_used,
                        'feedback_effect': tx.feedback_effect,
                        'status_info': tx.status
                    })
                
                response_data = {
                    'status': 'success',
                    'message': f'Loaded {len(tx_list)} of {total} transactions from SQLite',
                    'transactions': tx_list,
                    'pagination': {
                        'page': page,
                        'size': size,
                        'total': total,
                        'has_more': (page * size) < total
                    }
                }
                
                final_response = convert_numpy_types(response_data)
                return jsonify(final_response), 200
                
            except Exception as e:
                logger.error(f"SQLite read error: {e}")
                # Fallback to pickle
                transactions = load_from_pickle(REAL_TIME_RISK_SCORES_PKL)
                if not transactions:
                    return jsonify({
                        'status': 'success',
                        'message': 'No transactions found.',
                        'transactions': [],
                        'total': 0
                    }), 200
                
                tx_list = []
                for tx_id, tx_data in transactions.items():
                    cleaned_tx_data = convert_numpy_types(tx_data)
                    tx_list.append({
                        'transaction_id': tx_id,
                        'timestamp': cleaned_tx_data.get('timestamp', ''),
                        'risk_score': cleaned_tx_data.get('risk_score', 0),
                        'risk_category': cleaned_tx_data.get('risk_category', ''),
                        'transaction_details': cleaned_tx_data.get('transaction_details', {}),
                        'customer_info': cleaned_tx_data.get('customer_info', {}),
                        'recommended_action': cleaned_tx_data.get('recommended_action', ''),
                        'explanations': cleaned_tx_data.get('explanations', {}),
                        'llm_status': cleaned_tx_data.get('llm_status', 'disconnected')
                    })
                
                tx_list.sort(key=lambda x: x['timestamp'], reverse=True)
                total = len(tx_list)
                start_idx = (page - 1) * size
                end_idx = start_idx + size
                paginated = tx_list[start_idx:end_idx]
                
                return jsonify({
                    'status': 'success',
                    'message': f'Loaded {len(paginated)} of {total} transactions (pickle fallback)',
                    'transactions': paginated,
                    'pagination': {
                        'page': page,
                        'size': size,
                        'total': total,
                        'has_more': end_idx < total
                    }
                }), 200

    except Exception as e:
        logger.error(f"Error in transactions endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/v1/api/transactions/related', methods=['POST'])
@token_required
def get_related_transactions(current_user):
    """Get related transactions - SQLite first, pickle fallback"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided.'
            }), 400
            
        transaction_id = data.get('transaction_id')
        
        if not transaction_id:
            return jsonify({
                'status': 'error',
                'message': 'transaction_id is required in the request body.'
            }), 400
    
        # Try SQLite first
        try:
            db = SessionLocal()
            target_tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
            
            if not target_tx:
                db.close()
                # Fallback to pickle
                transactions = load_from_pickle(REAL_TIME_RISK_SCORES_PKL)
                if transaction_id not in transactions:
                    return jsonify({
                        'status': 'error',
                        'message': f'Transaction with ID {transaction_id} not found'
                    }), 404
                
                # Use pickle for related search
                target_tx_data = transactions[transaction_id]
                target_details = target_tx_data.get('transaction_details', {})
                target_customer = target_tx_data.get('customer_info', {})
                
                related = []
                for tx_id, tx_data in transactions.items():
                    if tx_id == transaction_id:
                        continue
                    
                    tx_details = tx_data.get('transaction_details', {})
                    tx_customer = tx_data.get('customer_info', {})
                    
                    relationship_score = 0
                    relationship_reasons = []
                    
                    if target_customer.get('customer_id') and tx_customer.get('customer_id') == target_customer.get('customer_id'):
                        relationship_score += 10
                        relationship_reasons.append('same_customer')
                    
                    if target_details.get('IP_Address') and tx_details.get('IP_Address') == target_details.get('IP_Address'):
                        relationship_score += 8
                        relationship_reasons.append('same_ip')
                    
                    target_amount = target_details.get('Transaction_Amount', 0)
                    tx_amount = tx_details.get('Transaction_Amount', 0)
                    if target_amount > 0 and tx_amount > 0:
                        amount_diff = abs(tx_amount - target_amount) / max(target_amount, 1)
                        if amount_diff < 0.3:
                            relationship_score += 5
                            relationship_reasons.append('similar_amount')
                    
                    if relationship_score >= 5:
                        if relationship_score >= 15:
                            strength = 'strong'
                        elif relationship_score >= 10:
                            strength = 'medium'
                        else:
                            strength = 'weak'
                        
                        tx_data_clean = convert_numpy_types(tx_data)
                        related.append({
                            'transaction_id': tx_id,
                            'timestamp': tx_data_clean.get('timestamp', ''),
                            'risk_score': tx_data_clean.get('risk_score', 0),
                            'risk_category': tx_data_clean.get('risk_category', ''),
                            'amount': tx_details.get('Transaction_Amount', 0),
                            'customer_info': tx_customer,
                            'status_info': tx_data_clean.get('status', {}),
                            'relationship': {
                                'score': relationship_score,
                                'strength': strength,
                                'reasons': relationship_reasons
                            }
                        })
                
                related.sort(key=lambda x: (-x['relationship']['score'], x['timestamp']), reverse=True)
                return jsonify({
                    'status': 'success',
                    'message': f'Found {len(related)} related transactions (pickle)',
                    'related_transactions': related[:10]
                }), 200
            
            # Use SQLite data
            target_details = target_tx.transaction_details or {}
            target_customer = target_tx.customer_info or {}
            
            # Get all transactions for relationship matching
            all_txs = db.query(Transaction).all()
            db.close()
            
            related = []
            for tx in all_txs:
                if tx.id == transaction_id:
                    continue
                
                tx_details = tx.transaction_details or {}
                tx_customer = tx.customer_info or {}
                
                relationship_score = 0
                relationship_reasons = []
                
                # Same customer
                if target_customer.get('customer_id') and tx_customer.get('customer_id') == target_customer.get('customer_id'):
                    relationship_score += 10
                    relationship_reasons.append('same_customer')
                
                # Same IP
                if target_details.get('IP_Address') and tx_details.get('IP_Address') == target_details.get('IP_Address'):
                    relationship_score += 8
                    relationship_reasons.append('same_ip')
                
                # Similar amount (within 30%)
                target_amount = target_details.get('Transaction_Amount', 0)
                tx_amount = tx_details.get('Transaction_Amount', 0)
                if target_amount > 0 and tx_amount > 0:
                    amount_diff = abs(tx_amount - target_amount) / max(target_amount, 1)
                    if amount_diff < 0.3:
                        relationship_score += 5
                        relationship_reasons.append('similar_amount')
                
                # Same location
                if (target_details.get('Transaction_Location_International', 0) == 1 and 
                    tx_details.get('Transaction_Location_International', 0) == 1):
                    relationship_score += 4
                    relationship_reasons.append('same_location_type')
                elif (target_details.get('Transaction_Location_Local', 0) == 1 and 
                      tx_details.get('Transaction_Location_Local', 0) == 1):
                    relationship_score += 4
                    relationship_reasons.append('same_location_type')
                
                # Same channel
                if (target_details.get('Transaction_Type_Online', 0) == 1 and 
                    tx_details.get('Transaction_Type_Online', 0) == 1):
                    relationship_score += 3
                    relationship_reasons.append('same_channel')
                elif (target_details.get('Transaction_Type_POS', 0) == 1 and 
                      tx_details.get('Transaction_Type_POS', 0) == 1):
                    relationship_score += 3
                    relationship_reasons.append('same_channel')
                
                if relationship_score >= 5:
                    if relationship_score >= 15:
                        strength = 'strong'
                    elif relationship_score >= 10:
                        strength = 'medium'
                    else:
                        strength = 'weak'
                    
                    related.append({
                        'transaction_id': tx.id,
                        'timestamp': tx.timestamp.isoformat(),
                        'risk_score': tx.risk_score,
                        'risk_category': tx.risk_category,
                        'amount': tx_details.get('Transaction_Amount', 0),
                        'customer_info': tx_customer,
                        'status_info': tx.status,
                        'relationship': {
                            'score': relationship_score,
                            'strength': strength,
                            'reasons': relationship_reasons
                        }
                    })
            
            related.sort(key=lambda x: (-x['relationship']['score'], x['timestamp']), reverse=True)
            
            return jsonify({
                'status': 'success',
                'message': f'Found {len(related)} related transactions',
                'related_transactions': related[:10]
            }), 200
            
        except Exception as e:
            logger.warning(f"SQLite related query failed: {e}")
            # Fallback to pickle
            transactions = load_from_pickle(REAL_TIME_RISK_SCORES_PKL)
            if transaction_id not in transactions:
                return jsonify({
                    'status': 'error',
                    'message': f'Transaction with ID {transaction_id} not found'
                }), 404
            
            target_tx_data = transactions[transaction_id]
            target_details = target_tx_data.get('transaction_details', {})
            target_customer = target_tx_data.get('customer_info', {})
            
            related = []
            for tx_id, tx_data in transactions.items():
                if tx_id == transaction_id:
                    continue
                
                tx_details = tx_data.get('transaction_details', {})
                tx_customer = tx_data.get('customer_info', {})
                
                relationship_score = 0
                relationship_reasons = []
                
                if target_customer.get('customer_id') and tx_customer.get('customer_id') == target_customer.get('customer_id'):
                    relationship_score += 10
                    relationship_reasons.append('same_customer')
                
                if target_details.get('IP_Address') and tx_details.get('IP_Address') == target_details.get('IP_Address'):
                    relationship_score += 8
                    relationship_reasons.append('same_ip')
                
                target_amount = target_details.get('Transaction_Amount', 0)
                tx_amount = tx_details.get('Transaction_Amount', 0)
                if target_amount > 0 and tx_amount > 0:
                    amount_diff = abs(tx_amount - target_amount) / max(target_amount, 1)
                    if amount_diff < 0.3:
                        relationship_score += 5
                        relationship_reasons.append('similar_amount')
                
                if relationship_score >= 5:
                    if relationship_score >= 15:
                        strength = 'strong'
                    elif relationship_score >= 10:
                        strength = 'medium'
                    else:
                        strength = 'weak'
                    
                    tx_data_clean = convert_numpy_types(tx_data)
                    related.append({
                        'transaction_id': tx_id,
                        'timestamp': tx_data_clean.get('timestamp', ''),
                        'risk_score': tx_data_clean.get('risk_score', 0),
                        'risk_category': tx_data_clean.get('risk_category', ''),
                        'amount': tx_details.get('Transaction_Amount', 0),
                        'customer_info': tx_customer,
                        'status_info': tx_data_clean.get('status', {}),
                        'relationship': {
                            'score': relationship_score,
                            'strength': strength,
                            'reasons': relationship_reasons
                        }
                    })
            
            related.sort(key=lambda x: (-x['relationship']['score'], x['timestamp']), reverse=True)
            return jsonify({
                'status': 'success',
                'message': f'Found {len(related)} related transactions (pickle fallback)',
                'related_transactions': related[:10]
            }), 200
                    
    except Exception as e:
        logger.error(f"Error fetching related transactions: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/v1/api/transactions_delete', methods=['POST'])
@token_required
def delete_transaction(current_user):
    """
    Delete a specific transaction by ID - SQLite first, pickle fallback
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided.'
            }), 400
            
        transaction_id = data.get('transaction_id')
        
        if not transaction_id:
            return jsonify({
                'status': 'error',
                'message': 'transaction_id is required in the request body.'
            }), 400
        
        deleted = False
        deleted_data = None
        
        # Try SQLite first
        try:
            db = SessionLocal()
            tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
            
            if tx:
                deleted_data = {
                    'transaction_id': tx.id,
                    'risk_score': tx.risk_score,
                    'risk_category': tx.risk_category
                }
                db.delete(tx)
                db.commit()
                deleted = True
                logger.info(f"Transaction {transaction_id} deleted from SQLite")
            
            db.close()
            
        except Exception as e:
            logger.warning(f"SQLite delete failed: {e}")
        
        # Also delete from pickle
        try:
            with file_lock:
                transactions = load_or_initialize_pickle(REAL_TIME_RISK_SCORES_PKL, {})
                if transaction_id in transactions:
                    if not deleted_data:
                        deleted_data = {
                            'transaction_id': transaction_id,
                            'risk_score': transactions[transaction_id].get('risk_score'),
                            'risk_category': transactions[transaction_id].get('risk_category')
                        }
                    del transactions[transaction_id]
                    save_to_pickle(transactions, REAL_TIME_RISK_SCORES_PKL)
                    deleted = True
                    logger.info(f"Transaction {transaction_id} deleted from pickle")
        except Exception as e:
            logger.warning(f"Pickle delete failed: {e}")
        
        if not deleted:
            return jsonify({
                'status': 'error',
                'message': f'Transaction with ID {transaction_id} not found.'
            }), 404
        
        return jsonify({
            'status': 'success',
            'message': f'Transaction {transaction_id} deleted successfully.',
            'deleted_transaction': deleted_data,
            'remaining_count': len(load_from_pickle(REAL_TIME_RISK_SCORES_PKL))
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting transaction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/v1/api/fraud_history', methods=['POST'])
@token_required
def get_fraud_history(current_user):
    """
    Get all transactions flagged as High Potential Fraud OR Critical Fraud Risk
    SQLite first, pickle fallback
    """
    try:
        data = request.get_json()
        
        page = data.get('page', 1) if data else 1
        size = data.get('size', 10) if data else 10
        
        # Validate pagination
        if not isinstance(page, int) or page < 1:
            return jsonify({
                'status': 'error',
                'message': 'Page must be an integer greater than 0.'
            }), 400
        
        if not isinstance(size, int) or size < 1 or size > 1000:
            return jsonify({
                'status': 'error',
                'message': 'Size must be an integer between 1 and 1000.'
            }), 400

        # Try SQLite first
        try:
            db = SessionLocal()
            
            # Get total count of fraud transactions
            total = db.query(Transaction).filter(
                Transaction.risk_category.in_(['High Potential Fraud', 'Critical Fraud Risk'])
            ).count()
            
            if total == 0:
                db.close()
                return jsonify({
                    'status': 'success',
                    'message': 'No fraud transactions found.',
                    'fraud_transactions': [],
                    'pagination': {
                        'page': page,
                        'size': size,
                        'total': 0,
                        'total_pages': 0,
                        'has_next': False,
                        'has_prev': False
                    }
                }), 200
            
            # Get paginated results
            fraud_txs = db.query(Transaction)\
                .filter(Transaction.risk_category.in_(['High Potential Fraud', 'Critical Fraud Risk']))\
                .order_by(desc(Transaction.risk_score))\
                .offset((page - 1) * size)\
                .limit(size)\
                .all()
            
            db.close()
            
            total_pages = max(1, (total + size - 1) // size)
            
            fraud_list = []
            for tx in fraud_txs:
                fraud_list.append({
                    'transaction_id': tx.id,
                    'timestamp': tx.timestamp.isoformat(),
                    'risk_score': tx.risk_score,
                    'risk_category': tx.risk_category,
                    'transaction_details': tx.transaction_details,
                    'recommended_action': tx.recommended_action,
                    'explanations': tx.explanations,
                    'customer_info': tx.customer_info,
                    'status_info': tx.status
                })
            
            return jsonify({
                'status': 'success',
                'message': f'Found {total} fraud transactions',
                'fraud_transactions': fraud_list,
                'pagination': {
                    'page': page,
                    'size': size,
                    'total': total,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_prev': page > 1
                }
            }), 200
            
        except Exception as e:
            logger.warning(f"SQLite fraud history failed: {e}")
            # Fallback to pickle
            transactions = load_from_pickle(REAL_TIME_RISK_SCORES_PKL)
            
            if not transactions:
                return jsonify({
                    'status': 'success',
                    'message': 'No transactions found.',
                    'fraud_transactions': [],
                    'pagination': {
                        'page': page,
                        'size': size,
                        'total': 0,
                        'total_pages': 0,
                        'has_next': False,
                        'has_prev': False
                    }
                }), 200
            
            fraud_transactions = {}
            for tx_id, tx_data in transactions.items():
                risk_category = tx_data.get('risk_category', '')
                if risk_category in ['High Potential Fraud', 'Critical Fraud Risk']:
                    fraud_transactions[tx_id] = tx_data
            
            if not fraud_transactions:
                return jsonify({
                    'status': 'success',
                    'message': 'No fraud transactions found.',
                    'fraud_transactions': [],
                    'pagination': {
                        'page': page,
                        'size': size,
                        'total': 0,
                        'total_pages': 0,
                        'has_next': False,
                        'has_prev': False
                    }
                }), 200
            
            fraud_list = []
            for tx_id, tx_data in fraud_transactions.items():
                cleaned_tx_data = convert_numpy_types(tx_data)
                fraud_list.append({
                    'transaction_id': tx_id,
                    'timestamp': cleaned_tx_data.get('timestamp', ''),
                    'risk_score': cleaned_tx_data.get('risk_score', 0),
                    'risk_category': cleaned_tx_data.get('risk_category', ''),
                    'transaction_details': cleaned_tx_data.get('transaction_details', {}),
                    'recommended_action': cleaned_tx_data.get('recommended_action', ''),
                    'explanations': cleaned_tx_data.get('explanations', {}),
                    'customer_info': cleaned_tx_data.get('customer_info', {}),
                    'status_info': cleaned_tx_data.get('status', {})
                })
            
            fraud_list.sort(key=lambda x: (-x['risk_score'], x['timestamp']), reverse=True)
            total = len(fraud_list)
            total_pages = max(1, (total + size - 1) // size)
            
            start_idx = (page - 1) * size
            end_idx = start_idx + size
            paginated_results = fraud_list[start_idx:end_idx]
            
            return jsonify({
                'status': 'success',
                'message': f'Found {total} fraud transactions (pickle fallback)',
                'fraud_transactions': paginated_results,
                'pagination': {
                    'page': page,
                    'size': size,
                    'total': total,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_prev': page > 1
                }
            }), 200

    except Exception as e:
        logger.error(f"Error in fraud-history endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500
   
@app.route("/v1/api/fraud_feedback", methods=["POST"])
@token_required
def fraud_feedback(current_user):
    """
    Endpoint to handle fraud feedback - SQLite first, pickle fallback
    """
    try:
        data = request.json
        transaction_id = data.get("transaction_id")
        feedback = data.get("feedback")  # "false_positive" or "confirmed_fraud"
        signals = data.get("signals")    

        if not all([transaction_id, feedback]):
            return jsonify({"error": "transaction_id and feedback are required"}), 400

        ##Check if transaction exists in SQLite
        tx_exists = False
        try:
            db = SessionLocal()
            tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
            if tx:
                tx_exists = True
            db.close()
        except Exception as e:
            logger.warning(f"SQLite feedback check failed: {e}")

        ##pickle as fallback
        if not tx_exists:
            stored_transactions = load_or_initialize_pickle(REAL_TIME_RISK_SCORES_PKL, {})
            if transaction_id not in stored_transactions:
                return jsonify({
                    "error": f"Transaction with ID {transaction_id} not found in records."
                }), 404
        
        if signals is None:
            # Try SQLite first
            try:
                db = SessionLocal()
                tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
                if tx and tx.transaction_details:
                    signals = tx.transaction_details
                db.close()
            except Exception:
                pass
            
            if signals is None:
                stored_transactions = load_or_initialize_pickle(REAL_TIME_RISK_SCORES_PKL, {})
                transaction_details = stored_transactions[transaction_id].get("transaction_details", {})
                signals = transaction_details

        store_feedback(transaction_id, feedback, signals)
        adapt_weights(signals, feedback)
        
        ##Save to SQLite
        save_feedback_to_db(transaction_id, feedback, signals)

        return jsonify({"message": f"Feedback for transaction {transaction_id} processed successfully !!!!!!"}), 200

    except Exception as e:
        logger.error(f"Error in fraud feedback endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/v1/api/transactions/status', methods=['POST'])
@token_required
def update_transaction_status(current_user):
    """
    Update the status of a transaction (Open, Investigating, Resolved, False Positive)
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided.'
            }), 400
            
        transaction_id = data.get('transaction_id')
        new_status = data.get('status')
        notes = data.get('notes', '')
        action_by = data.get('action_by', 'System')
        
        if not transaction_id or not new_status:
            return jsonify({
                'status': 'error',
                'message': 'transaction_id and status are required.'
            }), 400
        
        ##Validate status
        valid_statuses = ['Open', 'Investigating', 'Resolved', 'False Positive']
        if new_status not in valid_statuses:
            return jsonify({
                'status': 'error',
                'message': f'Invalid status. Must be one of: {", ".join(valid_statuses)}'
            }), 400
        
        transactions = load_from_pickle(REAL_TIME_RISK_SCORES_PKL)
        
        if not transactions:
            return jsonify({
                'status': 'error',
                'message': 'No transactions found.'
            }), 404
        
        # Checking if transaction exists
        if transaction_id not in transactions:
            return jsonify({
                'status': 'error',
                'message': f'Transaction with ID {transaction_id} not found.'
            }), 404
        
        if 'status' not in transactions[transaction_id]:
            transactions[transaction_id]['status'] = {}
        
        # status history
        old_status = transactions[transaction_id].get('status', {}).get('current', 'Open')
        
        transactions[transaction_id]['status'] = {
            'current': new_status,
            'history': transactions[transaction_id].get('status', {}).get('history', []) + [{
                'from': old_status,
                'to': new_status,
                'timestamp': get_nairobi_time(),
                'action_by': action_by,
                'notes': notes
            }],
            'last_updated': get_nairobi_time(),
            'updated_by': action_by
        }
        
        if new_status in ['Resolved', 'False Positive']:
            transactions[transaction_id]['resolution'] = {
                'resolved_by': action_by,
                'resolved_at': get_nairobi_time(),
                'notes': notes,
                'action': 'Blocked' if new_status == 'Resolved' else 'Approved'
            }
        
        save_to_pickle(transactions, REAL_TIME_RISK_SCORES_PKL)
        
        logger.info(f"Transaction {transaction_id} status updated from {old_status} to {new_status} by {action_by}")
        
        return jsonify({
            'status': 'success',
            'message': f'Transaction status updated to {new_status}',
            'transaction_id': transaction_id,
            'new_status': new_status,
            'old_status': old_status,
            'timestamp': get_nairobi_time()
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating transaction status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/v1/api/get_transactions/status', methods=['POST'])
@token_required
def get_transaction_status(current_user):
    """Get transaction status - SQLite first, pickle fallback"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided.'
            }), 400
            
        transaction_id = data.get('transaction_id')
        
        if not transaction_id:
            return jsonify({
                'status': 'error',
                'message': 'transaction_id is required in the request body.'
            }), 400
        
        # Try SQLite first
        try:
            db = SessionLocal()
            tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
            db.close()
            
            if tx:
                return jsonify({
                    'status': 'success',
                    'transaction_id': transaction_id,
                    'current_status': tx.status or 'Open',
                    'last_updated': tx.timestamp.isoformat()
                }), 200
        except Exception as e:
            logger.warning(f"SQLite status read failed: {e}")
        
        # Fallback to pickle
        transactions = load_from_pickle(REAL_TIME_RISK_SCORES_PKL)
        
        if not transactions or transaction_id not in transactions:
            return jsonify({
                'status': 'error',
                'message': f'Transaction with ID {transaction_id} not found.'
            }), 404
        
        status_info = transactions[transaction_id].get('status', {
            'current': 'Open',
            'history': [],
            'last_updated': transactions[transaction_id].get('timestamp')
        })
        
        status_info_clean = convert_numpy_types(status_info)
        
        return jsonify({
            'status': 'success',
            'transaction_id': transaction_id,
            'current_status': status_info_clean.get('current', 'Open'),
            'history': status_info_clean.get('history', []),
            'last_updated': status_info_clean.get('last_updated', transactions[transaction_id].get('timestamp'))
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching transaction status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/v1/api/model_metrics', methods=['GET'])
@token_required
def model_metrics_endpoint(current_user):
    try:
        ##Checking if metrics pickle already exists
        if os.path.exists(MODEL_METRICS_PKL):
         
            metrics_data = load_from_pickle(MODEL_METRICS_PKL)
            logger.info("Model metrics loaded from pickle file")
            
            return jsonify({
                "status": "success",
                "message": "Model metrics loaded from cache successfully !!!!!!!!!!!!",
                "model_version": MODEL_VERSION,
                "national_alert_mode": NATIONAL_ALERT_MODE,
                "threshold": get_active_threshold(),
                "metrics": metrics_data,
                "cached": True
            })
        
        logger.info("Model metrics pickle not found. Calculating metrics...")
        
        #data and load models
        X_train, X_test, y_train, y_test = prepare_and_split_data()
        models = load_model_from_JobLib(RISK_MODELS_JOBLIB)

        metrics = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics[name] = {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred), 4),
                "recall": round(recall_score(y_test, y_pred), 4),
                "f1_score": round(f1_score(y_test, y_pred), 4),
                "roc_auc": round(roc_auc_score(y_test, y_prob), 4)
            }
        
        save_to_pickle(metrics, MODEL_METRICS_PKL)
        logger.info(f"Model metrics saved to pickle file: {MODEL_METRICS_PKL}")

        return jsonify({
            "status": "success",
            "message": "Model metrics calculated and saved successfully !!!!!!!!!!!!",
            "model_version": MODEL_VERSION,
            "national_alert_mode": NATIONAL_ALERT_MODE,
            "threshold": get_active_threshold(),
            "metrics": metrics,
            "cached": False
        })

    except Exception as e:
        logger.error(f"Error in model metrics endpoint: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
 
@app.route('/v1/api/system/alert_mode', methods=['POST'])
@admin_required 
def toggle_alert_mode(current_user):
    global NATIONAL_ALERT_MODE
    data = request.get_json()
    mode = data.get("enable", False)

    NATIONAL_ALERT_MODE = bool(mode)

    return jsonify({
        "status": "success",
        "message": f"National Alert Mode {'enabled' if NATIONAL_ALERT_MODE else 'disabled'} successfully !!!!!!!!!!!!",
        "national_alert_mode": NATIONAL_ALERT_MODE,
        "active_threshold": get_active_threshold()
    })

@app.route('/v1/api/system/sovereign_mode', methods=['POST'])
@admin_required 
def toggle_sovereign_mode(current_user):
    global SOVEREIGN_MODE
    data = request.get_json()
    mode = data.get("enable", False)

    SOVEREIGN_MODE = bool(mode)

    return jsonify({
        "status": "success",
        "message": f"Sovereign Mode {'enabled' if SOVEREIGN_MODE else 'disabled'} successfully !!!!!!!!!!!!",
        "sovereign_mode": SOVEREIGN_MODE,
        "llm_status": "enabled" if SOVEREIGN_MODE else "disabled"   # ← SWAPPED
    }), 200

@app.route('/v1/api/system/sovereign_mode', methods=['GET'])
@token_required
def get_sovereign_mode(current_user):
    """Get current sovereign mode status"""
    return jsonify({
        "status": "success",
        "sovereign_mode": SOVEREIGN_MODE,
        "llm_status": "disabled" if SOVEREIGN_MODE else "enabled"
    }), 200

@app.route('/v1/api/audit_log', methods=['GET'])
@token_required
def get_audit_log(current_user):
    """Endpoint for transparency - show recent decisions"""
    try:
        if not os.path.exists("data/audit_log.json"):
            return jsonify({"logs": []})
        
        with open("data/audit_log.json", "r") as f:
            logs = [json.loads(line) for line in f.readlines()[-100:]]  # Last 100 entries
        
        return jsonify({
            "status": "success",
            "message": f"Retrieved {len(logs)} audit log entries !!!!!!!!!!!!",
            "log_count": len(logs),
            "logs": logs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/v1/api/system/stats', methods=['GET'])
@token_required
def system_stats(current_user):
    """Return system statistics - SQLite first, pickle fallback"""
    try:
        tx_count = 0
        
        # Try SQLite first
        try:
            db = SessionLocal()
            tx_count = db.query(Transaction).count()
            db.close()
        except Exception as e:
            logger.warning(f"SQLite stats failed: {e}")
            # Fallback to pickle
            transactions = load_from_pickle(REAL_TIME_RISK_SCORES_PKL)
            tx_count = len(transactions) if transactions else 0
        
        avg_response = 187  # ms
        
        return jsonify({
            'status': 'success',
            'transactions_analyzed': tx_count,
            'avg_response_ms': avg_response,
            'model_version': MODEL_VERSION,
            'threshold': get_active_threshold(),
            'national_alert_mode': NATIONAL_ALERT_MODE
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/api/db/transactions', methods=['GET'])
@token_required
def get_transactions_from_db(current_user):
    """Get transactions from SQLite database"""

    try:
        db = SessionLocal()
        page = request.args.get('page', 1, type=int)
        size = request.args.get('size', 10, type=int)
        
        transactions = db.query(Transaction)\
            .order_by(desc(Transaction.timestamp))\
            .offset((page - 1) * size)\
            .limit(size)\
            .all()
        
        total = db.query(Transaction).count()
        
        result = []
        for tx in transactions:
            result.append({
                'transaction_id': tx.id,
                'timestamp': tx.timestamp.isoformat(),
                'risk_score': tx.risk_score,
                'risk_category': tx.risk_category,
                'recommended_action': tx.recommended_action,
                'transaction_details': tx.transaction_details,
                'customer_info': tx.customer_info,
                'status': tx.status
            })
        
        db.close()
        
        return jsonify({
            'status': 'success',
            'transactions': result,
            'total': total,
            'page': page,
            'size': size
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/v1/api/queue_status', methods=['GET'])
@token_required
def get_queue_status(current_user):
    """Get current queue status"""
    from concurrent.futures import ThreadPoolExecutor
    
    ###active threads count (approximate)
    active_threads = threading.active_count()
    
    return jsonify({
        'status': 'success',
        'thread_pool_max_workers': ml_executor._max_workers,
        'active_threads': active_threads,
        'async_mode': True,
        'message': f'Can handle {ml_executor._max_workers} concurrent requests'
    })
    
@app.route('/v1/api/db/stats', methods=['GET'])
@token_required
def get_db_stats(current_user):
    """Get statistics from database"""
    try:
        db = SessionLocal()
        
        total = db.query(Transaction).count()
        high_risk = db.query(Transaction).filter(
            Transaction.risk_category.in_(['High Potential Fraud', 'Critical Fraud Risk'])
        ).count()
        pending = db.query(Transaction).filter(Transaction.status == 'Open').count()
        
        db.close()
        
        return jsonify({
            'status': 'success',
            'stats': {
                'total_transactions': total,
                'high_risk_transactions': high_risk,
                'pending_review': pending,
                'fraud_rate': round(high_risk / total * 100, 2) if total > 0 else 0
            }
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
  
@app.route('/v1/api/test', methods=['GET'])
def test():
    return "Testing endpoint, fraud detection apis working effectively !!!!!!!!!!!!"



# ============================================
# FINCA UGANDA FRAUD GUARD ROUTES
# ============================================


def generate_finca_id(prefix):
    """Generate FINCA-style ID"""
    import random
    import string
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = ''.join(random.choices(string.digits, k=4))
    return f"{prefix}{timestamp}_{random_suffix}"

@app.route('/v1/api/finca/health', methods=['GET'])
def finca_health():
    """FINCA health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'FINCA Fraud Guard',
        'timestamp': get_nairobi_time(),
        'engine': 'FinGuardAI v1.0.0',
        'endpoints': {
            'transactions': '/v1/api/finca/transactions',
            'alerts': '/v1/api/finca/alerts',
            'cases': '/v1/api/finca/cases',
            'dashboard': '/v1/api/finca/dashboard'
        }
    })
    
@app.route('/v1/api/finca/transactions', methods=['POST'])
@token_required
def finca_submit_transaction(current_user):
    """
    FINCA Transaction Submission with Rules Evaluation
    Separates ML risk from Rule risk - no mixing!
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        # Accept both lowercase AND capitalized
        customer_id = data.get('customer_id') or data.get('Customer_ID')
        transaction_amount = data.get('transaction_amount') or data.get('Transaction_Amount')
        
        if not customer_id:
            return jsonify({
                'status': 'error',
                'message': 'Missing required field: customer_id or Customer_ID'
            }), 400
        
        if not transaction_amount:
            return jsonify({
                'status': 'error',
                'message': 'Missing required field: transaction_amount or Transaction_Amount'
            }), 400
        
        # Normalize
        data['customer_id'] = customer_id
        data['transaction_amount'] = transaction_amount
        data['Customer_ID'] = customer_id
        data['Transaction_Amount'] = transaction_amount
        
        #FINCA-specific fields
        channel = data.get('channel') or data.get('Channel') or data.get('Transaction_Type') or ''
        device_type = data.get('device_type') or data.get('Device_Type') or ''
        location = data.get('location') or data.get('Location') or data.get('Transaction_Location') or ''
        
        data['channel'] = channel
        data['device_type'] = device_type
        data['location'] = location
        data['Channel'] = channel
        data['Device_Type'] = device_type
        data['Location'] = location
        
        # Run ML engine analysis
        from finca_adapter import get_adapter
        adapter = get_adapter()
        
        tx_id = generate_finca_id('TXN')
        result = adapter.analyze(data)
        
        if result is None:
            return jsonify({
                'status': 'error',
                'message': 'Risk Engine analysis failed'
            }), 500
        
        # 1. ML RISK LEVEL (based on ML score only)
        # ============================================
        ml_score = result['risk_score']
        
        # FINCA risk bands (0-100 scale)
        if ml_score >= 80:
            ml_risk_level = "CRITICAL"
            ml_decision = "BLOCK"
        elif ml_score >= 60:
            ml_risk_level = "HIGH"
            ml_decision = "CHALLENGE"
        elif ml_score >= 30:
            ml_risk_level = "MEDIUM"
            ml_decision = "CHALLENGE"
        else:
            ml_risk_level = "LOW"
            ml_decision = "APPROVE"
        
        # 2. EVALUATE RULES (separate from ML)
        # ============================================
        from finca_rules import evaluate_all_rules
        rule_evaluation = evaluate_all_rules(data)
        
        total_rule_points = rule_evaluation['total_risk_points']
        capped_rule_points = min(total_rule_points, 100)
        
        # Determine rule risk level based on capped points
        if capped_rule_points >= 80:
            rule_risk_level = "CRITICAL"
        elif capped_rule_points >= 60:
            rule_risk_level = "HIGH"
        elif capped_rule_points >= 30:
            rule_risk_level = "MEDIUM"
        else:
            rule_risk_level = "LOW"
        
        # 3. FINAL DECISION (combine ML + Rules)
        # Rules can ESCALATE (BLOCK overrides)
        # ============================================
        final_decision = ml_decision
        final_risk_level = ml_risk_level
        
        if rule_evaluation['triggered_rules']:
            result['rule_triggers'] = rule_evaluation['triggered_rules']
            result['rule_points'] = total_rule_points
            result['capped_rule_points'] = capped_rule_points
            result['rule_risk_level'] = rule_risk_level
            
            # Rules can ONLY escalate
            if rule_evaluation['final_decision'] == 'BLOCK':
                final_decision = 'BLOCK'
                final_risk_level = 'CRITICAL'
            elif rule_evaluation['final_decision'] == 'CHALLENGE' and final_risk_level not in ['CRITICAL', 'HIGH']:
                final_decision = 'CHALLENGE'
                if final_risk_level == 'LOW':
                    final_risk_level = 'MEDIUM'
        
        # Store both ML and final risk levels
        result['ml_risk_level'] = ml_risk_level
        result['ml_score'] = ml_score
        result['risk_level'] = final_risk_level
        result['decision'] = final_decision
        
        # 4. Build customer info
        # ============================================
        customer_info = {
            'customer_id': customer_id,
            'customer_name': data.get('customer_name', data.get('Customer_Name', f"Customer {tx_id[1:9]}")),
            'customer_email': data.get('customer_email', data.get('Customer_Email', '')),
            'customer_phone': data.get('customer_phone', data.get('Customer_Phone', '')),
            'account_age_days': data.get('account_age_days', data.get('Account_Age_Days', 0)),
            'avg_transaction_amount': data.get('avg_transaction_amount', data.get('Avg_Transaction_Amount', 0))
        }
        
        # 5. Build transaction details
        # ============================================
        transaction_details = result.get('transaction_details', {})
        transaction_details['finca_channel'] = channel
        transaction_details['finca_device_type'] = device_type
        transaction_details['finca_location'] = location
        
        # ML risk info to transaction_details
        transaction_details['ml_risk_score'] = ml_score
        transaction_details['ml_risk_level'] = ml_risk_level
        transaction_details['final_risk_level'] = final_risk_level
        
        # rule info to transaction_details (separate from ML)
        if rule_evaluation['triggered_rules']:
            transaction_details['finca_rules_triggered'] = [
                {
                    'rule_id': r.get('rule_id'),
                    'rule_name': r.get('rule_name'),
                    'reason': r.get('reason'),
                    'rule_points': r.get('risk_points')
                } for r in rule_evaluation['triggered_rules']
            ]
            transaction_details['finca_total_rule_points'] = total_rule_points
            transaction_details['finca_capped_rule_points'] = capped_rule_points
            transaction_details['finca_rule_risk_level'] = rule_risk_level
            transaction_details['finca_final_decision'] = rule_evaluation['final_decision']
            transaction_details['finca_rule_count'] = rule_evaluation['rule_count']
        
  
        # 6. Generate explanations (using ML risk level)
        # ============================================
        rule_based_explanation = generate_fraud_explanation(
            risk_score=ml_score,  # ML score only
            risk_category=ml_risk_level,  # ML risk level (not escalated)
            transaction_details=transaction_details
        )
        
        llm_explanation = generate_llm_explanation(
            risk_score=ml_score / 10,
            risk_category=ml_risk_level,  # ML risk level (not escalated)
            transaction_details=transaction_details,
            recommended_action=result.get('recommended_action', '')
        )
        
        final_explanation = llm_explanation if llm_explanation else rule_based_explanation
        

        # 7. alerts and cases (based on final risk level)
        # ============================================
        alert_id = None
        case_id = None
        
        if final_risk_level in ['HIGH', 'CRITICAL']:
            alert_id = generate_finca_id('ALT')
            finca_alerts[alert_id] = {
                'id': alert_id,
                'transaction_id': tx_id,
                'customer_id': customer_id,
                'risk_score': ml_score,
                'ml_risk_level': ml_risk_level,
                'final_risk_level': final_risk_level,
                'rule_risk_level': rule_risk_level if rule_evaluation['triggered_rules'] else None,
                'triggered_rules': result.get('triggered_rules', []),
                'reasons': result.get('reasons', []),
                'decision': final_decision,
                'status': 'NEW',
                'created_at': datetime.now().isoformat(),
                'assigned_to': None
            }
            
            if final_risk_level == 'CRITICAL':
                case_id = generate_finca_id('CASE')
                finca_cases[case_id] = {
                    'id': case_id,
                    'alert_id': alert_id,
                    'customer_id': customer_id,
                    'risk_score': ml_score,
                    'ml_risk_level': ml_risk_level,
                    'final_risk_level': final_risk_level,
                    'status': 'OPEN',
                    'priority': 'URGENT',
                    'assigned_to': None,
                    'notes': [],
                    'timeline': [
                        {
                            'timestamp': datetime.now().isoformat(),
                            'action': 'Case created from critical alert',
                            'actor': 'System'
                        }
                    ],
                    'resolution': None,
                    'created_at': datetime.now().isoformat()
                }
        
        # 8. Save to database
        # ============================================
        db_transaction_data = {
            'transaction_id': tx_id,
            'timestamp': get_nairobi_time(),
            'risk_score': ml_score,
            'risk_category': final_risk_level,
            'transaction_details': transaction_details,
            'customer_info': customer_info,
            'recommended_action': result.get('recommended_action', ''),
            'explanations': {
                'rule_based': rule_based_explanation,
                'llm': llm_explanation,
                'final': final_explanation
            },
            'llm_status': 'connected' if client is not None else 'disconnected',
            'model_version': MODEL_VERSION,
            'threshold_used': get_active_threshold(),
            'national_alert_mode': NATIONAL_ALERT_MODE,
            'feedback_used': None,
            'feedback_effect': None,
            'status': {'current': 'Open', 'history': []}
        }
        
        save_transaction_to_db(db_transaction_data)
        
        with file_lock:
            stored_scores = load_or_initialize_pickle(REAL_TIME_RISK_SCORES_PKL, {})
            stored_scores[tx_id] = db_transaction_data
            save_to_pickle(stored_scores, REAL_TIME_RISK_SCORES_PKL)
        
        log_decision(tx_id, ml_score, final_risk_level, result.get('recommended_action', ''))
        
        finca_transactions[tx_id] = {
            'id': tx_id,
            'customer_id': customer_id,
            'amount': transaction_amount,
            'data': data,
            'result': result,
            'alert_id': alert_id,
            'case_id': case_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # 9. Response - Clear separation of ML vs Rules
        # ============================================
        response = {
            'status': 'success',
            'message': 'Risk score calculated successfully (async mode) !!!!!!!',
            'async_mode': True,
            'async_processing': True,
            'result': {
                'transaction_id': tx_id,
                'timestamp': get_nairobi_time(),
                'risk_score': ml_score,  # ML score only
                'ml_risk_level': ml_risk_level,  # ML-based risk level
                'final_risk_level': final_risk_level,  # Final risk level (after rules)
                'decision': final_decision,
                'transaction_details': transaction_details,
                'customer_info': customer_info,
                'recommended_action': result.get('recommended_action', ''),
                'explanations': {
                    'rule_based': rule_based_explanation,
                    'llm': llm_explanation if llm_explanation else 'LLM not available',
                    'final': final_explanation
                },
                'llm_status': 'connected' if client is not None else 'disconnected',
                'feedback_used': None,
                'feedback_effect': None
            }
        }
        
        response['finca_specific'] = {
            'alert_id': alert_id,
            'case_id': case_id,
            'customer_id': customer_id,
            'customer_name': data.get('customer_name', data.get('Customer_Name', '')),
            'transaction_amount': transaction_amount,
            'channel': channel,
            'device_type': device_type,
            'location': location
        }
        
        logger.info(f"FINCA Transaction {tx_id}: ML Score={ml_score} ({ml_risk_level}), Rules={total_rule_points} (capped={capped_rule_points}, {rule_risk_level}), Final={final_risk_level}, Decision={final_decision}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"FINCA transaction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================
# BATCH TRANSACTION SIMULATOR (FIXED)
# ============================================

@app.route('/v1/api/finca/simulate_batch', methods=['POST'])
@token_required
def simulate_batch_transactions(current_user):
    """
    Simulate multiple transactions - uses SAME structure as finca_submit_transaction
    """
    try:
        data = request.json or {}
        count = data.get('count', 20)
        fraud_ratio = data.get('fraud_ratio', 0.3)
        
        transactions = generate_transactions_for_simulation(count, fraud_ratio)
        
        results = []
        summary = {
            'total': 0,
            'approved': 0,
            'challenged': 0,
            'blocked': 0,
            'alerts': 0,
            'cases': 0,
            'risk_distribution': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0},
            'by_channel': {},
            'by_location': {},
            'by_device': {}
        }
        
        for tx_data in transactions:
            full_response = process_transaction_like_finca(tx_data)
            results.append(full_response)
            
            # Extract from nested structure
            if full_response.get('status') == 'success':
                result_data = full_response.get('result', {})
                finca_specific = full_response.get('finca_specific', {})
                
                summary['total'] += 1
                
                decision = result_data.get('decision', '')
                if decision == 'APPROVE':
                    summary['approved'] += 1
                elif decision == 'CHALLENGE':
                    summary['challenged'] += 1
                elif decision == 'BLOCK':
                    summary['blocked'] += 1
                
                final_risk = result_data.get('final_risk_level', result_data.get('ml_risk_level', 'LOW'))
                if final_risk in summary['risk_distribution']:
                    summary['risk_distribution'][final_risk] += 1
                
                if finca_specific.get('alert_id'):
                    summary['alerts'] += 1
                if finca_specific.get('case_id'):
                    summary['cases'] += 1
                
                channel = finca_specific.get('channel', 'Unknown')
                summary['by_channel'][channel] = summary['by_channel'].get(channel, 0) + 1
                
                location = finca_specific.get('location', 'Unknown')
                summary['by_location'][location] = summary['by_location'].get(location, 0) + 1
                
                device = finca_specific.get('device_type', 'Unknown')
                summary['by_device'][device] = summary['by_device'].get(device, 0) + 1
        
        return jsonify({
            'status': 'success',
            'message': f'Processed {summary["total"]} transactions',
            'summary': summary,
            'transactions': results
        }), 200
        
    except Exception as e:
        logger.error(f"Batch simulation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def process_transaction_like_finca(tx_data):
    """
    Process a single transaction using EXACTLY the same logic and response format as finca_submit_transaction
    """
    try:
        # Generate transaction ID - NOW USING TXN prefix
        tx_id = generate_finca_id('TXN')
        
        # Get customer info
        customer_id = tx_data.get('customer_id', f'CUST-{tx_id[1:9]}')
        transaction_amount = tx_data.get('transaction_amount', 0)
        
        # FINCA-specific fields
        channel = tx_data.get('channel', '')
        device_type = tx_data.get('device_type', '')
        location = tx_data.get('location', '')
        
        # Run ML engine analysis using adapter
        from finca_adapter import get_adapter
        adapter = get_adapter()
        result = adapter.analyze(tx_data)
        
        if result is None:
            return {
                'status': 'error',
                'message': 'Risk Engine analysis failed',
                'transaction_id': tx_id
            }
        
        # 1. ML RISK LEVEL (same as finca_submit_transaction)
        ml_score = result['risk_score']
        
        if ml_score >= 80:
            ml_risk_level = "CRITICAL"
            ml_decision = "BLOCK"
        elif ml_score >= 60:
            ml_risk_level = "HIGH"
            ml_decision = "CHALLENGE"
        elif ml_score >= 30:
            ml_risk_level = "MEDIUM"
            ml_decision = "CHALLENGE"
        else:
            ml_risk_level = "LOW"
            ml_decision = "APPROVE"
        
        # 2. EVALUATE RULES (same as finca_submit_transaction)
        from finca_rules import evaluate_all_rules
        rule_evaluation = evaluate_all_rules(tx_data)
        
        total_rule_points = rule_evaluation['total_risk_points']
        capped_rule_points = min(total_rule_points, 100)
        
        if capped_rule_points >= 80:
            rule_risk_level = "CRITICAL"
        elif capped_rule_points >= 60:
            rule_risk_level = "HIGH"
        elif capped_rule_points >= 30:
            rule_risk_level = "MEDIUM"
        else:
            rule_risk_level = "LOW"
        
        # 3. FINAL DECISION (same as finca_submit_transaction)
        final_decision = ml_decision
        final_risk_level = ml_risk_level
        
        if rule_evaluation['triggered_rules']:
            if rule_evaluation['final_decision'] == 'BLOCK':
                final_decision = 'BLOCK'
                final_risk_level = 'CRITICAL'
            elif rule_evaluation['final_decision'] == 'CHALLENGE' and final_risk_level not in ['CRITICAL', 'HIGH']:
                final_decision = 'CHALLENGE'
                if final_risk_level == 'LOW':
                    final_risk_level = 'MEDIUM'
        
        # Store ML and final risk
        result['ml_risk_level'] = ml_risk_level
        result['ml_score'] = ml_score
        result['risk_level'] = final_risk_level
        result['decision'] = final_decision
        
        # 4. Build customer info (same as finca_submit_transaction)
        customer_info = {
            'customer_id': customer_id,
            'customer_name': tx_data.get('customer_name', f"Customer {tx_id[1:9]}"),
            'customer_email': tx_data.get('customer_email', ''),
            'customer_phone': tx_data.get('customer_phone', ''),
            'account_age_days': tx_data.get('account_age_days', 0),
            'avg_transaction_amount': tx_data.get('avg_transaction_amount', 0)
        }
        
        # 5. Build transaction details (same as finca_submit_transaction)
        transaction_details = result.get('transaction_details', {})
        transaction_details['finca_channel'] = channel
        transaction_details['finca_device_type'] = device_type
        transaction_details['finca_location'] = location
        transaction_details['ml_risk_score'] = ml_score
        transaction_details['ml_risk_level'] = ml_risk_level
        transaction_details['final_risk_level'] = final_risk_level
        
        # Add rule info
        if rule_evaluation['triggered_rules']:
            transaction_details['finca_rules_triggered'] = [
                {
                    'rule_id': r.get('rule_id'),
                    'rule_name': r.get('rule_name'),
                    'reason': r.get('reason'),
                    'rule_points': r.get('risk_points')
                } for r in rule_evaluation['triggered_rules']
            ]
            transaction_details['finca_total_rule_points'] = total_rule_points
            transaction_details['finca_capped_rule_points'] = capped_rule_points
            transaction_details['finca_rule_risk_level'] = rule_risk_level
            transaction_details['finca_final_decision'] = rule_evaluation['final_decision']
            transaction_details['finca_rule_count'] = rule_evaluation['rule_count']
        
        # 6. Generate explanations (same as finca_submit_transaction)
        rule_based_explanation = generate_fraud_explanation(
            risk_score=ml_score,
            risk_category=ml_risk_level,
            transaction_details=transaction_details
        )
        
        # Generate LLM explanation with logging
        logger.info(f"🔄 Generating LLM explanation for transaction {tx_id}...")
        llm_explanation = generate_llm_explanation(
            risk_score=ml_score / 10,
            risk_category=ml_risk_level,
            transaction_details=transaction_details,
            recommended_action=result.get('recommended_action', '')
        )
        
        # Log the result
        if llm_explanation:
            logger.info(f"✅ LLM explanation generated successfully for {tx_id}")
        else:
            logger.warning(f"⚠️ LLM explanation is None for {tx_id}, using rule-based")
        
        # Final explanation - use LLM if available, otherwise fallback to rule-based
        final_explanation = llm_explanation if llm_explanation else rule_based_explanation
        
        # 7. CREATE ALERTS AND CASES - 
        # ============================================
        alert_id = None
        case_id = None
        
        if final_risk_level in ['HIGH', 'CRITICAL']:
            alert_id = generate_finca_id('ALT')
            
            # 
            alert_data = {
                'id': alert_id,
                'transaction_id': tx_id,
                'customer_id': customer_id,
                'risk_score': ml_score,
                'ml_risk_level': ml_risk_level,
                'final_risk_level': final_risk_level,
                'rule_risk_level': rule_risk_level if rule_evaluation['triggered_rules'] else None,
                'triggered_rules': result.get('triggered_rules', []),
                'reasons': result.get('reasons', []),
                'decision': final_decision,
                'status': 'NEW',
                'created_at': datetime.now().isoformat(),
                'assigned_to': None
            }
            
            # Save to in-memory (SAME as finca_submit_transaction)
            finca_alerts[alert_id] = alert_data
            
            # Save to SQLite using the new function
            save_alert_to_db(alert_data)
            
            # Create case if CRITICAL
            if final_risk_level == 'CRITICAL':
                case_id = generate_finca_id('CASE')
                
                # Create case data
                case_data = {
                    'id': case_id,
                    'alert_id': alert_id,
                    'customer_id': customer_id,
                    'risk_score': ml_score,
                    'ml_risk_level': ml_risk_level,
                    'final_risk_level': final_risk_level,
                    'status': 'OPEN',
                    'priority': 'URGENT',
                    'assigned_to': None,
                    'notes': [],
                    'timeline': [
                        {
                            'timestamp': datetime.now().isoformat(),
                            'action': 'Case created from critical alert',
                            'actor': 'System'
                        }
                    ],
                    'resolution': None,
                    'created_at': datetime.now().isoformat()
                }
                
                # Save to in-memory 
                finca_cases[case_id] = case_data
                
                # Save to SQLite using the new function
                save_case_to_db(case_data)
        
        # 8. Save to database
        db_transaction_data = {
            'transaction_id': tx_id,
            'timestamp': get_nairobi_time(),
            'risk_score': ml_score,
            'risk_category': final_risk_level,
            'transaction_details': transaction_details,
            'customer_info': customer_info,
            'recommended_action': result.get('recommended_action', ''),
            'explanations': {
                'rule_based': rule_based_explanation,
                'llm': llm_explanation,  # Save the actual LLM explanation (may be None)
                'final': final_explanation
            },
            'llm_status': 'connected' if client is not None else 'disconnected',
            'model_version': MODEL_VERSION,
            'threshold_used': get_active_threshold(),
            'national_alert_mode': NATIONAL_ALERT_MODE,
            'feedback_used': None,
            'feedback_effect': None,
            'status': {'current': 'Open', 'history': []}
        }
        
        # Save to SQLite
        save_transaction_to_db(db_transaction_data)
        
        # Save to pickle with lock
        with file_lock:
            stored_scores = load_or_initialize_pickle(REAL_TIME_RISK_SCORES_PKL, {})
            stored_scores[tx_id] = db_transaction_data
            save_to_pickle(stored_scores, REAL_TIME_RISK_SCORES_PKL)
        
        # Log decision
        log_decision(tx_id, ml_score, final_risk_level, result.get('recommended_action', ''))
        
        # Store in FINCA in-memory storage
        finca_transactions[tx_id] = {
            'id': tx_id,
            'customer_id': customer_id,
            'amount': transaction_amount,
            'data': tx_data,
            'result': result,
            'alert_id': alert_id,
            'case_id': case_id,
            'timestamp': datetime.now().isoformat()
        }
        

        response = {
            'status': 'success',
            'message': 'Risk score calculated successfully (async mode) !!!!!!!',
            'async_mode': True,
            'async_processing': True,
            'finca_specific': {
                'alert_id': alert_id,
                'case_id': case_id,
                'customer_id': customer_id,
                'customer_name': tx_data.get('customer_name', ''),
                'channel': channel,
                'device_type': device_type,
                'location': location,
                'transaction_amount': transaction_amount
            },
            'result': {
                'transaction_id': tx_id,
                'timestamp': get_nairobi_time(),
                'risk_score': ml_score,
                'ml_risk_level': ml_risk_level,
                'final_risk_level': final_risk_level,
                'decision': final_decision,
                'transaction_details': transaction_details,
                'customer_info': customer_info,
                'recommended_action': result.get('recommended_action', ''),
                'explanations': {
                    'rule_based': rule_based_explanation,
                    'llm': llm_explanation if llm_explanation else 'LLM not available',
                    'final': final_explanation
                },
                'llm_status': 'connected' if client is not None else 'disconnected',
                'feedback_used': None,
                'feedback_effect': None
            }
        }
        
        logger.info(f"Batch Transaction {tx_id}: ML Score={ml_score} ({ml_risk_level}), Rules={total_rule_points}, Final={final_risk_level}, Decision={final_decision}, LLM={'✓' if llm_explanation else '✗'}")
        
        return response
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e),
            'transaction_id': generate_finca_id('ERR')
        }

def generate_transactions_for_simulation(count=20, fraud_ratio=0.3):
    """
    Generate transactions in EXACTLY the same format as your API expects
    with realistic CRITICAL cases
    """
    import random
    from datetime import datetime, timedelta
    
    # Enhanced customer data with more realistic profiles
    customers = [
        {'id': 'CUST-001', 'name': 'John Okello', 'location': 'Kampala', 'avg': 180000, 'phone': '+256-712-345-678', 'email': 'john.okello@email.com', 'account_age': 315},
        {'id': 'CUST-002', 'name': 'Sarah Atim', 'location': 'Kampala', 'avg': 250000, 'phone': '+256-713-456-789', 'email': 'sarah.atim@email.com', 'account_age': 208},
        {'id': 'CUST-003', 'name': 'Peter Ochieng', 'location': 'Nairobi', 'avg': 320000, 'phone': '+254-712-345-678', 'email': 'peter.ochieng@email.com', 'account_age': 144},
        {'id': 'CUST-004', 'name': 'Grace Mbugua', 'location': 'Mombasa', 'avg': 150000, 'phone': '+254-713-456-789', 'email': 'grace.mbugua@email.com', 'account_age': 176},
        {'id': 'CUST-005', 'name': 'David Mwesigwa', 'location': 'Kampala', 'avg': 450000, 'phone': '+256-714-567-890', 'email': 'david.mwesigwa@email.com', 'account_age': 360},
        {'id': 'CUST-006', 'name': 'Faith Akinyi', 'location': 'Kisumu', 'avg': 95000, 'phone': '+254-714-567-890', 'email': 'faith.akinyi@email.com', 'account_age': 805},
        {'id': 'CUST-007', 'name': 'James Omondi', 'location': 'Nakuru', 'avg': 210000, 'phone': '+254-715-678-901', 'email': 'james.omondi@email.com', 'account_age': 889},
        {'id': 'CUST-008', 'name': 'Mary Wanjiru', 'location': 'Nairobi', 'avg': 380000, 'phone': '+254-716-789-012', 'email': 'mary.wanjiru@email.com', 'account_age': 808},
        {'id': 'CUST-009', 'name': 'Robert Kiprop', 'location': 'Eldoret', 'avg': 175000, 'phone': '+254-717-890-123', 'email': 'robert.kiprop@email.com', 'account_age': 193},
        {'id': 'CUST-010', 'name': 'Jane Auma', 'location': 'Kampala', 'avg': 120000, 'phone': '+256-715-678-901', 'email': 'jane.auma@email.com', 'account_age': 95}
    ]
    
    devices = ['iPhone', 'Samsung', 'MacBook', 'Unknown', 'Huawei', 'Tecno']
    locations = ['Kampala', 'Nairobi', 'Mombasa', 'Kisumu', 'Entebbe', 'International', 'Nakuru', 'Eldoret']
    channels = ['MOBILE_BANKING', 'INTERNET_BANKING', 'ATM', 'AGENCY', 'USSD']
    transaction_types = ['Transfer', 'Withdrawal', 'Deposit', 'Payment', 'Bill Payment']
    
    transactions = []
    
    for i in range(count):
        customer = random.choice(customers)
        
        # Determine if fraud with weighted probability
        is_fraud = random.random() < fraud_ratio
        
        # Calculate risk severity level
        # 0 = Low, 1 = Medium, 2 = High, 3 = Critical
        risk_severity = 0
        
        if is_fraud:
            # Fraudulent transaction - more severe
            risk_severity = random.choices([1, 2, 3], weights=[0.1, 0.3, 0.6])[0]
        else:
            # Normal transaction - mostly low/medium
            risk_severity = random.choices([0, 1], weights=[0.7, 0.3])[0]
        
        # Build transaction based on severity
        if risk_severity == 0:  # LOW RISK
            # Normal transaction
            amount = int(random.gauss(customer['avg'], customer['avg'] * 0.25))
            amount = max(10000, min(amount, 2000000))
            device = random.choice(['iPhone', 'Samsung', 'MacBook', 'Huawei', 'Tecno'])
            location = customer['location']
            channel = random.choice(channels)
            hour = random.randint(8, 21)
            frequency = random.randint(1, 3)
            is_weekend = 1 if random.random() < 0.15 else 0
            transaction_type = random.choice(transaction_types)
            ip = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
            account_age = customer['account_age']
            
        elif risk_severity == 1:  # MEDIUM RISK
            # Slightly suspicious
            amount = int(random.gauss(customer['avg'] * 2.5, customer['avg'] * 0.8))
            amount = max(50000, min(amount, 5000000))
            device = random.choice(['iPhone', 'Samsung', 'MacBook', 'Huawei', 'Tecno'])
            # Sometimes unusual location
            location = random.choice([customer['location'], random.choice(locations)])
            channel = random.choice(channels)
            hour = random.randint(6, 23)
            frequency = random.randint(3, 6)
            is_weekend = 1 if random.random() < 0.3 else 0
            transaction_type = random.choice(['Transfer', 'Payment'])
            ip = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
            account_age = random.randint(30, customer['account_age'])
            
        elif risk_severity == 2:  # HIGH RISK
            # Suspicious transaction
            amount = int(random.gauss(customer['avg'] * 5, customer['avg'] * 1.5))
            amount = max(200000, min(amount, 10000000))
            device = random.choice(['Unknown', 'Unknown', 'Unknown', 'iPhone', 'Samsung'])
            location = random.choice(['International', 'International', customer['location'], random.choice(locations)])
            channel = random.choice(['MOBILE_BANKING', 'INTERNET_BANKING'])
            hour = random.choice([1, 2, 3, 4, 5, 22, 23, 0])
            frequency = random.randint(5, 10)
            is_weekend = 1
            transaction_type = 'Transfer'
            ip = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
            account_age = random.randint(5, 30)
            
        else:  # CRITICAL RISK (severity 3)
            # Very suspicious - multiple red flags
            amount = int(random.gauss(customer['avg'] * 15, customer['avg'] * 5))
            amount = max(1000000, min(amount, 25000000))
            device = random.choice(['Unknown', 'Unknown', 'Unknown', 'Unknown', 'Unknown'])
            location = random.choice(['International', 'International', 'International'])
            channel = random.choice(['MOBILE_BANKING', 'INTERNET_BANKING'])
            hour = random.choice([0, 1, 2, 3, 4, 5])
            frequency = random.randint(8, 20)
            is_weekend = 1
            transaction_type = 'Transfer'
            ip = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
            account_age = random.randint(1, 15)
        
        # Build transaction
        tx = {
            # Required fields
            'customer_id': customer['id'],
            'customer_name': customer['name'],
            'customer_email': customer['email'],
            'customer_phone': customer['phone'],
            'transaction_amount': amount,
            
            # FINCA-specific fields
            'device_type': device,
            'location': location,
            'channel': channel,
            
            # Additional fields
            'ip_address': ip,
            'tx_count_last_hour': frequency,
            'account_age_days': account_age,
            'avg_transaction_amount': customer['avg'],
            'Transaction_Hour': hour,
            'Is_Weekend': is_weekend,
            'Day_of_Week': random.randint(0, 6) if not is_weekend else random.randint(5, 6),
            
            # Transaction type
            'transaction_type': transaction_type,
            
            # Metadata for simulation tracking
            '_simulated': True,
            '_is_fraud': is_fraud,
            '_customer_avg': customer['avg'],
            '_risk_severity': risk_severity
        }
        
        transactions.append(tx)
    
    return transactions

# def generate_transactions_for_simulation(count=20, fraud_ratio=0.3):
#     """
#     Generate transactions in EXACTLY the same format as your API expects
#     """
#     import random
#     from datetime import datetime
    
#     # Customer data
#     customers = [
#         {'id': 'CUST-001', 'name': 'John Okello', 'location': 'Kampala', 'avg': 180000, 'phone': '+256-712-345-678', 'email': 'john.okello@email.com'},
#         {'id': 'CUST-002', 'name': 'Sarah Atim', 'location': 'Kampala', 'avg': 250000, 'phone': '+256-713-456-789', 'email': 'sarah.atim@email.com'},
#         {'id': 'CUST-003', 'name': 'Peter Ochieng', 'location': 'Nairobi', 'avg': 320000, 'phone': '+254-712-345-678', 'email': 'peter.ochieng@email.com'},
#         {'id': 'CUST-004', 'name': 'Grace Mbugua', 'location': 'Mombasa', 'avg': 150000, 'phone': '+254-713-456-789', 'email': 'grace.mbugua@email.com'},
#         {'id': 'CUST-005', 'name': 'David Mwesigwa', 'location': 'Kampala', 'avg': 450000, 'phone': '+256-714-567-890', 'email': 'david.mwesigwa@email.com'},
#         {'id': 'CUST-006', 'name': 'Faith Akinyi', 'location': 'Kisumu', 'avg': 95000, 'phone': '+254-714-567-890', 'email': 'faith.akinyi@email.com'},
#         {'id': 'CUST-007', 'name': 'James Omondi', 'location': 'Nakuru', 'avg': 210000, 'phone': '+254-715-678-901', 'email': 'james.omondi@email.com'},
#         {'id': 'CUST-008', 'name': 'Mary Wanjiru', 'location': 'Nairobi', 'avg': 380000, 'phone': '+254-716-789-012', 'email': 'mary.wanjiru@email.com'},
#         {'id': 'CUST-009', 'name': 'Robert Kiprop', 'location': 'Eldoret', 'avg': 175000, 'phone': '+254-717-890-123', 'email': 'robert.kiprop@email.com'},
#         {'id': 'CUST-010', 'name': 'Jane Auma', 'location': 'Kampala', 'avg': 120000, 'phone': '+256-715-678-901', 'email': 'jane.auma@email.com'}
#     ]
    
#     devices = ['iPhone', 'Samsung', 'MacBook', 'Unknown', 'Huawei', 'Tecno']
#     locations = ['Kampala', 'Nairobi', 'Mombasa', 'Kisumu', 'Entebbe', 'International', 'Nakuru', 'Eldoret']
#     channels = ['MOBILE_BANKING', 'INTERNET_BANKING', 'ATM', 'AGENCY', 'USSD']
    
#     transactions = []
    
#     for i in range(count):
#         customer = random.choice(customers)
        
#         # Determine if fraud
#         is_fraud = random.random() < fraud_ratio
        
#         if is_fraud:
#             # Fraudulent transaction
#             amount = random.randint(2000000, 15000000)
#             device = random.choice(['Unknown', 'Unknown', 'Unknown', 'iPhone'])
#             location = random.choice(['International', 'International', 'Nairobi', 'Mombasa'])
#             channel = random.choice(['MOBILE_BANKING', 'INTERNET_BANKING'])
#             hour = random.choice([1, 2, 3, 4, 5, 23, 0])
#             frequency = random.randint(5, 15)
#             is_weekend = 1
#             transaction_type = 'Transfer'
#             ip = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
#         else:
#             # Normal transaction
#             amount = int(random.gauss(customer['avg'], customer['avg'] * 0.3))
#             amount = max(10000, min(amount, 2000000))
#             device = random.choice(['iPhone', 'Samsung', 'MacBook', 'Huawei', 'Tecno'])
#             location = customer['location']
#             channel = random.choice(channels)
#             hour = random.randint(8, 21)
#             frequency = random.randint(1, 3)
#             is_weekend = 1 if random.random() < 0.15 else 0
#             transaction_type = random.choice(['Transfer', 'Withdrawal', 'Deposit', 'Payment'])
#             ip = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        
#         # Build transaction
#         tx = {
#             # Required fields
#             'customer_id': customer['id'],
#             'customer_name': customer['name'],
#             'customer_email': customer['email'],
#             'customer_phone': customer['phone'],
#             'transaction_amount': amount,
            
#             # FINCA-specific fields
#             'device_type': device,
#             'location': location,
#             'channel': channel,
            
#             # Additional fields
#             'ip_address': ip,
#             'tx_count_last_hour': frequency,
#             'account_age_days': random.randint(30, 1095),
#             'avg_transaction_amount': customer['avg'],
#             'Transaction_Hour': hour,
#             'Is_Weekend': is_weekend,
#             'Day_of_Week': random.randint(0, 6),
            
#             # Transaction type
#             'transaction_type': transaction_type,
            
#             # Metadata for simulation tracking
#             '_simulated': True,
#             '_is_fraud': is_fraud,
#             '_customer_avg': customer['avg']
#         }
        
#         transactions.append(tx)
    
#     return transactions

@app.route('/v1/api/finca/simulate_batch/quick', methods=['GET'])
@token_required
def simulate_batch_quick(current_user):
    """
    Quick demo with 20 transactions using the SAME logic
    """
    transactions = generate_transactions_for_simulation(count=20, fraud_ratio=0.3)
    
    results = []
    for tx in transactions:
        result = process_transaction_like_finca(tx)
        results.append(result)
    
    # Fix summary to extract data from nested structure
    summary = {
        'total': len(results),
        'approved': 0,
        'challenged': 0,
        'blocked': 0,
        'alerts': 0,
        'cases': 0,
        'risk_distribution': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0},
        'by_channel': {},
        'by_location': {},
        'by_device': {}
    }
    
    for r in results:
        # Extract from nested structure
        if r.get('status') == 'success':
            result_data = r.get('result', {})
            finca_specific = r.get('finca_specific', {})
            
            # Count decisions
            decision = result_data.get('decision', '')
            if decision == 'APPROVE':
                summary['approved'] += 1
            elif decision == 'CHALLENGE':
                summary['challenged'] += 1
            elif decision == 'BLOCK':
                summary['blocked'] += 1
            
            # Count risk levels
            final_risk = result_data.get('final_risk_level', result_data.get('ml_risk_level', 'LOW'))
            if final_risk in summary['risk_distribution']:
                summary['risk_distribution'][final_risk] += 1
            
            # Count alerts and cases
            if finca_specific.get('alert_id'):
                summary['alerts'] += 1
            if finca_specific.get('case_id'):
                summary['cases'] += 1
            
            # Count by channel
            channel = finca_specific.get('channel', 'Unknown')
            summary['by_channel'][channel] = summary['by_channel'].get(channel, 0) + 1
            
            location = finca_specific.get('location', 'Unknown')
            summary['by_location'][location] = summary['by_location'].get(location, 0) + 1
            
            device = finca_specific.get('device_type', 'Unknown')
            summary['by_device'][device] = summary['by_device'].get(device, 0) + 1
    
    return jsonify({
        'status': 'success',
        'message': f'Processed {summary["total"]} transactions',
        'summary': summary,
        'transactions': results
    }), 200

@app.route('/v1/api/finca/get_transactions', methods=['POST'])
@token_required
def finca_list_transactions(current_user):
    """List FINCA transactions with pagination (POST with JSON body)"""
    try:
        data = request.json or {}
        
        page = int(data.get('page', 1))
        size = int(data.get('size', 20))
        
        if page < 1:
            page = 1
        if size < 1 or size > 100:
            size = 20
        
        tx_list = list(finca_transactions.values())
        tx_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        total = len(tx_list)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated = tx_list[start_idx:end_idx]
        
        return jsonify({
            'status': 'success',
            'transactions': paginated,
            'pagination': {
                'page': page,
                'size': size,
                'total': total,
                'total_pages': (total + size - 1) // size,
                'has_next': end_idx < total,
                'has_prev': page > 1
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing transactions: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/v1/api/finca/alerts', methods=['POST'])
@token_required
def finca_list_alerts(current_user):
    """List FINCA alerts with pagination - SQLite first, in-memory fallback"""
    try:
        data = request.json or {}
        
        status = data.get('status')
        page = int(data.get('page', 1))
        size = int(data.get('size', 1000))
        
        if page < 1:
            page = 1
        if size < 1 or size > 1000:
            size = 1000
        
        # Try SQLite first
        try:
            from database.db_manager import get_alerts_from_db
            alerts, total = get_alerts_from_db(status=status, page=page, size=size)
            
            alert_list = []
            for alert in alerts:
                alert_dict = {
                    'id': alert.id,
                    'transaction_id': alert.transaction_id,
                    'customer_id': alert.customer_id,
                    'risk_score': alert.risk_score,
                    'ml_risk_level': alert.ml_risk_level,
                    'final_risk_level': alert.final_risk_level,
                    'rule_risk_level': alert.rule_risk_level,
                    'triggered_rules': alert.triggered_rules,
                    'reasons': alert.reasons,
                    'decision': alert.decision,
                    'status': alert.status,
                    'created_at': alert.created_at.isoformat() if alert.created_at else None,
                    'assigned_to': alert.assigned_to,
                    'assigned_at': alert.assigned_at.isoformat() if alert.assigned_at else None,
                    'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
                    'resolution_notes': alert.resolution_notes
                }
                alert_list.append(alert_dict)
                
                # ✅ POPULATE IN-MEMORY DICTIONARY
                finca_alerts[alert.id] = alert_dict
            
            return jsonify({
                'status': 'success',
                'message': f'Loaded {len(alert_list)} alerts from SQLite',
                'alerts': alert_list,
                'pagination': {
                    'page': page,
                    'size': size,
                    'total': total,
                    'total_pages': (total + size - 1) // size if total > 0 else 0,
                    'has_next': (page * size) < total if total > 0 else False,
                    'has_prev': page > 1
                }
            }), 200
            
        except Exception as e:
            logger.warning(f"SQLite alerts read failed, falling back to in-memory: {e}")
            # Fallback to in-memory
            alerts = list(finca_alerts.values())
            
            if status:
                alerts = [a for a in alerts if a.get('status') == status]
            
            alerts.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            total = len(alerts)
            start_idx = (page - 1) * size
            end_idx = start_idx + size
            paginated_alerts = alerts[start_idx:end_idx]
            
            return jsonify({
                'status': 'success',
                'message': f'Loaded {len(paginated_alerts)} alerts from memory (fallback)',
                'alerts': paginated_alerts,
                'pagination': {
                    'page': page,
                    'size': size,
                    'total': total,
                    'total_pages': (total + size - 1) // size if total > 0 else 0,
                    'has_next': end_idx < total,
                    'has_prev': page > 1
                }
            }), 200
        
    except Exception as e:
        logger.error(f"Error listing alerts: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/v1/api/finca/alerts/<alert_id>', methods=['GET'])
@token_required
def finca_get_alert(current_user, alert_id):
    """Get single alert - SQLite first, in-memory fallback"""
    try:
        # Try SQLite first
        try:
            from database.db_manager import get_alerts_from_db
            alert = get_alerts_from_db(alert_id=alert_id)
            
            if alert:
                return jsonify({
                    'id': alert.id,
                    'transaction_id': alert.transaction_id,
                    'customer_id': alert.customer_id,
                    'risk_score': alert.risk_score,
                    'ml_risk_level': alert.ml_risk_level,
                    'final_risk_level': alert.final_risk_level,
                    'rule_risk_level': alert.rule_risk_level,
                    'triggered_rules': alert.triggered_rules,
                    'reasons': alert.reasons,
                    'decision': alert.decision,
                    'status': alert.status,
                    'created_at': alert.created_at.isoformat() if alert.created_at else None,
                    'assigned_to': alert.assigned_to,
                    'assigned_at': alert.assigned_at.isoformat() if alert.assigned_at else None
                }), 200
        except Exception as e:
            logger.warning(f"SQLite alert read failed, falling back to in-memory: {e}")
        
        # Fallback to in-memory
        alert = finca_alerts.get(alert_id)
        if not alert:
            return jsonify({
                'status': 'error',
                'message': 'Alert not found'
            }), 404
        return jsonify(alert), 200
        
    except Exception as e:
        logger.error(f"Error getting alert: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/v1/api/finca/case', methods=['POST'])
@token_required
def finca_create_case(current_user):
    """
    Create a new case - POST /v1/api/finca/case
    """
    try:
        data = request.json
        
        # Generate case ID
        case_id = generate_finca_id('CASE')
        
        # Extract data
        alert_id = data.get('alert_id')
        customer_id = data.get('customer_id')
        final_risk_level = data.get('final_risk_level', 'MEDIUM')
        risk_score = data.get('risk_score', 50)
        ml_risk_level = data.get('ml_risk_level', 'MEDIUM')
        status = data.get('status', 'OPEN')
        priority = data.get('priority', 'NORMAL')
        assigned_to = data.get('assigned_to')
        notes = data.get('notes', [])
        timeline = data.get('timeline', [])
        
        # ✅ Get username safely from User object
        username = current_user.username if hasattr(current_user, 'username') else 'System'
        
        # Create case data
        case_data = {
            'id': case_id,
            'alert_id': alert_id,
            'customer_id': customer_id,
            'final_risk_level': final_risk_level,
            'risk_score': risk_score,
            'ml_risk_level': ml_risk_level,
            'status': status,
            'priority': priority,
            'assigned_to': assigned_to,
            'notes': notes,
            'timeline': timeline or [
                {
                    'timestamp': get_nairobi_time(),
                    'action': 'Case created manually by analyst',
                    'actor': username
                }
            ],
            'resolution': None,
            'created_at': get_nairobi_time(),
            'updated_at': get_nairobi_time()
        }
        
        # Save to in-memory storage
        finca_cases[case_id] = case_data
        
        # Save to SQLite database
        from database.db_manager import save_case_to_db
        save_case_to_db(case_data)
        
        logger.info(f"Case {case_id} created by {username}")
        
        return jsonify({
            'status': 'success',
            'message': 'Case created successfully',
            'case': case_data
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating case: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/v1/api/finca/case/<case_id>', methods=['PUT'])
@token_required
def finca_update_case(current_user, case_id):
    """
    Update an existing case - PUT /v1/api/finca/case/<case_id>
    """
    try:
        data = request.json
        
        # Check if case exists
        if case_id not in finca_cases:
            return jsonify({
                'status': 'error',
                'message': 'Case not found'
            }), 404
        
        # Update case
        case = finca_cases[case_id]
        case.update({
            'status': data.get('status', case.get('status')),
            'priority': data.get('priority', case.get('priority')),
            'assigned_to': data.get('assigned_to', case.get('assigned_to')),
            'final_risk_level': data.get('final_risk_level', case.get('final_risk_level')),
            'risk_score': data.get('risk_score', case.get('risk_score')),
            'notes': data.get('notes', case.get('notes', [])),
            'timeline': data.get('timeline', case.get('timeline', [])),
            'updated_at': get_nairobi_time()
        })
        
        # Save to database
        from database.db_manager import update_case_in_db
        update_case_in_db(case_id, case)
        
        return jsonify({
            'status': 'success',
            'message': 'Case updated successfully',
            'case': case
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating case: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/v1/api/finca/case/<case_id>', methods=['DELETE'])
@token_required
def finca_delete_case(current_user, case_id):
    """
    Delete a case - DELETE /v1/api/finca/case/<case_id>
    """
    try:
        if case_id not in finca_cases:
            return jsonify({
                'status': 'error',
                'message': 'Case not found'
            }), 404
        
        # Remove from in-memory
        del finca_cases[case_id]
        
        # Remove from database
        from database.db_manager import delete_case_from_db
        delete_case_from_db(case_id)
        
        logger.info(f"Case {case_id} deleted by {current_user.get('username', 'Analyst')}")
        
        return jsonify({
            'status': 'success',
            'message': 'Case deleted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting case: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
        
@app.route('/v1/api/finca/cases', methods=['POST'])
@token_required
def finca_list_cases(current_user):
    """List FINCA cases with pagination - SQLite first, in-memory fallback"""
    try:
        data = request.json or {}
        
        status = data.get('status')
        page = int(data.get('page', 1))
        size = int(data.get('size', 1000))
        
        if page < 1:
            page = 1
        if size < 1 or size > 1000:
            size = 1000
        
        # Try SQLite first (like transactions_endpoint)
        try:
            from database.db_manager import get_cases_from_db
            cases, total = get_cases_from_db(status=status, page=page, size=size)
            
            case_list = []
            for case in cases:
                case_list.append({
                    'id': case.id,
                    'alert_id': case.alert_id,
                    'customer_id': case.customer_id,
                    'risk_score': case.risk_score,
                    'ml_risk_level': case.ml_risk_level,
                    'final_risk_level': case.final_risk_level,
                    'status': case.status,
                    'priority': case.priority,
                    'assigned_to': case.assigned_to,
                    'notes': case.notes,
                    'timeline': case.timeline,
                    'resolution': case.resolution,
                    'created_at': case.created_at.isoformat() if case.created_at else None,
                    'updated_at': case.updated_at.isoformat() if case.updated_at else None
                })
            
            return jsonify({
                'status': 'success',
                'message': f'Loaded {len(case_list)} cases from SQLite',
                'cases': case_list,
                'pagination': {
                    'page': page,
                    'size': size,
                    'total': total,
                    'total_pages': (total + size - 1) // size if total > 0 else 0,
                    'has_next': (page * size) < total if total > 0 else False,
                    'has_prev': page > 1
                }
            }), 200
            
        except Exception as e:
            logger.warning(f"SQLite cases read failed, falling back to in-memory: {e}")
            # Fallback to in-memory (same as before)
            cases = list(finca_cases.values())
            
            if status:
                cases = [c for c in cases if c.get('status') == status]
            
            cases.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            total = len(cases)
            start_idx = (page - 1) * size
            end_idx = start_idx + size
            paginated_cases = cases[start_idx:end_idx]
            
            return jsonify({
                'status': 'success',
                'message': f'Loaded {len(paginated_cases)} cases from memory (fallback)',
                'cases': paginated_cases,
                'pagination': {
                    'page': page,
                    'size': size,
                    'total': total,
                    'total_pages': (total + size - 1) // size if total > 0 else 0,
                    'has_next': end_idx < total,
                    'has_prev': page > 1
                }
            }), 200
        
    except Exception as e:
        logger.error(f"Error listing cases: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/v1/api/finca/alerts/<alert_id>/read', methods=['POST'])
@token_required
def finca_mark_alert_read(current_user, alert_id):
    """Mark alert as read - SQLite first, in-memory fallback"""
    try:
        alert = None
        
        # === TRY SQLITE FIRST ===
        try:
            from database.db_manager import SessionLocal, Alert
            db = SessionLocal()
            alert_db = db.query(Alert).filter(Alert.id == alert_id).first()
            
            if alert_db:
                # SQLite doesn't have a 'read' column, so we track it via status
                # Or you can add a 'read' column to the Alert model
                # For now, we'll just update status if it's NEW
                if alert_db.status == 'NEW':
                    alert_db.status = 'READ'
                
                db.commit()
                db.refresh(alert_db)
                db.close()
                
                alert = {
                    'id': alert_db.id,
                    'status': alert_db.status,
                    'read': True
                }
                
                # Update in-memory
                if alert_id in finca_alerts:
                    finca_alerts[alert_id]['status'] = alert_db.status
                    finca_alerts[alert_id]['read'] = True
                
                logger.info(f"Alert {alert_id} marked as read by {current_user.username if hasattr(current_user, 'username') else 'System'}")
                
                return jsonify({
                    'status': 'success',
                    'message': 'Alert marked as read',
                    'alert': alert
                }), 200
                
        except Exception as e:
            logger.warning(f"SQLite read failed: {e}")
        
        # === FALLBACK TO IN-MEMORY ===
        if alert_id in finca_alerts:
            alert = finca_alerts[alert_id]
            alert['read'] = True
            if alert.get('status') == 'NEW':
                alert['status'] = 'READ'
            
            return jsonify({
                'status': 'success',
                'message': 'Alert marked as read (memory)',
                'alert': alert
            }), 200
        
        return jsonify({
            'status': 'error',
            'message': f'Alert with ID {alert_id} not found'
        }), 404
        
    except Exception as e:
        logger.error(f"Error marking alert read: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
        
@app.route('/v1/api/finca/cases/<case_id>', methods=['GET'])
@token_required
def finca_get_case(current_user, case_id):
    """Get single case - SQLite first, in-memory fallback"""
    try:
        # Try SQLite first
        try:
            from database.db_manager import get_cases_from_db
            case = get_cases_from_db(case_id=case_id)
            
            if case:
                return jsonify({
                    'id': case.id,
                    'alert_id': case.alert_id,
                    'customer_id': case.customer_id,
                    'risk_score': case.risk_score,
                    'ml_risk_level': case.ml_risk_level,
                    'final_risk_level': case.final_risk_level,
                    'status': case.status,
                    'priority': case.priority,
                    'assigned_to': case.assigned_to,
                    'notes': case.notes,
                    'timeline': case.timeline,
                    'resolution': case.resolution,
                    'created_at': case.created_at.isoformat() if case.created_at else None,
                    'updated_at': case.updated_at.isoformat() if case.updated_at else None
                }), 200
        except Exception as e:
            logger.warning(f"SQLite case read failed, falling back to in-memory: {e}")
        
        # Fallback to in-memory
        case = finca_cases.get(case_id)
        if not case:
            return jsonify({
                'status': 'error',
                'message': 'Case not found'
            }), 404
        return jsonify(case), 200
        
    except Exception as e:
        logger.error(f"Error getting case: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/v1/api/finca/alerts/<alert_id>/assign', methods=['POST'])
@token_required
def finca_assign_alert(current_user, alert_id):
    """Assign alert to analyst - SQLite first, in-memory fallback"""
    try:
        data = request.json
        analyst = data.get('analyst')
        
        if not analyst:
            return jsonify({
                'status': 'error',
                'message': 'Analyst name required'
            }), 400
        
        alert = None
        
        # === TRY SQLITE FIRST ===
        try:
            from database.db_manager import SessionLocal, Alert
            db = SessionLocal()
            alert_db = db.query(Alert).filter(Alert.id == alert_id).first()
            
            if alert_db:
                alert_db.assigned_to = analyst
                alert_db.status = 'ASSIGNED'
                alert_db.assigned_at = datetime.utcnow()
                
                db.commit()
                db.refresh(alert_db)
                db.close()
                
                # Build alert response
                alert = {
                    'id': alert_db.id,
                    'transaction_id': alert_db.transaction_id,
                    'customer_id': alert_db.customer_id,
                    'risk_score': alert_db.risk_score,
                    'ml_risk_level': alert_db.ml_risk_level,
                    'final_risk_level': alert_db.final_risk_level,
                    'rule_risk_level': alert_db.rule_risk_level,
                    'triggered_rules': alert_db.triggered_rules,
                    'reasons': alert_db.reasons,
                    'decision': alert_db.decision,
                    'status': alert_db.status,
                    'created_at': alert_db.created_at.isoformat() if alert_db.created_at else None,
                    'assigned_to': alert_db.assigned_to,
                    'assigned_at': alert_db.assigned_at.isoformat() if alert_db.assigned_at else None
                }
                
                # Update in-memory
                finca_alerts[alert_id] = alert
                
                logger.info(f"Alert {alert_id} assigned to {analyst} in SQLite")
                
                return jsonify({
                    'status': 'success',
                    'message': f'Alert assigned to {analyst}',
                    'alert': alert
                }), 200
                
        except Exception as e:
            logger.warning(f"SQLite assign failed: {e}")
        
        # === FALLBACK TO IN-MEMORY ===
        if alert_id in finca_alerts:
            alert = finca_alerts[alert_id]
            alert['assigned_to'] = analyst
            alert['status'] = 'ASSIGNED'
            alert['assigned_at'] = datetime.now().isoformat()
            
            return jsonify({
                'status': 'success',
                'message': f'Alert assigned to {analyst} (memory)',
                'alert': alert
            }), 200
        
        return jsonify({
            'status': 'error',
            'message': f'Alert with ID {alert_id} not found'
        }), 404
        
    except Exception as e:
        logger.error(f"Error assigning alert: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/v1/api/finca/cases/<case_id>/assign', methods=['POST'])
@token_required
def finca_assign_case(current_user, case_id):
    """Assign case to analyst"""
    try:
        data = request.json
        analyst = data.get('analyst')
        
        if not analyst:
            return jsonify({
                'status': 'error',
                'message': 'Analyst name required'
            }), 400
        
        # ✅ Get username safely
        username = current_user.username if hasattr(current_user, 'username') else 'System'
        
        # Try SQLite first
        try:
            from database.db_manager import SessionLocal, Case
            db = SessionLocal()
            case_db = db.query(Case).filter(Case.id == case_id).first()
            
            if case_db:
                case_db.assigned_to = analyst
                case_db.status = 'INVESTIGATING'
                
                timeline = case_db.timeline or []
                timeline.append({
                    'timestamp': get_nairobi_time(),
                    'action': f'Assigned to {analyst}',
                    'actor': username
                })
                case_db.timeline = timeline
                case_db.updated_at = datetime.utcnow()
                
                db.commit()
                db.refresh(case_db)
                db.close()
                
                # Build case response
                case = {
                    'id': case_db.id,
                    'alert_id': case_db.alert_id,
                    'customer_id': case_db.customer_id,
                    'risk_score': case_db.risk_score,
                    'ml_risk_level': case_db.ml_risk_level,
                    'final_risk_level': case_db.final_risk_level,
                    'status': case_db.status,
                    'priority': case_db.priority,
                    'assigned_to': case_db.assigned_to,
                    'notes': case_db.notes,
                    'timeline': case_db.timeline,
                    'resolution': case_db.resolution,
                    'created_at': case_db.created_at.isoformat() if case_db.created_at else None,
                    'updated_at': case_db.updated_at.isoformat() if case_db.updated_at else None
                }
                
                finca_cases[case_id] = case
                
                return jsonify({
                    'status': 'success',
                    'message': f'Case assigned to {analyst}',
                    'case': case
                }), 200
                
        except Exception as e:
            logger.warning(f"SQLite assign failed: {e}")
        
        # Fallback to in-memory
        if case_id in finca_cases:
            case = finca_cases[case_id]
            case['assigned_to'] = analyst
            case['status'] = 'INVESTIGATING'
            case['timeline'].append({
                'timestamp': get_nairobi_time(),
                'action': f'Assigned to {analyst}',
                'actor': username
            })
            
            return jsonify({
                'status': 'success',
                'message': f'Case assigned to {analyst} (memory)',
                'case': case
            }), 200
        
        return jsonify({
            'status': 'error',
            'message': f'Case with ID {case_id} not found'
        }), 404
        
    except Exception as e:
        logger.error(f"Error assigning case: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
       
@app.route('/v1/api/finca/cases/<case_id>/notes', methods=['POST'])
@token_required
def finca_add_note(current_user, case_id):
    """Add investigation note"""
    try:
        data = request.json
        note = data.get('note')
        analyst = data.get('analyst', 'Analyst')
        
        if not note:
            return jsonify({
                'status': 'error',
                'message': 'Note required'
            }), 400
        
        # ✅ Get username safely
        username = current_user.username if hasattr(current_user, 'username') else 'System'
        
        # Try SQLite first
        try:
            from database.db_manager import SessionLocal, Case
            db = SessionLocal()
            case_db = db.query(Case).filter(Case.id == case_id).first()
            
            if case_db:
                notes = case_db.notes or []
                notes.append({
                    'timestamp': get_nairobi_time(),
                    'analyst': analyst,
                    'note': note
                })
                case_db.notes = notes
                
                timeline = case_db.timeline or []
                timeline.append({
                    'timestamp': get_nairobi_time(),
                    'action': f'Added note: {note[:50]}...',
                    'actor': username
                })
                case_db.timeline = timeline
                case_db.updated_at = datetime.utcnow()
                
                db.commit()
                db.refresh(case_db)
                db.close()
                
                case = {
                    'id': case_db.id,
                    'notes': case_db.notes,
                    'timeline': case_db.timeline,
                    'updated_at': case_db.updated_at.isoformat() if case_db.updated_at else None
                }
                
                if case_id in finca_cases:
                    finca_cases[case_id]['notes'] = case_db.notes
                    finca_cases[case_id]['timeline'] = case_db.timeline
                
                return jsonify({
                    'status': 'success',
                    'message': 'Note added',
                    'case': case
                }), 200
        except Exception as e:
            logger.warning(f"SQLite note failed: {e}")
        
        # Fallback to in-memory
        if case_id in finca_cases:
            case = finca_cases[case_id]
            case['notes'].append({
                'timestamp': get_nairobi_time(),
                'analyst': analyst,
                'note': note
            })
            case['timeline'].append({
                'timestamp': get_nairobi_time(),
                'action': f'Added note: {note[:50]}...',
                'actor': username
            })
            
            return jsonify({
                'status': 'success',
                'message': 'Note added (memory)',
                'case': case
            }), 200
        
        return jsonify({
            'status': 'error',
            'message': f'Case with ID {case_id} not found'
        }), 404
        
    except Exception as e:
        logger.error(f"Error adding note: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/v1/api/finca/cases/<case_id>/resolve', methods=['POST'])
@token_required
def finca_resolve_case(current_user, case_id):
    """Resolve a case"""
    try:
        data = request.json
        resolution = data.get('resolution')
        notes = data.get('notes', '')
        analyst = data.get('analyst', 'Analyst')
        
        if resolution not in ['FRAUD_CONFIRMED', 'FALSE_POSITIVE']:
            return jsonify({
                'status': 'error',
                'message': 'Resolution must be FRAUD_CONFIRMED or FALSE_POSITIVE'
            }), 400
        
        # ✅ Get username safely
        username = current_user.username if hasattr(current_user, 'username') else 'System'
        
        # Try SQLite first
        try:
            from database.db_manager import SessionLocal, Case
            db = SessionLocal()
            case_db = db.query(Case).filter(Case.id == case_id).first()
            
            if case_db:
                case_db.status = 'RESOLVED'
                case_db.resolution = {
                    'verdict': resolution,
                    'notes': notes,
                    'resolved_by': analyst,
                    'resolved_at': get_nairobi_time()
                }
                
                timeline = case_db.timeline or []
                timeline.append({
                    'timestamp': get_nairobi_time(),
                    'action': f'Case resolved: {resolution}',
                    'actor': username
                })
                case_db.timeline = timeline
                case_db.updated_at = datetime.utcnow()
                
                db.commit()
                db.refresh(case_db)
                db.close()
                
                case = {
                    'id': case_db.id,
                    'status': case_db.status,
                    'resolution': case_db.resolution,
                    'timeline': case_db.timeline,
                    'updated_at': case_db.updated_at.isoformat() if case_db.updated_at else None
                }
                
                if case_id in finca_cases:
                    finca_cases[case_id]['status'] = case_db.status
                    finca_cases[case_id]['resolution'] = case_db.resolution
                    finca_cases[case_id]['timeline'] = case_db.timeline
                
                return jsonify({
                    'status': 'success',
                    'message': f'Case resolved as {resolution}',
                    'case': case
                }), 200
        except Exception as e:
            logger.warning(f"SQLite resolve failed: {e}")
        
        # Fallback to in-memory
        if case_id in finca_cases:
            case = finca_cases[case_id]
            case['status'] = 'RESOLVED'
            case['resolution'] = {
                'verdict': resolution,
                'notes': notes,
                'resolved_by': analyst,
                'resolved_at': get_nairobi_time()
            }
            case['timeline'].append({
                'timestamp': get_nairobi_time(),
                'action': f'Case resolved: {resolution}',
                'actor': username
            })
            
            return jsonify({
                'status': 'success',
                'message': f'Case resolved as {resolution} (memory)',
                'case': case
            }), 200
        
        return jsonify({
            'status': 'error',
            'message': f'Case with ID {case_id} not found'
        }), 404
        
    except Exception as e:
        logger.error(f"Error resolving case: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
   
@app.route('/v1/api/finca/dashboard', methods=['GET'])
@token_required
def finca_dashboard(current_user):
    """FINCA Dashboard metrics"""
    
    total_tx = len(finca_transactions)
    total_alerts = len(finca_alerts)
    open_cases = len([c for c in finca_cases.values() if c['status'] in ['OPEN', 'INVESTIGATING']])
    
    # Count by risk level
    risk_dist = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
    for tx in finca_transactions.values():
        level = tx['result'].get('risk_level', 'LOW')
        risk_dist[level] = risk_dist.get(level, 0) + 1
    
    return jsonify({
        'status': 'success',
        'metrics': {
            'total_transactions': total_tx,
            'total_alerts': total_alerts,
            'open_cases': open_cases,
            'blocked': len([t for t in finca_transactions.values() if t['result'].get('decision') == 'BLOCK'])
        },
        'risk_distribution': risk_dist
    })
    
if __name__ == '__main__':
    logger.info("Starting FINCA Fraud Guard API...")
    logger.info("  - Main API: http://localhost:5001/v1/api")
    logger.info("  - FINCA API: http://localhost:5001/v1/api/finca")
    logger.info("  - Adapter: FINCAAdapter loaded")
    app.run(debug=True, host='0.0.0.0', port=5001)