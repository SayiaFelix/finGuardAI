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

GROQ_API_KEY = os.getenv("OPENAI_API_KEY")

if GROQ_API_KEY:
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"  
    )
    logger.info(f"Groq API Key (first 8): {GROQ_API_KEY[:8]}")
    logger.info(f"Groq Status: CONNECTED ✓")
else:
    client = None
    logger.warning("Groq Status: DISCONNECTED ✗ - No API key found")
     

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

        data = prepare_data(file_path)
        logger.info('Data loaded', extra={'columns': data.columns.tolist()})
        
        # Extract target variable
        y = data['Class']
        
        # Load selected features from pickle
        overall_selected_features = load_from_pickle(IMPORTANT_FEATURES_PKL)
        logger.info('Loaded selected features from pickle', extra={'features': overall_selected_features})
        
        # Prepare feature
        X = data[overall_selected_features]
        logger.info('Prepared feature set', extra={'features': X.columns.tolist()})
        
        # Splitting data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        logger.info(f'Training set size: {X_train.shape}, Test set size: {X_test.shape}')

        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error in preparing and splitting data: {str(e)}")
        raise


# Function to calculate feature importance weight
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

    # DF for feature importances
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
    
    # Setting the index to Feature
    weights_df.set_index('Feature', inplace=True)

    # Normalize the weights
    weights_df['RF_Importance'] = weights_df['RF_Importance'] / weights_df['RF_Importance'].sum()
    # weights_df['Lasso_Coefficients'] = weights_df['Lasso_Coefficients'] / np.abs(weights_df['Lasso_Coefficients']).sum()
    weights_df['XGB_Importance'] = weights_df['XGB_Importance'] / weights_df['XGB_Importance'].sum()

    # Combine weights (average for simplicity)
    weights_df['Combined_Weight'] = (weights_df['RF_Importance'] +
                                      weights_df['XGB_Importance']) / 2

    # Sort the features by combined weight in descending order
    sorted_weights = weights_df.sort_values(by='Combined_Weight', ascending=False)
    print('\n\n================================ Sorted Weights ===========================================')
    print(sorted_weights)
    print('================================================================================================')

    save_to_pickle(sorted_weights, IMPORTANT_FEATURES_WEIGHTS_PKL)
    logger.info('IMPORTANT_FEATURES_WEIGHTS pickle Saved Successfully !!!!!!')
    
    return sorted_weights

#  normalize_and_categorize_risk_scores
def normalize_and_categorize_risk_scores():
  
    X_train, X_test, y_train, y_test = prepare_and_split_data()

    # Fitting each model to the training data
    print('Training models, Saving and Calculating risk scores...')
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f'{name} model saved as {name}_model.joblib successfully!!!!!!!!!')
    
    # Saving all models 
    save_model_to_JobLib(models, RISK_MODELS_JOBLIB)
    print(f"All models saved in '{RISK_MODELS_JOBLIB}' successfully!!!!!")
    
    results_df = pd.DataFrame(X_test)
    results_df['True_Label'] = y_test

    # Placeholder for aggregated scores
    aggregated_scores = np.zeros(len(X_test))

    # Risk scores for each model and average them
    for name, model in models.items():
        # Predict probabilities (risk scores)
        risk_scores = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        print('Model', name,'risk scores ===========>', risk_scores)
        aggregated_scores += risk_scores
        
    aggregated_scores /= len(models)

    # # Normalizing aggregated scores
    # min_score = aggregated_scores.min()
    # max_score = aggregated_scores.max()
    
    # normalized_scores = 100 * (aggregated_scores - min_score) / (max_score - min_score)
  
    normalized_scores = np.clip(aggregated_scores * 100, 0, 10)

    # print('Risk score =========>', normalized_scores)
    # print("Max Score:", max_score)
    # print("Min Score:", min_score)

    results_df['Average_Risk_Score'] = aggregated_scores
    results_df['Normalized_Risk_Score'] = normalized_scores
    results_df['Fraud_Prediction'] = (normalized_scores >= 5).astype(int)

    # Risk categories using binning
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
    """
    
    overall_selected_features = load_from_pickle(IMPORTANT_FEATURES_PKL)
    
    # Ensure transaction has all required features
    for feature in overall_selected_features:
        if feature not in transaction:
            transaction[feature] = 0
    
    transaction_features = transaction[overall_selected_features]
    
    # Get predictions from all models
    predictions = []
    probabilities = []
    
    for name, model in models.items():
        # Get probability of fraud (class 1)
        prob = model.predict_proba(transaction_features.values.reshape(1, -1))[:, 1][0]
        probabilities.append(prob)
        
        # Get binary prediction
        pred = model.predict(transaction_features.values.reshape(1, -1))[0]
        predictions.append(pred)
    
    # Calculate ensemble scores
    avg_probability = np.mean(probabilities)
    fraud_votes = sum(predictions)
    total_models = len(models)
    
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
            feature_score += 2 # lets add 2 on high-risk feature
    
    # Normalize feature score
    normalized_feature_score = min(feature_score / 10, 1.0)
    
    weighted_score = sum(
            transaction_features[f] * weights_map.get(f, 0)
            for f in transaction_features.index
        )
    
   # Combining scores: 60% model probability, 20% voting, 20% features
    final_score = (0.6 * avg_probability + 
                   0.2 * (fraud_votes / total_models) + 
                   0.2 * normalized_feature_score)
    
    # Scaling to 0-10 range
    risk_score = round(final_score * 10, 2)
    
    # Risk categorization and recommended actions classification
    if risk_score >= 7:
        risk_category = "Critical Fraud Risk"
        recommended_action = "Block transaction immediately and notify authorities."
    elif risk_score >= 5:
        risk_category = "High Potential Fraud"
        recommended_action = "Flag for review and escalate to fraud investigation team."
    elif risk_score >= 3:
        risk_category = "Medium Risk"
        recommended_action = "Require additional verification (2FA)."
    else:
        risk_category = "Low Potential Fraud"
        recommended_action = "Approve transaction with monitoring."
    
    transaction_details = {
        'Transaction_Amount': transaction.get('Transaction_Amount', 0),
        'Risk_Score': risk_score,
        'Model_Agreement': f"{fraud_votes}/{total_models} models flagged as fraud"
    }
    
    print(f"\n{'='*60}")
    print(f"Transaction Risk Assessment")
    print(f"{'='*60}")
    print(f"Risk Score: {risk_score}/10")
    print(f"Risk Category: {risk_category}")
    print(f"Model Probability Average: {avg_probability:.3f}")
    print(f"Models Flagging as Fraud: {fraud_votes}/{total_models}")
    print(f"High-Risk Features Detected: {feature_score/2}")
    print(f"Recommended Action: {recommended_action}")
    print(f"{'='*60}")
    
    return risk_score, risk_category, transaction_details, recommended_action








@app.route('/v2/api/test', methods=['GET'])
def test():
    return "Testing endpoint, fraud detection apis working effectively !!!!!!!!!!!!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
