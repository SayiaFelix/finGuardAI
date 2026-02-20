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

#####################################################  Helper functions #####################################################################

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
    
    # predictions from all models
    predictions = []
    probabilities = []
    
    for name, model in models.items():
        # probability of fraud (class 1)
        prob = model.predict_proba(transaction_features.values.reshape(1, -1))[:, 1][0]
        probabilities.append(prob)
        
        # binary prediction
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

def generate_transaction_id():
    random_letter = random.choice(string.ascii_uppercase)  
    date_str = datetime.now().strftime("%Y%m%d")  
    random_digits = f"{random.randint(0, 9999):04d}" 
    return f"T{random_letter}{date_str}{random_digits}I" 

def generate_llm_explanation(
    risk_score,
    risk_category,
    transaction_details,
    recommended_action
):
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
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": "You are a financial fraud analyst explaining risk decisions to banking customers. Be clear, concise, and reassuring in a customer-friendly way. NB: Do NOT mention machine learning or models explicitly. Always use KES instead of $. "},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,  ### Lower temperature for more consistent outputs
            max_tokens=200 
        )
        
        explanation = response.choices[0].message.content.strip()
        logger.info(f"Groq LLM explanation generated successfully")
        return explanation


    except Exception as e:
        logger.warning(f"LLM explanation failed: {e}")
        return None
  
   
def build_llm_prompt(
    risk_score,
    risk_category,
    transaction_details,
    recommended_action
):
    return f"""
        You are a financial fraud explanation assistant for a bank.

        Explain the transaction decision clearly, professionally, and calmly.
        Do NOT alarm the customer unnecessarily.
        Do NOT mention machine learning or models explicitly.

        Transaction summary:
        - Risk Score: {risk_score}/10
        - Risk Category: {risk_category}
        - Transaction Amount: {transaction_details.get("Transaction_Amount")}
        - Model Agreement: {transaction_details.get("Model_Agreement")}

        Required output:
        1. Brief explanation (2–3 sentences)
        2. Why this risk level makes sense
        3. What action (if any) is recommended

        Tone:
        - Clear
        - Trustworthy
        - Customer-friendly
        """


def generate_fraud_explanation(risk_score, risk_category, transaction_details):
    """
    Generates a human-readable explanation for ALL risk categories
    (Low, Medium, High, Critical).
    """

    amount = transaction_details.get("Transaction_Amount", 0)
    model_agreement = transaction_details.get(
        "Model_Agreement", "multiple models evaluated this transaction"
    )

    # -----------------------------
    # Build contributing signals
    # -----------------------------
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

    # Default 
    if not signals:
        signals.append("normal transaction behavior patterns")

    signals_text = ", ".join(signals)

    # -----------------------------
    # Risk-aware explanation tone
    # -----------------------------
    if risk_category == "Low Potential Fraud":
        explanation = (
            f"This transaction was assessed as **Low Risk** with a risk score of "
            f"{round(risk_score, 2)}. The transaction aligns closely with the "
            f"customer’s typical behavior and historical transaction patterns. "
            f"Only minimal risk indicators were observed, including {signals_text}. "
            f"As a result, the transaction was approved while remaining under routine monitoring."
        )

    elif risk_category == "Medium Risk":
        explanation = (
            f"This transaction was classified as **Medium Risk** with a risk score of "
            f"{round(risk_score, 2)}. While the transaction does not strongly indicate fraud, "
            f"the system detected {signals_text}, which slightly deviates from normal patterns. "
            f"As a precaution, additional verification is recommended to confirm transaction legitimacy."
        )

    elif risk_category == "High Potential Fraud":
        explanation = (
            f"This transaction was flagged as **High Potential Fraud** with a risk score of "
            f"{round(risk_score, 2)}. The system detected {signals_text}, along with behavioral patterns "
            f"that differ significantly from the customer’s historical activity. "
            f"These indicators are consistent with known fraud scenarios observed across similar accounts. "
            f"Immediate review by the fraud investigation team is recommended. "
            f"({model_agreement})."
        )

    else:  # Critical Fraud Risk
        explanation = (
            f"This transaction was identified as **Critical Fraud Risk** with a risk score of "
            f"{round(risk_score, 2)}. Strong risk signals were detected, including {signals_text}, "
            f"and a high level of consensus among fraud detection models. "
            f"The observed patterns closely resemble confirmed fraud cases, posing a significant threat "
            f"of financial loss. As a result, the transaction was blocked automatically and escalated "
            f"for immediate investigation. ({model_agreement})."
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
        # Loaddig current weights from pickle
        weights_df = load_from_pickle(weights_file)
        
        if weights_df.empty:
            logger.error(f"Weights DataFrame is empty from {weights_file}")
            weights_df = calculate_feature_importance_weights()
        
        #Converting to dict for easier update
        weights_map = weights_df['Combined_Weight'].to_dict()
        
        # Get selected features to ensure we only update relevant ones
        selected_features = load_from_pickle(IMPORTANT_FEATURES_PKL)
        
        ###adaptive step size (can be adjusted)
        step_size = 0.02  
        
        logger.info(f"Adapting weights for feedback: {feedback}")
        logger.info(f"Features in transaction: {list(transaction_features.keys())[:5]}...")
        
        # Update weights for features present in this transaction
        updated_count = 0
        for feature in selected_features:
            if feature in transaction_features:
                current_weight = weights_map.get(feature, 0)
                
                if feedback == "confirmed_fraud":
                    # Increase weight for features
                    new_weight = min(current_weight + step_size, 1.0)
                    weights_map[feature] = new_weight
                    updated_count += 1
                    
                elif feedback == "false_positive":
                    # Decrease weight for features
                    new_weight = max(current_weight - step_size, 0.0)
                    weights_map[feature] = new_weight
                    updated_count += 1
        
        # Updating  the DataFrame
        weights_df['Combined_Weight'] = weights_df.index.map(lambda f: weights_map.get(f, weights_df.loc[f, 'Combined_Weight'] if f in weights_df.index else 0))
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
    avg_amount=50000,
    tx_count_last_hour=1
):
    """
    Layer 3 Lite:
    Real-time risk adjustment using simple behavioral signals.
    """

    # Amount anomaly (0–1)
    amount_risk = min(abs(transaction_amount - avg_amount) / max(avg_amount, 1), 1)

    # Velocity risk (0–1)
    velocity_risk = min(tx_count_last_hour / 5, 1)

    # Combine with base score (base_risk_score is 0–10)
    adjusted_score = (
        0.6 * (base_risk_score / 10) +
        0.25 * amount_risk +
        0.15 * velocity_risk
    )

    final_score = round(min(adjusted_score * 10, 10), 2)

    signals = {
        "amount_risk": round(amount_risk, 3),
        "velocity_risk": round(velocity_risk, 3)
    }

    return final_score, signals

#########################################################################################################################################
######################################## -------------------- APIS End Points ------------------------###################################
#########################################################################################################################################

@app.route('/v1/api/data_preparation', methods=['POST'])
def prepare_data_endpoint():
    # Ensure a file path is provided in the request
    filename = request.json.get("filename")
    if not filename:
        return jsonify({"error": "filename not provided"}), 400

    file_path = os.path.join(DATA_DIR, filename)

    if os.path.exists(DATA_WRANGLE_PKL):
        try:
            # Loadding the data from pickle if it exists
            processed_data = load_from_pickle(DATA_WRANGLE_PKL)
            return jsonify({
                "status": "success",
                "message": "Data loaded from pickle.",
                "data_shape": processed_data.shape
            })
        except Exception as e:
            logger.error(f"Error loading processed data from pickle: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        # Perform data preparation if pickle doesn't exist
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


@app.route('/v2/api/feature_selection', methods=['GET'])
def feature_selection():
    """ Endpoint for performing feature selection """
    try:
        # features pickle file exists
        if os.path.exists(IMPORTANT_FEATURES_PKL):
            # Load the saved features if the file exists
            overall_selected_features = load_from_pickle(IMPORTANT_FEATURES_PKL)
            logger.info("Data Loaded from the pickle file !!!!!!!")
            
            return jsonify({
                'status': 'success',
                'message': 'Loaded selected features from pickle file.',
                'selected_features': overall_selected_features
            })
        else:
            logger.info("Important Feature pickle file not found. Running Feature Selection !!!!!!!!!!!!!!.")

            data = load_from_pickle(DATA_WRANGLE_PKL)
            
            # Perform feature selection only if data is loaded successfully and is a DataFrame
            if isinstance(data, pd.DataFrame):
                # Split data into features (X) and target (y)
                X = data.drop('Class', axis=1) 
                y = data['Class']
                
                # Get original feature names
                original_columns = X.columns.tolist()
                logger.info('Original Columns from our DataFrame !!!!!!!', original_columns)
                
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
   

@app.route('/v2/api/load_feature_importance', methods=['GET'])
def load_model_endpoint():
    """ Endpoint for loading the model """
    try:
        
        selected_features = load_from_pickle('important_features.pkl')
        
        return jsonify({
            "status": "success",
            "message": 'The selected features loaded successfully !!!!!!!', 
            "selected_features": selected_features
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
     

@app.route('/v2/api/feature_importance_weight', methods=['GET'])
def feature_importance_weight_endpoint():
    """ Endpoint for loading or calculating feature importance weights """
    try:
        # Checking if the feature importance pickle file exists
        if os.path.exists(IMPORTANT_FEATURES_WEIGHTS_PKL):
            # Load the saved feature importance weights 
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

@app.route('/v2/api/batch_risk_scores', methods=['POST'])
def normalized_scores_endpoint():
    """Endpoint for loading or calculating normalized scores sample"""
    try:
        # Parse JSON request body
        data = request.get_json()
        page = data.get('page', 1)  # Default to page 1
        size = data.get('size', 10)  # Default to 10 items per page

        # Validate that page and size are positive integers
        if not isinstance(page, int) or not isinstance(size, int) or page < 1 or size < 1:
            return jsonify({
                'status': 'error',
                'message': 'Page and size must be positive integers.'
            }), 400

        # If the normalized scores pickle file exists
        if os.path.exists(NORMALIZED_RISK_SCORES_PKL):
            # Load the saved normalized scores from pickle file
            normalized_scores_df = load_from_pickle(NORMALIZED_RISK_SCORES_PKL)
        else:
            # If the pickle file doesn't exist, calculate the normalized scores
            normalized_scores_df = normalize_and_categorize_risk_scores()
            save_to_pickle(normalized_scores_df, NORMALIZED_RISK_SCORES_PKL)

        # Convert DataFrame to dictionary
        normalized_scores_dict = normalized_scores_df.to_dict(orient='index')

        # Implement pagination
        total_records = len(normalized_scores_dict)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated_data = dict(list(normalized_scores_dict.items())[start_idx:end_idx])

        # Return paginated response
        return jsonify({
            'status': 'success',
            'message': 'Normalized scores retrieved successfully.',
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




@app.route('/v1/api/test', methods=['GET'])
def test():
    return "Testing endpoint, fraud detection apis working effectively !!!!!!!!!!!!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
