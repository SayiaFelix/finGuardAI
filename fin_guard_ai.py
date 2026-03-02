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

    # Scale Transaction_Hour
    robust_scaler = RobustScaler()
    data['Transaction_Hour'] = robust_scaler.fit_transform(data['Transaction_Hour'].values.reshape(-1, 1))

    # One-hot encode categorical columns
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

        y = data['Class']
        
        overall_selected_features = load_from_pickle(IMPORTANT_FEATURES_PKL)
        logger.info('Loaded selected features from pickle', extra={'features': overall_selected_features})
        
        X = data[overall_selected_features]
        logger.info('Prepared feature set', extra={'features': X.columns.tolist()})
        
        # Split data into training and testing sets
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

    print('Training models, Saving and Calculating risk scores...')
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f'{name} model saved as {name}_model.joblib successfully!!!!!!!!!')

    save_model_to_JobLib(models, RISK_MODELS_JOBLIB)
    print(f"All models saved in '{RISK_MODELS_JOBLIB}' successfully!!!!!")
    
    # Initialize DataFrame to hold results
    results_df = pd.DataFrame(X_test)
    results_df['True_Label'] = y_test

    aggregated_scores = np.zeros(len(X_test))

    for name, model in models.items():
        # Predict probabilities (risk scores)
        risk_scores = model.predict_proba(X_test)[:, 1]  
        print('Model', name,'risk scores ===========>', risk_scores)
        aggregated_scores += risk_scores
        
    # Average the scores across all models
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
    
        prob = model.predict_proba(transaction_features.values.reshape(1, -1))[:, 1][0]
        probabilities.append(prob)
        
        # Get binary prediction
        pred = model.predict(transaction_features.values.reshape(1, -1))[0]
        predictions.append(pred)
    
    #ensemble scores
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
            feature_score += 2 
    
    # Normalize feature score
    normalized_feature_score = min(feature_score / 10, 1.0)
    
    weighted_score = sum(
            transaction_features[f] * weights_map.get(f, 0)
            for f in transaction_features.index
        )
    
   # Combine scores: 60% model probability, 20% voting, 20% features
    final_score = (0.6 * avg_probability + 
                   0.2 * (fraud_votes / total_models) + 
                   0.2 * normalized_feature_score)
    
    # Scale to 0-10 range
    risk_score = round(final_score * 10, 2)
    
    # Risk categorization
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
            temperature=0.5, 
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
        weights_df = load_from_pickle(weights_file)
        
        # Ensure weights_df is not empty
        if weights_df.empty:
            logger.error(f"Weights DataFrame is empty from {weights_file}")
            weights_df = calculate_feature_importance_weights()
 
        weights_map = weights_df['Combined_Weight'].to_dict()
        
        # Get selected features to ensure we only update relevant ones
        selected_features = load_from_pickle(IMPORTANT_FEATURES_PKL)
        
        # Define adaptive step size (can be adjusted)
        step_size = 0.02 
        
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
                    
                elif feedback == "false_positive":
                  
                    new_weight = max(current_weight - step_size, 0.0)
                    weights_map[feature] = new_weight
                    updated_count += 1
        
        # Update the DataFrame with new weights
        weights_df['Combined_Weight'] = weights_df.index.map(lambda f: weights_map.get(f, weights_df.loc[f, 'Combined_Weight'] if f in weights_df.index else 0))
        
        # Normalize to ensure weights sum to 1
        weights_df['Combined_Weight'] = weights_df['Combined_Weight'] / weights_df['Combined_Weight'].sum()
        
        save_to_pickle(weights_df, weights_file)
        
        logger.info(f"Adaptive weights updated: {updated_count} features adjusted for {feedback}")
        logger.info(f"Updated weight range: [{weights_df['Combined_Weight'].min():.4f}, {weights_df['Combined_Weight'].max():.4f}]")
        
        return weights_map

    except Exception as e:
        logger.error(f"Error updating adaptive weights: {str(e)}")
        # Return original weights to avoid breaking the system
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

@app.route('/v2/api/data_preparation', methods=['POST'])
def prepare_data_endpoint():
    # Ensure a file path is provided in the request
    filename = request.json.get("filename")
    if not filename:
        return jsonify({"error": "filename not provided"}), 400

    file_path = os.path.join(DATA_DIR, filename)

    if os.path.exists(DATA_WRANGLE_PKL):
        try:
       
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

            if isinstance(data, pd.DataFrame):
               
                X = data.drop('Class', axis=1) 
                y = data['Class']
                
                ### Get original feature names
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

@app.route('/v2/api/batch_risk_scores', methods=['POST'])
def normalized_scores_endpoint():
    """Endpoint for loading or calculating normalized scores sample"""
    try:
        # Parse JSON request body
        data = request.get_json()
        page = data.get('page', 1) 
        size = data.get('size', 10) 

        # Validate that page and size are positive integers
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

@app.route('/v2/api/real_time_risk_score', methods=['POST'])
def real_time_risk_score_endpoint():
    """Endpoint for real-time calculation and storage of risk scores with JSON feedback integration."""
    try:
   
        data = request.json
        transaction_id = data.get("transaction_id") or generate_transaction_id()
        transaction_data = pd.Series(data, name=transaction_id)
        timestamp = datetime.now().isoformat()

        models = load_model_from_JobLib(RISK_MODELS_JOBLIB)
        weights_pickle = load_from_pickle(IMPORTANT_FEATURES_WEIGHTS_PKL)
        weights_map = weights_pickle['Combined_Weight']

        stored_scores = load_or_initialize_pickle(REAL_TIME_RISK_SCORES_PKL, {})
        stored_feedback = load_feedback()  # JSON feedback

        existing_feedback = None
        for fb in stored_feedback:
            fb_signals = fb.get("signals", {})
            # Use 5% tolerance for transaction amount similarity
            if abs(transaction_data.get("Transaction_Amount", 0) - fb_signals.get("Transaction_Amount", 0)) / max(fb_signals.get("Transaction_Amount", 1), 1) < 0.05:
                existing_feedback = fb
                break

        baseline_score, baseline_category, baseline_details, baseline_action = real_time_risk_scoring(
            transaction_data, models, weights_map
        )

        avg_amount = 50000  # for demo default
        tx_count_last_hour = data.get("tx_count_last_hour", 1)

        adjusted_score, layer3_signals = layer3_lite_adjustment(
            base_risk_score=baseline_score,
            transaction_amount=transaction_data.get("Transaction_Amount", 0),
            avg_amount=avg_amount,
            tx_count_last_hour=tx_count_last_hour
        )
        
        ## Override risk score ONLY if Layer 3 increases risk meaningfully
        if adjusted_score > baseline_score:
            risk_score = adjusted_score
            risk_category = (
                "Critical Fraud Risk" if risk_score >= 7 else
                "High Potential Fraud" if risk_score >= 5 else
                "Medium Risk" if risk_score >= 3 else
                "Low Potential Fraud"
            )
        else:
            risk_score = baseline_score
            risk_category = baseline_category
            
        feedback_effect = None

        # --- Adapt weights if feedback exists and recalc ---
        if existing_feedback:
            print(f"Similar transaction found in feedback (ID: {existing_feedback['transaction_id']}). Adapting weights...")
            adapted_weights = adapt_weights(transaction_data, existing_feedback['outcome'], IMPORTANT_FEATURES_WEIGHTS_PKL)

            ### Recalculating risk with adapted weights
            risk_score, risk_category, transaction_details, recommended_action = real_time_risk_scoring(
                transaction_data, models, adapted_weights
            )

            # Capture feedback effect
            if abs(risk_score - baseline_score) > 0.01:  
                feedback_effect = {
                    "original_score": baseline_score,
                    "adjusted_score": risk_score,
                    "original_category": baseline_category,
                    "adjusted_category": risk_category
                }
        else:
            risk_score, risk_category, transaction_details, recommended_action = baseline_score, baseline_category, baseline_details, baseline_action

        transaction_details["real_time_signals"] = layer3_signals
        
        ## 1. Rule-based explanation
        rule_based_explanation = generate_fraud_explanation(
            risk_score=risk_score,
            risk_category=risk_category,
            transaction_details=transaction_details
        )
        #llm
        llm_explanation = generate_llm_explanation(
            risk_score=risk_score,
            risk_category=risk_category,
            transaction_details=transaction_details,
            recommended_action=recommended_action
        )
        
        ### 3. Combined/Final explanation (prefer LLM, fallback to rule-based)
        final_explanation = llm_explanation if llm_explanation else rule_based_explanation

        stored_scores[transaction_id] = {
            'timestamp': timestamp,
            'risk_score': risk_score,
            'risk_category': risk_category,
            'transaction_details': transaction_details,
            'recommended_action': recommended_action,
            'explanations': {
                'rule_based': rule_based_explanation,
                'llm': llm_explanation,
                'final': final_explanation
            },
            'llm_status': 'connected' if client is not None else 'disconnected',
            'feedback_used': existing_feedback['transaction_id'] if existing_feedback else None,
            'feedback_effect': feedback_effect
        }
        save_to_pickle(stored_scores, REAL_TIME_RISK_SCORES_PKL)

        return jsonify({
            'status': 'success',
            'message': 'Risk score was calculated successfully !!!!!!!!!!!!!!!!!!!',
            'result': {
                'transaction_id': transaction_id,
                'timestamp': timestamp,
                'risk_score': risk_score,
                'risk_category': risk_category,
                'transaction_details': transaction_details,
                'recommended_action': recommended_action,
                'explanations': {
                    'rule_based': rule_based_explanation,
                    'llm': llm_explanation if llm_explanation else 'LLM not available - check OpenAI API key',
                    'final': final_explanation
                },
                'llm_status': 'connected' if client is not None else 'disconnected',
                'feedback_used': existing_feedback['transaction_id'] if existing_feedback else None,
                'feedback_effect': feedback_effect
            }
        })

    except Exception as e:
        logger.error(f"Error in real-time risk scoring: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500
        
    
@app.route('/v2/api/transactions', methods=['POST'])
def transactions_endpoint():
    """
    Unified endpoint for transactions.
    """
    try:
    
        data = request.get_json() or {}
        transaction_id = data.get('transaction_id')
        
        if transaction_id:
            # ========== SINGLE TRANSACTION ==========
            transactions = load_from_pickle(REAL_TIME_RISK_SCORES_PKL)

            if not transactions:
                return jsonify({
                    'status': 'error',
                    'message': 'No transactions data available !!!!!'
                }), 500

            ### Checking if the transaction exists
            transaction = transactions.get(transaction_id)

            if not transaction:
                return jsonify({
                    'status': 'error',
                    'message': f'Transaction with ID {transaction_id} not found !!!!!!!!!!!!!'
                }), 404

         
            def make_json_serializable(data):
                """Recursively convert data to JSON-serializable types."""
                if isinstance(data, dict):
                    return {key: make_json_serializable(value) for key, value in data.items()}
                elif isinstance(data, list):
                    return [make_json_serializable(item) for item in data]
                elif isinstance(data, (np.integer, np.floating)):
                    return float(data) if isinstance(data, np.floating) else int(data)
                elif isinstance(data, (int, float, str)):
                    return data
                elif isinstance(data, bool):
                    return bool(data)  
                elif data is None:
                    return None
                else:
                    return str(data)  
            
            cleaned_transaction = make_json_serializable(transaction)
            
            risk_category = cleaned_transaction.get('risk_category', 'Unknown')
            risk_score = float(cleaned_transaction.get('risk_score', 0))
            timestamp = cleaned_transaction.get('timestamp', '')
            transaction_details = cleaned_transaction.get('transaction_details', {})
            recommended_action = cleaned_transaction.get('recommended_action', '')
            
            if risk_category in ['Critical Fraud Risk', 'High Potential Fraud']:
                risk_level = 'HIGH_RISK'
                status_message = f'Transaction ID {transaction_id} is flagged as {risk_category} !!!!!!!!!'
            elif risk_category == 'Medium Risk':
                risk_level = 'MEDIUM_RISK'
                status_message = f'Transaction ID {transaction_id} has medium risk !!!!!!!!!!!!!'
            else:
                risk_level = 'LOW_RISK'
                status_message = f'Transaction ID {transaction_id} has low risk !!!!!!!!!!!!'

            response_data = {
                'status': 'success',
                'message': status_message,
                'transaction_id': transaction_id,
                'timestamp': str(timestamp),
                'risk_assessment': {
                    'risk_score': float(risk_score),
                    'risk_category': str(risk_category),
                    'risk_alert_level': str(risk_level),
                    'threshold': 5.0,
                    'is_high_risk': bool(risk_score >= 5.0)
                },
                'transaction_details': transaction_details,
                'recommended_action': str(recommended_action),
            }

            return jsonify(response_data), 200
            
        else:
    
            page = data.get('page', 1)
            size = data.get('size', 10)
            
            ## Validate pagination parameters
            if not isinstance(page, int) or page < 1:
                return jsonify({
                    'status': 'error',
                    'message': 'Page must be an integer greater than 0.'
                }), 400
            
            if not isinstance(size, int) or size < 1 or size > 100:
                return jsonify({
                    'status': 'error',
                    'message': 'Size must be an integer between 1 and 100.'
                }), 400

            transactions = load_from_pickle(REAL_TIME_RISK_SCORES_PKL)

            if not transactions:
                return jsonify({
                    'status': 'success',
                    'message': 'No transactions found.',
                    'transactions': [],
                    'total': 0
                }), 200

            # Convert to list and sort
            tx_list = []
            for tx_id, tx_data in transactions.items():
                tx_list.append({
                    'transaction_id': tx_id,
                    'timestamp': tx_data.get('timestamp', ''),
                    'risk_score': tx_data.get('risk_score', 0),
                    'risk_category': tx_data.get('risk_category', ''),
                    'transaction_details': tx_data.get('transaction_details', {}),
                    'recommended_action': tx_data.get('recommended_action', '')
                })
            
            tx_list.sort(key=lambda x: x['timestamp'], reverse=True)
            total = len(tx_list)
            start_idx = (page - 1) * size
            end_idx = start_idx + size
            
            paginated = tx_list[start_idx:end_idx]
            
            return jsonify({
                'status': 'success',
                'message': f'Loaded {len(paginated)} of {total} transactions !!!!!!!!!!!!',
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
       
@app.route('/v2/api/fraud_history', methods=['POST'])
def get_fraud_history():
    """
    Endpoint to get all transactions flagged as High Potential Fraud OR Critical Fraud Risk.
    """
    try:
        data = request.get_json()
        
        page = data.get('page', 1) if data else 1
        size = data.get('size', 10) if data else 10
        
        # Validate pagination parameters
        if not isinstance(page, int) or page < 1:
            return jsonify({
                'status': 'error',
                'message': 'Page must be an integer greater than 0.'
            }), 400
        
        if not isinstance(size, int) or size < 1 or size > 100:
            return jsonify({
                'status': 'error',
                'message': 'Size must be an integer between 1 and 100.'
            }), 400

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

        # Filter transactions that are flagged as High Potential Fraud OR Critical Fraud Risk
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
            fraud_list.append({
                'transaction_id': tx_id,
                'timestamp': tx_data.get('timestamp', ''),
                'risk_score': tx_data.get('risk_score', 0),
                'risk_category': tx_data.get('risk_category', ''),
                'transaction_details': tx_data.get('transaction_details', {}),
                'recommended_action': tx_data.get('recommended_action', '')
            })

        # Sort by risk score (highest first), then by timestamp (most recent first)
        fraud_list.sort(key=lambda x: (-x['risk_score'], x['timestamp']), reverse=True)

        total = len(fraud_list)
        total_pages = max(1, (total + size - 1) // size) 
   
        if page > total_pages:
            page = total_pages
        
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated_results = fraud_list[start_idx:end_idx]

        return jsonify({
            'status': 'success',
            'message': f'Found {total} fraud transactions !!!!!!!!!!!',
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

@app.route("/v2/api/fraud_feedback", methods=["POST"])
def fraud_feedback():
    """
    Endpoint to handle fraud feedback from users or analysts.
    """
    try:
        data = request.json
        transaction_id = data.get("transaction_id")
        feedback = data.get("feedback")  # "false_positive" or "confirmed_fraud"
        signals = data.get("signals")    

        if not all([transaction_id, feedback]):
            return jsonify({"error": "transaction_id and feedback are required"}), 400

        # Load the stored real-time transactions
        stored_transactions = load_or_initialize_pickle(REAL_TIME_RISK_SCORES_PKL, {})

        if transaction_id not in stored_transactions:
            return jsonify({
                "error": f"Transaction with ID {transaction_id} not found in records."
            }), 404

        if signals is None:
            transaction_details = stored_transactions[transaction_id].get("transaction_details", {})
            signals = transaction_details  

        store_feedback(transaction_id, feedback, signals)
        adapt_weights(signals, feedback)

        return jsonify({"message": f"Feedback for transaction {transaction_id} processed successfully !!!!!!"}), 200

    except Exception as e:
        logger.error(f"Error in fraud feedback endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/v2/api/test', methods=['GET'])
def test():
    return "Testing endpoint, fraud detection apis working effectively !!!!!!!!!!!!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
