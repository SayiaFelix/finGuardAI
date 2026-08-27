# finca_adapter.py
"""
FINCA Uganda Fraud Guard Adapter
Maps between FINCA transaction format and FinGuardAI engine
Accepts BOTH lowercase AND capitalized field names
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
import random
import pandas as pd

logger = logging.getLogger(__name__)

class FINCAAdapter:
    """Adapter to translate between FINCA format and FinGuardAI engine"""
    
    def __init__(self):
        self.engine_available = False
        try:
            from fin_guard_ai import (
                real_time_risk_scoring, 
                load_model_from_JobLib,
                load_from_pickle,
                RISK_MODELS_JOBLIB,
                IMPORTANT_FEATURES_WEIGHTS_PKL,
                get_active_threshold,
                IMPORTANT_FEATURES_PKL
            )
            self.real_time_risk_scoring = real_time_risk_scoring
            self.load_model = load_model_from_JobLib
            self.load_pickle = load_from_pickle
            self.RISK_MODELS_JOBLIB = RISK_MODELS_JOBLIB
            self.IMPORTANT_FEATURES_WEIGHTS_PKL = IMPORTANT_FEATURES_WEIGHTS_PKL
            self.IMPORTANT_FEATURES_PKL = IMPORTANT_FEATURES_PKL
            self.get_threshold = get_active_threshold
            self.engine_available = True
            logger.info("FINCAAdapter: Engine loaded successfully")
        except ImportError as e:
            logger.error(f"FINCAAdapter: Could not import engine: {e}")
    
    def map_to_engine_format(self, finca_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map FINCA transaction format to engine format
        Accepts BOTH lowercase AND capitalized field names
        """
        engine_data = {}
        
        # ============================================
        # 1. Amount - Accept both formats
        # ============================================
        amount = finca_data.get('transaction_amount', finca_data.get('Transaction_Amount', 0))
        engine_data['Transaction_Amount'] = amount
        
        # ============================================
        # 2. Customer ID - Accept both formats
        # ============================================
        engine_data['customer_id'] = finca_data.get('customer_id', finca_data.get('Customer_ID', 'UNKNOWN'))
        
        # ============================================
        # 3. Transaction Hour - Accept both formats
        # ============================================
        if 'Transaction_Hour' in finca_data:
            engine_data['Transaction_Hour'] = finca_data['Transaction_Hour']
        elif 'transaction_hour' in finca_data:
            engine_data['Transaction_Hour'] = finca_data['transaction_hour']
        else:
            engine_data['Transaction_Hour'] = float(datetime.now().hour)
        
        # ============================================
        # 4. Transaction Frequency (velocity) - Accept both
        # ============================================
        if 'tx_count_last_hour' in finca_data:
            engine_data['Transaction_Frequency'] = finca_data['tx_count_last_hour']
        elif 'Transaction_Frequency' in finca_data:
            engine_data['Transaction_Frequency'] = finca_data['Transaction_Frequency']
        else:
            engine_data['Transaction_Frequency'] = 1
        
        # ============================================
        # 5. Day of Week - Accept both
        # ============================================
        if 'Day_of_Week' in finca_data:
            engine_data['Day_of_Week'] = finca_data['Day_of_Week']
        elif 'day_of_week' in finca_data:
            engine_data['Day_of_Week'] = finca_data['day_of_week']
        else:
            engine_data['Day_of_Week'] = datetime.now().weekday()
        
        # ============================================
        # 6. IP Address - Accept both
        # ============================================
        ip = finca_data.get('ip_address', finca_data.get('IP_Address', None))
        
        if ip:
            try:
                import ipaddress
                engine_data['IP_Address'] = int(ipaddress.ip_address(ip))
            except:
                engine_data['IP_Address'] = random.randint(100000000, 999999999)
        else:
            engine_data['IP_Address'] = random.randint(100000000, 999999999)
        
        # ============================================
        # 7. Is Weekend - Accept both
        # ============================================
        if 'Is_Weekend' in finca_data:
            engine_data['Is_Weekend'] = finca_data['Is_Weekend']
        elif 'is_weekend' in finca_data:
            engine_data['Is_Weekend'] = finca_data['is_weekend']
        else:
            engine_data['Is_Weekend'] = 1 if datetime.now().weekday() >= 5 else 0
        
        # ============================================
        # 8. Account Activity (CRITICAL for risk score!)
        # ============================================
        if 'avg_transaction_amount' in finca_data and finca_data['avg_transaction_amount'] > 0:
            engine_data['Account_Activity'] = finca_data['avg_transaction_amount']
        elif 'Account_Activity' in finca_data:
            engine_data['Account_Activity'] = finca_data['Account_Activity']
        else:
            engine_data['Account_Activity'] = 25000
        
        # ============================================
        # 9. Amount Categories (one-hot encoded)
        # ============================================
        amount = engine_data['Transaction_Amount']
        
        engine_data['Amount_Category_Very High'] = 0
        engine_data['Amount_Category_High'] = 0
        engine_data['Amount_Category_Medium'] = 0
        engine_data['Amount_Category_Low'] = 0
        
        if amount >= 10000000:  # 10M+
            engine_data['Amount_Category_Very High'] = 1
        elif amount >= 5000000:  # 5M-10M
            engine_data['Amount_Category_High'] = 1
        elif amount >= 1000000:  # 1M-5M
            engine_data['Amount_Category_Medium'] = 1
        else:
            engine_data['Amount_Category_Low'] = 1
        
        # ============================================
        # 10. Location Categories - Accept both
        # ============================================
        engine_data['Transaction_Location_International'] = 0
        engine_data['Transaction_Location_Local'] = 0
        
        location = finca_data.get('location', finca_data.get('Transaction_Location', 'Local'))
        if location.lower() in ['international', 'foreign', 'out of country']:
            engine_data['Transaction_Location_International'] = 1
        else:
            engine_data['Transaction_Location_Local'] = 1
        
        # ============================================
        # 11. Device Type - Accept both formats
        # ============================================
        device = finca_data.get('device_type', finca_data.get('Device_Type', 'Unknown'))
        
        engine_data['Device_Type_Unknown_Device'] = 0
        engine_data['Device_Type_iPhone'] = 0
        engine_data['Device_Type_MacBook'] = 0
        engine_data['Device_Type_Samsung'] = 0
        engine_data['Device_Type_Huawei'] = 0
        engine_data['Device_Type_Tecno'] = 0
        
        device_lower = device.lower()
        if device_lower in ['unknown', 'new', 'unrecognized']:
            engine_data['Device_Type_Unknown_Device'] = 1
        elif 'iphone' in device_lower or 'ios' in device_lower:
            engine_data['Device_Type_iPhone'] = 1
        elif 'mac' in device_lower or 'apple' in device_lower:
            engine_data['Device_Type_MacBook'] = 1
        elif 'samsung' in device_lower or 'android' in device_lower:
            engine_data['Device_Type_Samsung'] = 1
        elif 'huawei' in device_lower:
            engine_data['Device_Type_Huawei'] = 1
        elif 'tecno' in device_lower:
            engine_data['Device_Type_Tecno'] = 1
        else:
            engine_data['Device_Type_Unknown_Device'] = 1
        
        # ============================================
        # 12. Channel / Transaction Type - Accept both
        # ============================================
        engine_data['Transaction_Type_Online'] = 0
        engine_data['Transaction_Type_POS'] = 0
        
        channel = finca_data.get('channel', finca_data.get('Transaction_Type', ''))
        if channel.upper() in ['MOBILE', 'MOBILE_BANKING', 'INTERNET', 'INTERNET_BANKING', 'ONLINE']:
            engine_data['Transaction_Type_Online'] = 1
        elif channel.upper() in ['ATM', 'POS', 'CARD']:
            engine_data['Transaction_Type_POS'] = 1
        else:
            engine_data['Transaction_Type_Online'] = 1
        
        # ============================================
        # 13. Transaction Period - Accept both
        # ============================================
        hour = engine_data['Transaction_Hour']
        
        engine_data['Transaction_Period_Evening'] = 0
        engine_data['Transaction_Period_Afternoon'] = 0
        engine_data['Transaction_Period_Morning'] = 0
        engine_data['Transaction_Period_Night'] = 0
        
        if 18 <= hour < 24:
            engine_data['Transaction_Period_Evening'] = 1
        elif 12 <= hour < 18:
            engine_data['Transaction_Period_Afternoon'] = 1
        elif 6 <= hour < 12:
            engine_data['Transaction_Period_Morning'] = 1
        else:
            engine_data['Transaction_Period_Night'] = 1
        
        # ============================================
        # 14. Additional fields from customer profile
        # ============================================
        if 'avg_transaction_amount' in finca_data:
            engine_data['avg_transaction_amount'] = finca_data['avg_transaction_amount']
        
        if 'account_age_days' in finca_data:
            engine_data['account_age_days'] = finca_data['account_age_days']
        
        # ============================================
        # 15. Beneficiary - Accept both
        # ============================================
        beneficiary = finca_data.get('beneficiary', finca_data.get('Beneficiary', ''))
        if 'NEW_BEN' in beneficiary or 'new' in beneficiary.lower():
            engine_data['Beneficiary'] = 'NEW_BENEFICIARY'
        else:
            engine_data['Beneficiary'] = beneficiary
        
        # ============================================
        # 16. Customer name/email/phone - Accept both
        # ============================================
        engine_data['customer_name'] = finca_data.get('customer_name', finca_data.get('Customer_Name', ''))
        engine_data['customer_email'] = finca_data.get('customer_email', finca_data.get('Customer_Email', ''))
        engine_data['customer_phone'] = finca_data.get('customer_phone', finca_data.get('Customer_Phone', ''))
        
        logger.debug(f"Mapped FINCA -> Engine: {list(engine_data.keys())}")
        
        return engine_data
    
    def map_to_finca_format(self, engine_result: tuple) -> Dict[str, Any]:
        """
        Map engine result to FINCA format
        
        engine_result: (risk_score, risk_category, transaction_details, recommended_action)
        """
        risk_score, risk_category, transaction_details, recommended_action = engine_result
        
        # Convert engine risk score (0-10) to FINCA (0-100)
        finca_risk_score = risk_score * 10
        
        if 'Risk_Score' in transaction_details:
            transaction_details['Risk_Score'] = finca_risk_score
        
        risk_category_lower = risk_category.lower()
        
        if "critical" in risk_category_lower:
            finca_risk_level = "CRITICAL"
        elif "high" in risk_category_lower:
            finca_risk_level = "HIGH"
        elif "medium" in risk_category_lower:
            finca_risk_level = "MEDIUM"
        else:
            finca_risk_level = "LOW"
        
        # Decision based on FINCA risk level
        if finca_risk_level == "CRITICAL":
            decision = "BLOCK"
        elif finca_risk_level in ["HIGH", "MEDIUM"]:
            decision = "CHALLENGE"
        else:
            decision = "APPROVE"
        
        # Map reasons to FINCA format
        reasons = []
        rule_flags = transaction_details.get('Rule_Flags', [])
        for flag in rule_flags:
            reasons.append(flag)
        
        # Map to FINCA rule names
        finca_rules = []
        rule_map = {
            'Unknown device': 'NEW_DEVICE',
            'International location': 'LOCATION_ANOMALY',
            'High transaction frequency': 'VELOCITY_ANOMALY',
            'Amount exceeds threshold': 'AMOUNT_ANOMALY',
            'Weekend evening transaction': 'UNUSUAL_TIME',
            'Unusual transaction hour': 'UNUSUAL_TIME'
        }
        for flag in rule_flags:
            if flag in rule_map:
                finca_rules.append(rule_map[flag])
        
        return {
            'risk_score': finca_risk_score,
            'risk_level': finca_risk_level,
            'decision': decision,
            'triggered_rules': finca_rules,
            'reasons': reasons,
            'ml_score': risk_score / 10,
            'recommended_action': recommended_action,
            'transaction_details': transaction_details
        }
    
    def analyze(self, finca_transaction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Main analysis method - maps, calls engine, maps back
        """
        if not self.engine_available:
            logger.error("Engine not available")
            return None
        
        try:
            # 1. Map to engine format (returns DICT)
            engine_data = self.map_to_engine_format(finca_transaction)
            
            # 2. Load models
            models = self.load_model(self.RISK_MODELS_JOBLIB)
            weights = self.load_pickle(self.IMPORTANT_FEATURES_WEIGHTS_PKL)
            weights_map = weights['Combined_Weight']
            
            # 3. Convert to pandas Series (engine expects this)
      
            # Load the selected features list
            selected_features = self.load_pickle(self.IMPORTANT_FEATURES_PKL)
            
            # Create a Series with all required features
            transaction_series = pd.Series(engine_data)
            
            # Ensure all selected features exist (fill missing with 0)
            for feature in selected_features:
                if feature not in transaction_series.index:
                    transaction_series[feature] = 0
            
            logger.info(f"Transaction Series ready with {len(transaction_series)} features")
            
            # 4. Call engine with Series
            risk_score, risk_category, details, action = self.real_time_risk_scoring(
                transaction_series, models, weights_map
            )
            
            # 5. Map back to FINCA format
            result = self.map_to_finca_format(
                (risk_score, risk_category, details, action)
            )
            
            # 6. Add timestamp and status
            result['timestamp'] = datetime.now().isoformat()
            result['status'] = 'PROCESSED'
            
            return result
            
        except Exception as e:
            logger.error(f"FINCAAdapter analysis error: {e}")
            import traceback
            traceback.print_exc()
            return None


# Singleton instance
_adapter_instance = None

def get_adapter():
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = FINCAAdapter()
    return _adapter_instance