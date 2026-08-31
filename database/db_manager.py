import json
import os
from datetime import datetime

try:
    from sqlalchemy import (
        create_engine, Column, String, Float, DateTime, Integer, JSON, Boolean,
        ForeignKey, func
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import relationship, sessionmaker
except ImportError:  # pragma: no cover - handled by environment setup
    from sqlalchemy import (  # type: ignore
        create_engine, Column, String, Float, DateTime, Integer, JSON, Boolean,
        ForeignKey, func
    )
    from sqlalchemy.ext.declarative import declarative_base  # type: ignore
    from sqlalchemy.orm import relationship, sessionmaker  # type: ignore

import pytz
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import bcrypt

load_dotenv()

# SQLite database file 
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///database/fraudsentinel.db')

#engine
engine = create_engine(
    DATABASE_URL,
    echo=False, 
    connect_args={"check_same_thread": False} 
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

def get_nairobi_time():
    """Returns current time in Africa/Nairobi timezone"""
    nairobi_tz = pytz.timezone('Africa/Nairobi')
    return datetime.now(nairobi_tz)

class User(Base):
    """User accounts for authentication"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default='analyst')
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=get_nairobi_time)
    last_login = Column(DateTime, nullable=True)
    
    #Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    
    def set_password(self, password):
        """Hash and set password - works with detached session"""
        salt = bcrypt.gensalt()
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
        return self

    def check_password(self, password):
        """Verify password - works with detached session"""
        # This doesn't need database access, just compares strings
        if not self.password_hash:
            return False
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

class APIKey(Base):
    """API keys for programmatic access (commercial clients)"""
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True)
    key = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    tier = Column(String, default='free')  # 'free', 'basic', 'enterprise'
    rate_limit = Column(Integer, default=100)  # requests per minute
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=get_nairobi_time)
    expires_at = Column(DateTime, nullable=True)
    
    #Relationship
    user = relationship("User", back_populates="api_keys")

class RefreshToken(Base):
    """Store refresh tokens for rotation"""
    __tablename__ = 'refresh_tokens'
    
    id = Column(Integer, primary_key=True)
    token = Column(String, unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=get_nairobi_time)
    revoked = Column(Boolean, default=False)

    # Relationship
    user = relationship("User", back_populates="refresh_tokens")

def create_admin_user():
    """Create default admin user if none exists"""
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.role == 'admin').first()
        if not admin:
            admin_user = User(
                email=os.getenv('ADMIN_EMAIL', 'admin@fraudsentinel.com'),
                username=os.getenv('ADMIN_USERNAME', 'fraudsentinelAdmin'),
                role='admin',
                is_active=True
            )
            admin_user.set_password(os.getenv('ADMIN_PASSWORD', 'admin@123'))
            db.add(admin_user)
            db.commit()
            print(f" Default admin user created: {admin_user.username} (role: admin)")
        else:
            print(f" Admin user already exists: {admin.username}")
    except Exception as e:
        print(f" Admin user creation: {e}")
    finally:
        db.close()

class Transaction(Base):
    """Matches exactly what you save in REAL_TIME_RISK_SCORES_PKL"""
    __tablename__ = 'transactions'
    
    #Primary key
    id = Column(String, primary_key=True) 
    
    # Core fields
    timestamp = Column(DateTime, default=get_nairobi_time)
    risk_score = Column(Float)
    risk_category = Column(String)
    recommended_action = Column(String)
    
    # JSON fields 
    transaction_details = Column(JSON)
    customer_info = Column(JSON)
    explanations = Column(JSON)
    
    # Status tracking
    status = Column(String, default='Open')
    status_history = Column(JSON, default=[])
    
    # Feedback fields
    feedback_used = Column(String, nullable=True)
    feedback_effect = Column(JSON, nullable=True)
    
    # System metadata
    model_version = Column(String)
    threshold_used = Column(Float)
    national_alert_mode = Column(Boolean, default=False)
    llm_status = Column(String, default='disconnected')

class Rule(Base):
    """Fraud Detection Rules - matches your rules_storage structure"""
    __tablename__ = 'rules'
    
    id = Column(String, primary_key=True)  # e.g., 'R001'
    name = Column(String, nullable=False)
    description = Column(String, default='')
    
    # JSON fields for complex structures
    conditions = Column(JSON, nullable=False)  # {'field': 'device_type', 'operator': 'is_new', ...}
    action = Column(JSON, nullable=False)     # {'risk_points': 25, 'decision': 'BLOCK', ...}
    
    priority = Column(Integer, default=999)
    is_active = Column(Boolean, default=True)
    category = Column(String, default='CUSTOM')
    
    # Metadata
    created_at = Column(DateTime, default=get_nairobi_time)
    updated_at = Column(DateTime, default=get_nairobi_time, onupdate=get_nairobi_time)
    created_by = Column(String, default='System')
    version = Column(Integer, default=1)
    
    # Statistics
    trigger_count = Column(Integer, default=0)
    false_positive_rate = Column(Float, default=0.0)
    last_triggered = Column(DateTime, nullable=True)
    
    # Soft delete
    deleted_at = Column(DateTime, nullable=True)

class RuleHistory(Base):
    """Rule version history for audit trail"""
    __tablename__ = 'rule_history'
    
    id = Column(Integer, primary_key=True)
    rule_id = Column(String, ForeignKey('rules.id'), nullable=False)
    version = Column(Integer, nullable=False)
    snapshot = Column(JSON, nullable=False)  # Full rule snapshot
    saved_at = Column(DateTime, default=get_nairobi_time)
    
    # Relationship
    rule = relationship("Rule", backref="history")

class RuleFeedback(Base):
    """Track rule performance feedback"""
    __tablename__ = 'rule_feedback'
    
    id = Column(Integer, primary_key=True)
    rule_id = Column(String, ForeignKey('rules.id'))
    transaction_id = Column(String, nullable=True)
    feedback_type = Column(String)  # 'correct', 'false_positive', 'missed'
    notes = Column(String, default='')
    created_at = Column(DateTime, default=get_nairobi_time)
    created_by = Column(String, default='System')
    
class Feedback(Base):
    """Matches your feedback.json structure"""
    __tablename__ = 'feedbacks'
    
    id = Column(Integer, primary_key=True)
    transaction_id = Column(String)
    feedback_type = Column(String)  # 'confirmed_fraud' or 'false_positive'
    signals = Column(JSON)
    timestamp = Column(DateTime, default=get_nairobi_time)
    analyst_id = Column(String, default='system')

def init_database():
    """Create all tables"""
    Base.metadata.create_all(engine)
    print(" SQLite database created successfully! (fraudsentinel.db)")
    create_admin_user()

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_transaction_to_db(transaction_data):
    db = SessionLocal()
    try:
        transaction_details = convert_for_json(transaction_data.get('transaction_details', {}))
        customer_info = convert_for_json(transaction_data.get('customer_info', {}))
        explanations = convert_for_json(transaction_data.get('explanations', {}))
        
        timestamp_str = transaction_data.get('timestamp')
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                timestamp = get_nairobi_time()
        else:
            timestamp = get_nairobi_time()
        
        db_transaction = Transaction(
            id=transaction_data.get('transaction_id'),
            timestamp=timestamp,
            risk_score=float(transaction_data.get('risk_score', 0)),
            risk_category=transaction_data.get('risk_category', 'Unknown'),
            recommended_action=transaction_data.get('recommended_action', ''),
            transaction_details=transaction_details,
            customer_info=customer_info,
            explanations=explanations,
            status=transaction_data.get('status', {}).get('current', 'Open'),
            status_history=transaction_data.get('status', {}).get('history', []),
            feedback_used=transaction_data.get('feedback_used'),
            feedback_effect=transaction_data.get('feedback_effect'),
            model_version=transaction_data.get('model_version', 'v1.0.0'),
            threshold_used=float(transaction_data.get('threshold_used', 6.0)),
            national_alert_mode=transaction_data.get('national_alert_mode', False),
            llm_status=transaction_data.get('llm_status', 'disconnected')
        )
        
        # Upsert
        existing = db.query(Transaction).filter(Transaction.id == transaction_data.get('transaction_id')).first()
        if existing:
            for key, value in db_transaction.__dict__.items():
                if not key.startswith('_') and key != 'id':
                    setattr(existing, key, value)
            db.commit()
            print(f"Updated transaction {transaction_data.get('transaction_id')} in SQLite")
        else:
            db.add(db_transaction)
            db.commit()
            print(f"Saved transaction {transaction_data.get('transaction_id')} to SQLite")
            
        return True
        
    except Exception as e:
        db.rollback()
        print(f"Database error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

def save_feedback_to_db(transaction_id, feedback_type, signals):
    """Save feedback to database"""
    db = SessionLocal()
    try:
        fb = Feedback(
            transaction_id=transaction_id,
            feedback_type=feedback_type,
            signals=convert_for_json(signals)
        )
        db.add(fb)
        db.commit()
        print(f" Saved feedback for {transaction_id} to SQLite")
        return True
    except Exception as e:
        db.rollback()
        print(f" Database error saving feedback: {e}")
        return False
    finally:
        db.close()

# ============================================
# RULES CRUD OPERATIONS (Same pattern as transactions)
# ============================================

def save_rule_to_db(rule_data):
    """Save a rule to SQLite (same pattern as save_transaction_to_db)"""
    db = SessionLocal()
    try:
        # Convert to JSON-safe format
        conditions = convert_for_json(rule_data.get('conditions', {}))
        action = convert_for_json(rule_data.get('action', {}))
        
        # Parse timestamp
        created_at = None
        if rule_data.get('created_at'):
            try:
                created_at = datetime.fromisoformat(rule_data['created_at'].replace('Z', '+00:00'))
            except:
                created_at = get_nairobi_time()
        else:
            created_at = get_nairobi_time()
        
        updated_at = None
        if rule_data.get('updated_at'):
            try:
                updated_at = datetime.fromisoformat(rule_data['updated_at'].replace('Z', '+00:00'))
            except:
                updated_at = get_nairobi_time()
        
        last_triggered = None
        if rule_data.get('last_triggered'):
            try:
                last_triggered = datetime.fromisoformat(rule_data['last_triggered'].replace('Z', '+00:00'))
            except:
                pass
        
        deleted_at = None
        if rule_data.get('deleted_at'):
            try:
                deleted_at = datetime.fromisoformat(rule_data['deleted_at'].replace('Z', '+00:00'))
            except:
                pass
        
        db_rule = Rule(
            id=rule_data.get('id'),
            name=rule_data.get('name', 'Unnamed Rule'),
            description=rule_data.get('description', ''),
            conditions=conditions,
            action=action,
            priority=rule_data.get('priority', 999),
            is_active=rule_data.get('is_active', True),
            category=rule_data.get('category', 'CUSTOM'),
            created_at=created_at,
            updated_at=updated_at or get_nairobi_time(),
            created_by=rule_data.get('created_by', 'System'),
            version=rule_data.get('version', 1),
            trigger_count=rule_data.get('trigger_count', 0),
            false_positive_rate=rule_data.get('false_positive_rate', 0.0),
            last_triggered=last_triggered,
            deleted_at=deleted_at
        )
        
        # Upsert
        existing = db.query(Rule).filter(Rule.id == rule_data.get('id')).first()
        if existing:
            for key, value in db_rule.__dict__.items():
                if not key.startswith('_') and key != 'id':
                    setattr(existing, key, value)
            db.commit()
            print(f"Updated rule {rule_data.get('id')} in SQLite")
        else:
            db.add(db_rule)
            db.commit()
            print(f"Saved rule {rule_data.get('id')} to SQLite")
        
        return True
        
    except Exception as e:
        db.rollback()
        print(f"Database error saving rule: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

def get_all_rules_from_db(include_inactive=False):
    """Get all rules from SQLite"""
    db = SessionLocal()
    try:
        query = db.query(Rule)
        
        if not include_inactive:
            query = query.filter(Rule.is_active == True, Rule.deleted_at.is_(None))
        else:
            query = query.filter(Rule.deleted_at.is_(None))
        
        rules = query.order_by(Rule.priority.asc()).all()
        
        result = []
        for rule in rules:
            result.append({
                'id': rule.id,
                'name': rule.name,
                'description': rule.description,
                'conditions': rule.conditions,
                'action': rule.action,
                'priority': rule.priority,
                'is_active': rule.is_active,
                'category': rule.category,
                'created_at': rule.created_at.isoformat() if rule.created_at else None,
                'updated_at': rule.updated_at.isoformat() if rule.updated_at else None,
                'created_by': rule.created_by,
                'version': rule.version,
                'trigger_count': rule.trigger_count,
                'false_positive_rate': rule.false_positive_rate,
                'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None,
                'deleted_at': rule.deleted_at.isoformat() if rule.deleted_at else None
            })
        
        return result
        
    except Exception as e:
        print(f"Database error getting rules: {e}")
        return []
    finally:
        db.close()

def get_rule_from_db(rule_id):
    """Get a single rule from SQLite"""
    db = SessionLocal()
    try:
        rule = db.query(Rule).filter(Rule.id == rule_id, Rule.deleted_at.is_(None)).first()
        
        if not rule:
            return None
        
        return {
            'id': rule.id,
            'name': rule.name,
            'description': rule.description,
            'conditions': rule.conditions,
            'action': rule.action,
            'priority': rule.priority,
            'is_active': rule.is_active,
            'category': rule.category,
            'created_at': rule.created_at.isoformat() if rule.created_at else None,
            'updated_at': rule.updated_at.isoformat() if rule.updated_at else None,
            'created_by': rule.created_by,
            'version': rule.version,
            'trigger_count': rule.trigger_count,
            'false_positive_rate': rule.false_positive_rate,
            'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None,
            'deleted_at': rule.deleted_at.isoformat() if rule.deleted_at else None
        }
        
    except Exception as e:
        print(f"Database error getting rule {rule_id}: {e}")
        return None
    finally:
        db.close()

def delete_rule_from_db(rule_id):
    """Soft delete a rule from SQLite"""
    db = SessionLocal()
    try:
        rule = db.query(Rule).filter(Rule.id == rule_id).first()
        if rule:
            rule.is_active = False
            rule.deleted_at = get_nairobi_time()
            db.commit()
            print(f"Rule {rule_id} soft deleted from SQLite")
            return True
        return False
    except Exception as e:
        db.rollback()
        print(f"Database error deleting rule: {e}")
        return False
    finally:
        db.close()

def toggle_rule_in_db(rule_id):
    """Toggle rule active status in SQLite"""
    db = SessionLocal()
    try:
        rule = db.query(Rule).filter(Rule.id == rule_id, Rule.deleted_at.is_(None)).first()
        if rule:
            rule.is_active = not rule.is_active
            rule.updated_at = get_nairobi_time()
            db.commit()
            return rule.is_active
        return None
    except Exception as e:
        db.rollback()
        print(f"Database error toggling rule: {e}")
        return None
    finally:
        db.close()

def get_rules_stats_from_db():
    """Get rule statistics from SQLite"""
    db = SessionLocal()
    try:
        total = db.query(Rule).filter(Rule.deleted_at.is_(None)).count()
        active = db.query(Rule).filter(Rule.is_active == True, Rule.deleted_at.is_(None)).count()
        inactive = total - active
        
        # By category
        from sqlalchemy import func
        categories = db.query(Rule.category, func.count(Rule.id)).filter(Rule.deleted_at.is_(None)).group_by(Rule.category).all()
        categories_dict = {cat: count for cat, count in categories}
        
        # Top triggered rules
        top_rules = db.query(Rule).filter(Rule.deleted_at.is_(None)).order_by(Rule.trigger_count.desc()).limit(5).all()
        top_rules_list = [
            {'id': r.id, 'name': r.name, 'trigger_count': r.trigger_count}
            for r in top_rules
        ]
        
        return {
            'total_rules': total,
            'active_rules': active,
            'inactive_rules': inactive,
            'by_category': categories_dict,
            'top_rules': top_rules_list
        }
        
    except Exception as e:
        print(f"Database error getting rule stats: {e}")
        return {}
    finally:
        db.close()

def save_rule_history_to_db(rule_id, version, snapshot):
    """Save rule version history"""
    db = SessionLocal()
    try:
        history = RuleHistory(
            rule_id=rule_id,
            version=version,
            snapshot=convert_for_json(snapshot)
        )
        db.add(history)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Database error saving rule history: {e}")
        return False
    finally:
        db.close()

def get_rule_history_from_db(rule_id):
    """Get rule history from SQLite"""
    db = SessionLocal()
    try:
        history = db.query(RuleHistory).filter(RuleHistory.rule_id == rule_id).order_by(RuleHistory.version.asc()).all()
        return [
            {
                'version': h.version,
                'snapshot': h.snapshot,
                'saved_at': h.saved_at.isoformat() if h.saved_at else None
            }
            for h in history
        ]
    except Exception as e:
        print(f"Database error getting rule history: {e}")
        return []
    finally:
        db.close()

def get_default_rules():
    """Get default FINCA rules for initialization"""
    return {
        'R001': {
            'id': 'R001',
            'name': 'New Device + High Value',
            'description': 'Transaction from new device with high amount',
            'conditions': {
                'field': 'device_type',
                'operator': 'is_new',
                'value': True,
                'amount_threshold': 5000000
            },
            'action': {
                'risk_points': 25,
                'decision': 'BLOCK',
                'alert': True
            },
            'priority': 1,
            'is_active': True,
            'category': 'DEVICE',
            'created_at': get_nairobi_time().isoformat(),
            'updated_at': get_nairobi_time().isoformat(),
            'created_by': 'System',
            'version': 1,
            'trigger_count': 0,
            'false_positive_rate': 0.0,
            'last_triggered': None
        },
        'R002': {
            'id': 'R002',
            'name': 'New Beneficiary + High Value',
            'description': 'Transaction to newly added beneficiary',
            'conditions': {
                'field': 'beneficiary',
                'operator': 'is_new',
                'value': True,
                'amount_threshold': 2000000
            },
            'action': {
                'risk_points': 15,
                'decision': 'CHALLENGE',
                'alert': True
            },
            'priority': 2,
            'is_active': True,
            'category': 'BENEFICIARY',
            'created_at': get_nairobi_time().isoformat(),
            'updated_at': get_nairobi_time().isoformat(),
            'created_by': 'System',
            'version': 1,
            'trigger_count': 0,
            'false_positive_rate': 0.0,
            'last_triggered': None
        },
        'R003': {
            'id': 'R003',
            'name': 'Transaction Velocity',
            'description': 'Multiple transactions in short period',
            'conditions': {
                'field': 'tx_count_last_hour',
                'operator': 'greater_than',
                'value': 5,
                'timeframe_minutes': 10
            },
            'action': {
                'risk_points': 20,
                'decision': 'CHALLENGE',
                'alert': True
            },
            'priority': 3,
            'is_active': True,
            'category': 'VELOCITY',
            'created_at': get_nairobi_time().isoformat(),
            'updated_at': get_nairobi_time().isoformat(),
            'created_by': 'System',
            'version': 1,
            'trigger_count': 0,
            'false_positive_rate': 0.0,
            'last_triggered': None
        },
        'R004': {
            'id': 'R004',
            'name': 'Amount Anomaly',
            'description': 'Transaction > 5x customer average',
            'conditions': {
                'field': 'transaction_amount',
                'operator': 'greater_than_multiplier',
                'value': 5,
                'based_on': 'customer_avg'
            },
            'action': {
                'risk_points': 20,
                'decision': 'BLOCK',
                'alert': True
            },
            'priority': 4,
            'is_active': True,
            'category': 'AMOUNT',
            'created_at': get_nairobi_time().isoformat(),
            'updated_at': get_nairobi_time().isoformat(),
            'created_by': 'System',
            'version': 1,
            'trigger_count': 0,
            'false_positive_rate': 0.0,
            'last_triggered': None
        },
        'R005': {
            'id': 'R005',
            'name': 'Location Anomaly',
            'description': 'Transaction from unusual location',
            'conditions': {
                'field': 'location',
                'operator': 'is_new_location',
                'value': True
            },
            'action': {
                'risk_points': 15,
                'decision': 'CHALLENGE',
                'alert': True
            },
            'priority': 5,
            'is_active': True,
            'category': 'LOCATION',
            'created_at': get_nairobi_time().isoformat(),
            'updated_at': get_nairobi_time().isoformat(),
            'created_by': 'System',
            'version': 1,
            'trigger_count': 0,
            'false_positive_rate': 0.0,
            'last_triggered': None
        },
        'R006': {
            'id': 'R006',
            'name': 'Unusual Time',
            'description': 'Transaction at unusual hour (midnight - 5am)',
            'conditions': {
                'field': 'transaction_hour',
                'operator': 'between',
                'value': [0, 5]
            },
            'action': {
                'risk_points': 10,
                'decision': 'CHALLENGE',
                'alert': False
            },
            'priority': 6,
            'is_active': True,
            'category': 'TIME',
            'created_at': get_nairobi_time().isoformat(),
            'updated_at': get_nairobi_time().isoformat(),
            'created_by': 'System',
            'version': 1,
            'trigger_count': 0,
            'false_positive_rate': 0.0,
            'last_triggered': None
        }
    }

def init_default_rules():
    """Initialize default rules if no rules exist"""
    rules = get_all_rules_from_db(include_inactive=True)
    if not rules:
        default_rules = get_default_rules()
        for rule_id, rule in default_rules.items():
            save_rule_to_db(rule)
        print(f"✅ Initialized {len(default_rules)} default rules")
    else:
        print(f"✅ Rules already exist: {len(rules)} rules found")
 
def init_database():
    """Create all tables and initialize default data"""
    Base.metadata.create_all(engine)
    print("✅ SQLite database created successfully! (fraudsentinel.db)")
    create_admin_user()
    
    # Initialize default rules
    init_default_rules()
                 
def convert_for_json(obj):
    """Convert any numpy/pandas types to Python native types for JSON storage"""

    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    else:
        return obj

init_database()