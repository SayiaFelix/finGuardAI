import json
import os

from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import bcrypt
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

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
        ### Converting to native Python types for JSON storage
        transaction_details = convert_for_json(transaction_data.get('transaction_details', {}))
        customer_info = convert_for_json(transaction_data.get('customer_info', {}))
        explanations = convert_for_json(transaction_data.get('explanations', {}))
        
        # Parse timestamp
        timestamp_str = transaction_data.get('timestamp')
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                timestamp = get_nairobi_time()
        else:
            timestamp = get_nairobi_time()
        
        #transaction object
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
        
        # Merge or insert (upsert)
        existing = db.query(Transaction).filter(Transaction.id == transaction_data.get('transaction_id')).first()
        if existing:
            # Update existing
            for key, value in db_transaction.__dict__.items():
                if not key.startswith('_') and key != 'id':
                    setattr(existing, key, value)
            db.commit()
            print(f" Updated transaction {transaction_data.get('transaction_id')} in SQLite")
        else:
            # Insert new
            db.add(db_transaction)
            db.commit()
            print(f" Saved transaction {transaction_data.get('transaction_id')} to SQLite")
            
        return True
        
    except Exception as e:
        db.rollback()
        print(f" Database error: {e}")
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

# Initialize database (create tables)
init_database()