import jwt
import os
import secrets
import pytz
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify
from database.db_manager import SessionLocal, User, APIKey, RefreshToken

# JWT Configuration
SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'super-secret-jwt-key-change-this-in-production-12345')
ACCESS_TOKEN_EXPIRES = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES', 300))  # 5 minutes
REFRESH_TOKEN_EXPIRES = int(os.getenv('JWT_REFRESH_TOKEN_EXPIRES', 600))  # 10 minutes

def get_nairobi_time():
    """Returns current time in Africa/Nairobi timezone"""
    nairobi_tz = pytz.timezone('Africa/Nairobi')
    return datetime.now(nairobi_tz)

def generate_access_token(user_id, username, role):
    """Generate JWT access token with Nairobi time"""
    nairobi_time = get_nairobi_time()
    payload = {
        'user_id': user_id,
        'username': username,
        'role': role,
        'exp': nairobi_time + timedelta(seconds=ACCESS_TOKEN_EXPIRES),
        'iat': nairobi_time,
        'type': 'access'
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def generate_refresh_token(user_id):
    """Generate refresh token with Nairobi time"""
    nairobi_time = get_nairobi_time()
    payload = {
        'user_id': user_id,
        'exp': nairobi_time + timedelta(seconds=REFRESH_TOKEN_EXPIRES),
        'iat': nairobi_time,
        'type': 'refresh'
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def generate_api_key():
    """Generate a unique API key"""
    return f"fs_{secrets.token_urlsafe(32)}"

def token_required(f):
    """Decorator to protect endpoints with JWT"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Getting token from Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == 'bearer':
                token = parts[1]
        
        if not token:
            return jsonify({'error': 'Token is missing', 'message': 'Please provide a valid JWT token'}), 401
        
        try:
            # Decode token
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            
            # Check token type
            if payload.get('type') != 'access':
                return jsonify({'error': 'Invalid token type', 'message': 'Use access token'}), 401
            
            # Get user from database
            db = SessionLocal()
            current_user = db.query(User).filter(User.id == payload['user_id']).first()
            db.close()
            
            if not current_user:
                return jsonify({'error': 'User not found'}), 401
            
            if not current_user.is_active:
                return jsonify({'error': 'User account disabled', 'message': 'Contact administrator'}), 401
            
            #user to endpoint
            return f(current_user, *args, **kwargs)
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired', 'message': 'Please refresh your token'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token', 'message': 'Token validation failed'}), 401
        except Exception as e:
            return jsonify({'error': 'Authentication failed', 'message': str(e)}), 401
            
    return decorated

def admin_required(f):
    """Decorator for admin-only endpoints"""
    @wraps(f)
    @token_required
    def decorated(current_user, *args, **kwargs):
        if current_user.role != 'admin':
            return jsonify({'error': 'Admin access required', 'message': 'This endpoint requires administrator privileges'}), 403
        return f(current_user, *args, **kwargs)
    return decorated

def analyst_required(f):
    """Decorator for analyst/investigator/compliance endpoints"""
    @wraps(f)
    @token_required
    def decorated(current_user, *args, **kwargs):
        allowed_roles = ['admin', 'analyst', 'investigator', 'compliance']
        if current_user.role not in allowed_roles:
            return jsonify({'error': 'Access required', 'message': 'This endpoint requires analyst privileges'}), 403
        return f(current_user, *args, **kwargs)
    return decorated

def api_key_required(f):
    """Decorator for API key authentication (for commercial clients)"""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({'error': 'API key required', 'message': 'Please provide X-API-Key header'}), 401
        
        db = SessionLocal()
        try:
            key_record = db.query(APIKey).filter(APIKey.key == api_key, APIKey.is_active == True).first()
            
            if not key_record:
                return jsonify({'error': 'Invalid API key'}), 401
            
            #Check expiration with Nairobi time
            nairobi_time = get_nairobi_time()
            if key_record.expires_at and key_record.expires_at < nairobi_time:
                return jsonify({'error': 'API key expired', 'message': 'Please renew your API key'}), 401
            
            request.api_user = key_record.user
            request.api_key_info = key_record
            
            return f(*args, **kwargs)
            
        except Exception as e:
            return jsonify({'error': 'API key validation failed', 'message': str(e)}), 500
        finally:
            db.close()
            
    return decorated