# auth/middleware.py
from functools import wraps
from flask import request, jsonify
import jwt
import os
from database.db_manager import SessionLocal, User

SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key')

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from header
        auth_header = request.headers.get('Authorization')
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == 'bearer':
                token = parts[1]
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            # Decode token
            data = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            current_user_id = data['user_id']
            
            # Get user from database
            db = SessionLocal()
            current_user = db.query(User).filter(User.id == current_user_id).first()
            db.close()
            
            if not current_user:
                return jsonify({'error': 'User not found'}), 401
            
            if not current_user.is_active:
                return jsonify({'error': 'User account disabled'}), 401
            
            # Pass user to endpoint
            return f(current_user, *args, **kwargs)
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
            
    return decorated

def admin_required(f):
    @wraps(f)
    @token_required
    def decorated(current_user, *args, **kwargs):
        if current_user.role != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(current_user, *args, **kwargs)
    return decorated

def api_key_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        db = SessionLocal()
        key_record = db.query(APIKey).filter(APIKey.key == api_key).first()
        db.close()
        
        if not key_record or not key_record.is_active:
            return jsonify({'error': 'Invalid or inactive API key'}), 401
        
        # Check expiration
        if key_record.expires_at and key_record.expires_at < datetime.now():
            return jsonify({'error': 'API key expired'}), 401
        
        # Attach user info to request
        request.api_user = key_record.user
        
        return f(*args, **kwargs)
    return decorated