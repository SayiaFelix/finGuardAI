# auth/auth_routes.py
from flask import request, jsonify
from datetime import datetime, timedelta
import jwt
import secrets
from database.db_manager import SessionLocal, User, APIKey, RefreshToken
from auth.jwt_auth import (
    generate_access_token, generate_refresh_token, generate_api_key,
    token_required, admin_required, SECRET_KEY
)

def register_auth_routes(app):
    """Register all authentication routes with the Flask app"""
    
    @app.route('/v1/api/auth/register', methods=['POST'])
    def register():
        """Register a new user"""
        try:
            data = request.get_json()
            
            # Validating required fields
            required_fields = ['email', 'username', 'password']
            for field in required_fields:
                if not data.get(field):
                    return jsonify({'error': f'{field} is required'}), 400
            
            # Validating role if provided
            valid_roles = ['admin', 'analyst', 'viewer', 'investigator', 'compliance']
            role = data.get('role', 'analyst')
            
            if role not in valid_roles:
                return jsonify({
                    'error': f'Invalid role. Must be one of: {", ".join(valid_roles)}'
                }), 400
            
            db = SessionLocal()
            
            try:
                # Check if user exists
                existing = db.query(User).filter(
                    (User.email == data['email']) | (User.username == data['username'])
                ).first()
                
                if existing:
                    return jsonify({'error': 'User already exists with this email or username'}), 409
                
                #new user
                new_user = User(
                    email=data['email'],
                    username=data['username'],
                    role=role,
                    is_active=True
                )
                new_user.set_password(data['password'])
                
                db.add(new_user)
                db.commit()
                db.refresh(new_user)
                
                ### Getting data before closing session
                user_data = {
                    'id': new_user.id,
                    'email': new_user.email,
                    'username': new_user.username,
                    'role': new_user.role
                }
                
                return jsonify({
                    'status': 'success',
                    'message': f'${new_user.username} User registered successfully !!!!!!!!!!',
                    'user': user_data
                }), 201
                
            finally:
                db.close()
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    @app.route('/v1/api/auth/login', methods=['POST'])
    def login(): 
        """Login and get JWT tokens"""
        try:
            data = request.get_json()
            
            if not data.get('username') or not data.get('password'):
                return jsonify({'error': 'Username and password required'}), 400
            
            db = SessionLocal()
            
            try:
                # Finding user by username or email
                user = db.query(User).filter(
                    (User.username == data['username']) | (User.email == data['username'])
                ).first()
                
                if not user:
                    return jsonify({'error': 'Invalid credentials'}), 401
                
                # Check password
                password_valid = user.check_password(data['password'])
                
                if not password_valid:
                    return jsonify({'error': 'Invalid credentials'}), 401
           
                if not user.is_active:
                    return jsonify({'error': 'Account disabled. Contact administrator.'}), 403
                
                # Update last login (this needs session, do it before closing)
                user.last_login = datetime.utcnow()
                db.commit()
                
                #user data
                user_id = user.id
                username = user.username
                role = user.role
                
                # Generate tokens (doesn't need database)
                access_token = generate_access_token(user_id, username, role)
                refresh_token = generate_refresh_token(user_id)
                
                # Store refresh token
                db_refresh = RefreshToken(
                    token=refresh_token,
                    user_id=user_id,
                    expires_at=datetime.utcnow() + timedelta(seconds=86400)
                )
                db.add(db_refresh)
                db.commit()
                
                return jsonify({
                    'status': 'success',
                    'message': f'{username} login successful !!!!!!!!',
                    'access_token': access_token,
                    'refresh_token': refresh_token,
                    'user': {
                        'id': user_id,
                        'username': username,
                        'email': user.email,
                        'role': role
                    }
                }), 200
                
            finally:
                db.close()  
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    @app.route('/v1/api/auth/refresh', methods=['POST'])
    def refresh():
        """Refresh expired access token"""
        try:
            data = request.get_json()
            refresh_token = data.get('refresh_token')
            
            if not refresh_token:
                return jsonify({'error': 'Refresh token required'}), 400
            
            # Verify refresh token
            try:
                payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=['HS256'])
                if payload.get('type') != 'refresh':
                    return jsonify({'error': 'Invalid token type'}), 401
            except jwt.ExpiredSignatureError:
                return jsonify({'error': 'Refresh token expired. Please login again.'}), 401
            except jwt.InvalidTokenError:
                return jsonify({'error': 'Invalid refresh token'}), 401
            
            db = SessionLocal()
            
            # Check if token exists and is not revoked
            token_record = db.query(RefreshToken).filter(
                RefreshToken.token == refresh_token,
                RefreshToken.revoked == False
            ).first()
            
            if not token_record:
                db.close()
                return jsonify({'error': 'Refresh token not found or revoked'}), 401
            
            # Get user
            user = db.query(User).filter(User.id == payload['user_id']).first()
            
            if not user or not user.is_active:
                db.close()
                return jsonify({'error': 'User not found or inactive'}), 401
            
            # Generate new access token
            new_access_token = generate_access_token(user.id, user.username, user.role)
            
            db.close()
            
            return jsonify({
                'status': 'success',
                'access_token': new_access_token
            }), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/v1/api/auth/logout', methods=['POST'])
    @token_required
    def logout(current_user):
        """Logout and revoke refresh token"""
        try:
            data = request.get_json()
            refresh_token = data.get('refresh_token')
            
            if refresh_token:
                db = SessionLocal()
                token_record = db.query(RefreshToken).filter(
                    RefreshToken.token == refresh_token,
                    RefreshToken.user_id == current_user.id
                ).first()
                
                if token_record:
                    token_record.revoked = True
                    db.commit()
                
                db.close()
            
            return jsonify({
                'status': 'success',
                'message': 'Logged out successfully !!!!!!!!!'
            }), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/v1/api/auth/me', methods=['GET'])
    @token_required
    def get_current_user(current_user):
        """Get current user info"""
        return jsonify({
            'status': 'success',
            'message': f'Hello {current_user.username} !!!!!!!!!',
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'email': current_user.email,
                'role': current_user.role,
                'is_active': current_user.is_active,
                'created_at': current_user.created_at.isoformat(),
                'last_login': current_user.last_login.isoformat() if current_user.last_login else None
            }
        }), 200
    
    @app.route('/v1/api/auth/api-keys', methods=['POST'])
    @token_required
    def create_api_key(current_user):
        """Generate API key for programmatic access"""
        try:
            data = request.get_json()
            name = data.get('name', f"{current_user.username}'s API Key")
            tier = data.get('tier', 'free')
            
            db = SessionLocal()
            
            api_key = APIKey(
                key=generate_api_key(),
                name=name,
                user_id=current_user.id,
                tier=tier,
                rate_limit=100 if tier == 'free' else 500 if tier == 'basic' else 5000,
                is_active=True
            )
            
            db.add(api_key)
            db.commit()
            db.refresh(api_key)
            db.close()
            
            return jsonify({
                'status': 'success',
                'api_key': api_key.key,
                'name': api_key.name,
                'tier': api_key.tier,
                'rate_limit': api_key.rate_limit
            }), 201
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/v1/api/auth/api-keys', methods=['GET'])
    @token_required
    def list_api_keys(current_user):
        """List all API keys for current user"""
        try:
            db = SessionLocal()
            keys = db.query(APIKey).filter(APIKey.user_id == current_user.id).all()
            
            result = []
            for key in keys:
                result.append({
                    'key': key.key,
                    'name': key.name,
                    'tier': key.tier,
                    'rate_limit': key.rate_limit,
                    'is_active': key.is_active,
                    'created_at': key.created_at.isoformat(),
                    'expires_at': key.expires_at.isoformat() if key.expires_at else None
                })
            
            db.close()
            
            return jsonify({
                'status': 'success',
                'api_keys': result
            }), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/v1/api/auth/api-keys/<key>', methods=['DELETE'])
    @token_required
    def revoke_api_key(current_user, key):
        """Revoke an API key"""
        try:
            db = SessionLocal()
            api_key = db.query(APIKey).filter(
                APIKey.key == key,
                APIKey.user_id == current_user.id
            ).first()
            
            if not api_key:
                db.close()
                return jsonify({'error': 'API key not found'}), 404
            
            api_key.is_active = False
            db.commit()
            db.close()
            
            return jsonify({
                'status': 'success',
                'message': 'API key revoked successfully'
            }), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    ### Admin-only endpoints
    @app.route('/v1/api/admin/users', methods=['GET'])
    @admin_required
    def list_users(current_user):
        """List all users (admin only)"""
        try:
            db = SessionLocal()
            users = db.query(User).all()
            
            result = []
            for user in users:
                result.append({
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role,
                    'is_active': user.is_active,
                    'created_at': user.created_at.isoformat(),
                    'last_login': user.last_login.isoformat() if user.last_login else None
                })
            
            db.close()
            
            return jsonify({
                'status': 'success',
                'message': 'User list retrieved successfully !!!!!!!!!!!!',
                'users': result
            }), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/v1/api/admin/users/<int:user_id>/role', methods=['PUT'])
    @admin_required
    def update_user_role(current_user, user_id):
        """Update user role (admin only)"""
        try:
            data = request.get_json()
            new_role = data.get('role')
            
            # Valid roles
            if new_role not in ['admin', 'analyst', 'viewer', 'investigator', 'compliance']:
                return jsonify({'error': 'Invalid role'}), 400
            
            db = SessionLocal()
            user = db.query(User).filter(User.id == user_id).first()
            
            if not user:
                db.close()
                return jsonify({'error': 'User not found'}), 404
            
            user.role = new_role
            db.commit()
            db.close()
            
            return jsonify({
                'status': 'success',
                'message': f"User {user.username} role updated to {new_role} successfully !!!!!!!!!"
            }), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/v1/api/admin/users/<int:user_id>/disable', methods=['PUT'])
    @admin_required
    def disable_user(current_user, user_id):
        """Disable a user account (admin only)"""
        try:
            db = SessionLocal()
            user = db.query(User).filter(User.id == user_id).first()
            
            if not user:
                db.close()
                return jsonify({'error': 'User not found'}), 404
            
            if user.id == current_user.id:
                db.close()
                return jsonify({'error': 'Cannot disable your own account'}), 400
            
            user.is_active = False
            db.commit()
            db.close()
            
            return jsonify({
                'status': 'success',
                'message': f"User {user.username} has been disabled successfully !!!!!!!!!"
            }), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/v1/api/admin/users/<int:user_id>/enable', methods=['PUT'])
    @admin_required
    def enable_user(current_user, user_id):
        """Enable a disabled user account (admin only)"""
        try:
            db = SessionLocal()
            user = db.query(User).filter(User.id == user_id).first()
            
            if not user:
                db.close()
                return jsonify({'error': 'User not found'}), 404
            
            user.is_active = True
            db.commit()
            db.close()
            
            return jsonify({
                'status': 'success',
                'message': f"User {user.username} has been enabled successfully !!!!!!!!!"
            }), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500