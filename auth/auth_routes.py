from flask import request, jsonify
from datetime import datetime, timedelta
import jwt
import secrets
import string
from database.db_manager import SessionLocal, User, APIKey, RefreshToken
from auth.jwt_auth import (
    generate_access_token, generate_refresh_token, generate_api_key,
    token_required, admin_required, SECRET_KEY
)

def register_auth_routes(app):
    """Register all authentication routes with the Flask app"""
    
    # ==================== PUBLIC AUTH ROUTES ====================
    
    @app.route('/v1/api/auth/register', methods=['POST'])
    def register():
        """Public registration - users can register themselves"""
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
            
            # Non-admin users cannot register as admin
            if role == 'admin':
                return jsonify({'error': 'Cannot register as admin. Contact system administrator.'}), 403
            
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
                
                user_data = {
                    'id': new_user.id,
                    'email': new_user.email,
                    'username': new_user.username,
                    'role': new_user.role
                }
                
                return jsonify({
                    'status': 'success',
                    'message': f'{new_user.username} registered successfully!',
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
                
                user.last_login = datetime.utcnow()
                db.commit()
                
                user_id = user.id
                username = user.username
                role = user.role
                
                # Generate tokens
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
                    'message': f'{username} login successful!',
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
            
            # Checking if token exists and is not revoked
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
                'message': 'Logged out successfully!'
            }), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/v1/api/auth/me', methods=['GET'])
    @token_required
    def get_current_user(current_user):
        """Get current user info"""
        return jsonify({
            'status': 'success',
            'message': f'Hello {current_user.username}!',
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
    
    # ==================== API KEY MANAGEMENT ====================
    
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
    
    # ==================== ADMIN USER MANAGEMENT ====================
    
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
                'message': 'User list retrieved successfully!',
                'users': result
            }), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/v1/api/admin/create_users', methods=['POST'])
    @admin_required
    def create_user_admin(current_user):
        """Admin creates a new user (full control)"""
        try:
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['email', 'username', 'password', 'role']
            for field in required_fields:
                if not data.get(field):
                    return jsonify({'error': f'{field} is required'}), 400
            
            # Validate role
            valid_roles = ['admin', 'analyst', 'viewer', 'investigator', 'compliance']
            role = data.get('role')
            
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
                
                # Create new user
                new_user = User(
                    email=data['email'],
                    username=data['username'],
                    role=role,
                    is_active=data.get('is_active', True)
                )
                new_user.set_password(data['password'])
                
                db.add(new_user)
                db.commit()
                db.refresh(new_user)
                
                user_data = {
                    'id': new_user.id,
                    'email': new_user.email,
                    'username': new_user.username,
                    'role': new_user.role,
                    'is_active': new_user.is_active,
                    'created_at': new_user.created_at.isoformat()
                }
                
                return jsonify({
                    'status': 'success',
                    'message': f'User {new_user.username} created successfully!',
                    'user': user_data
                }), 201
                
            finally:
                db.close()
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/v1/api/admin/users/<int:user_id>', methods=['GET'])
    @admin_required
    def get_user_details(current_user, user_id):
        """Get single user details by ID (admin only)"""
        try:
            db = SessionLocal()
            
            try:
                user = db.query(User).filter(User.id == user_id).first()
                
                if not user:
                    return jsonify({'error': 'User not found'}), 404
                
                user_data = {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role,
                    'is_active': user.is_active,
                    'created_at': user.created_at.isoformat(),
                    'last_login': user.last_login.isoformat() if user.last_login else None
                }
                
                return jsonify({
                    'status': 'success',
                    'user': user_data
                }), 200
                
            finally:
                db.close()
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/v1/api/admin/users/<int:user_id>', methods=['DELETE'])
    @admin_required
    def delete_user(current_user, user_id):
        """Delete a user (admin only)"""
        try:
            db = SessionLocal()
            
            try:
                user = db.query(User).filter(User.id == user_id).first()
                
                if not user:
                    return jsonify({'error': 'User not found'}), 404
                
                # Cannot delete your own account
                if user.id == current_user.id:
                    return jsonify({'error': 'Cannot delete your own account'}), 400
                
                # Delete user's refresh tokens first
                db.query(RefreshToken).filter(RefreshToken.user_id == user_id).delete()
                
                # Delete user's API keys
                db.query(APIKey).filter(APIKey.user_id == user_id).delete()
                
                # Delete the user
                db.delete(user)
                db.commit()
                
                return jsonify({
                    'status': 'success',
                    'message': f'User {user.username} deleted successfully!'
                }), 200
                
            finally:
                db.close()
                
        except Exception as e:
            db.rollback()
            return jsonify({'error': str(e)}), 500
    
    @app.route('/v1/api/admin/users/<int:user_id>/update', methods=['PUT'])
    @admin_required
    def update_user(current_user, user_id):
        """Update user details (admin only)"""
        try:
            data = request.get_json()
            
            db = SessionLocal()
            
            try:
                user = db.query(User).filter(User.id == user_id).first()
                
                if not user:
                    return jsonify({'error': 'User not found'}), 404
                
                # Update username if provided
                if 'username' in data and data['username']:
                    existing = db.query(User).filter(
                        User.username == data['username'],
                        User.id != user_id
                    ).first()
                    if existing:
                        return jsonify({'error': 'Username already taken'}), 409
                    user.username = data['username']
                
                # Update email if provided
                if 'email' in data and data['email']:
                    existing = db.query(User).filter(
                        User.email == data['email'],
                        User.id != user_id
                    ).first()
                    if existing:
                        return jsonify({'error': 'Email already taken'}), 409
                    user.email = data['email']
                
                # Update role if provided
                if 'role' in data and data['role']:
                    valid_roles = ['admin', 'analyst', 'viewer', 'investigator', 'compliance']
                    if data['role'] not in valid_roles:
                        return jsonify({'error': 'Invalid role'}), 400
                    user.role = data['role']
                
                # Update password if provided
                if 'password' in data and data['password']:
                    user.set_password(data['password'])
                
                db.commit()
                db.refresh(user)
                
                user_data = {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role,
                    'is_active': user.is_active,
                    'created_at': user.created_at.isoformat(),
                    'last_login': user.last_login.isoformat() if user.last_login else None
                }
                
                return jsonify({
                    'status': 'success',
                    'message': 'User updated successfully',
                    'user': user_data
                }), 200
                
            finally:
                db.close()
                
        except Exception as e:
            db.rollback()
            return jsonify({'error': str(e)}), 500
    
    @app.route('/v1/api/admin/users/<int:user_id>/role', methods=['PUT'])
    @admin_required
    def update_user_role(current_user, user_id):
        """Update user role only (admin only)"""
        try:
            data = request.get_json()
            new_role = data.get('role')
            
            if not new_role:
                return jsonify({'error': 'Role is required'}), 400
            
            valid_roles = ['admin', 'analyst', 'viewer', 'investigator', 'compliance']
            if new_role not in valid_roles:
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
                'message': f"User {user.username} role updated to {new_role} successfully!"
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
                'message': f"User {user.username} has been disabled successfully!"
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
                'message': f"User {user.username} has been enabled successfully!"
            }), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/v1/api/admin/users/<int:user_id>/reset-password', methods=['POST'])
    @admin_required
    def reset_password(current_user, user_id):
        """Reset password - sends email or generates temporary password (admin only)"""
        try:
            data = request.get_json()
            reset_type = data.get('type', 'email')  # 'email' or 'temporary'
            
            db = SessionLocal()
            
            try:
                user = db.query(User).filter(User.id == user_id).first()
                
                if not user:
                    return jsonify({'error': 'User not found'}), 404
                
                if reset_type == 'temporary':
                    # Generate temporary password
                    temp_password = generate_temporary_password()
                    user.set_password(temp_password)
                    db.commit()
                    
                    return jsonify({
                        'status': 'success',
                        'message': 'Temporary password generated successfully',
                        'temporaryPassword': temp_password
                    }), 200
                    
                else:
                    # Send reset email (you can integrate with email service here)
                    reset_token = generate_reset_token(user.id)
                    
                    # In production, you would send an email here
                    # For now, return the reset link
                    reset_link = f"/auth/reset-password?token={reset_token}"
                    
                    return jsonify({
                        'status': 'success',
                        'message': 'Password reset email sent',
                        'reset_token': reset_token,
                        'reset_link': reset_link,
                        'email': user.email
                    }), 200
                
            finally:
                db.close()
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500


# ==================== HELPER FUNCTIONS ====================

def generate_temporary_password(length=12):
    """Generate a random temporary password"""
    characters = string.ascii_letters + string.digits + "!@#$%^&*"
    password = ''.join(secrets.choice(characters) for _ in range(length))
    return password

def generate_reset_token(user_id):
    """Generate a password reset token"""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=24),
        'type': 'reset_password',
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')