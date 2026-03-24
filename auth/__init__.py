# This file makes the auth directory a Python package
from auth.jwt_auth import token_required, admin_required, analyst_required, api_key_required
from auth.auth_routes import register_auth_routes

__all__ = [
    'token_required',
    'admin_required', 
    'analyst_required',
    'api_key_required',
    'register_auth_routes'
]