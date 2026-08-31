# finca_rules.py
"""
FINCA Rules Management Backend API
Provides CRUD operations and rule evaluation engine
Uses SQLite database (same pattern as transactions)
"""

from flask import request, jsonify
from datetime import datetime
import json
import logging

# Import database functions (same pattern as transactions)
from database.db_manager import (
    save_rule_to_db,
    get_all_rules_from_db,
    get_rule_from_db,
    delete_rule_from_db,
    toggle_rule_in_db,
    get_rules_stats_from_db,
    save_rule_history_to_db,
    get_rule_history_from_db,
    get_nairobi_time,
    init_default_rules
)

logger = logging.getLogger(__name__)

# Cache for performance (optional - like REAL_TIME_RISK_SCORES_PKL)
rules_cache = {}
cache_loaded = False

def load_rules_cache():
    """Load all rules into cache for faster access"""
    global rules_cache, cache_loaded
    rules = get_all_rules_from_db(include_inactive=True)
    rules_cache = {r['id']: r for r in rules}
    cache_loaded = True
    logger.info(f"Loaded {len(rules_cache)} rules into cache")

def get_cached_rules():
    """Get cached rules, load if not loaded"""
    global cache_loaded
    if not cache_loaded:
        load_rules_cache()
    return rules_cache

def invalidate_cache():
    """Invalidate cache (call after any rule modification)"""
    global cache_loaded
    cache_loaded = False

# ============================================
# EVALUATION FUNCTIONS
# ============================================

def get_transaction_field(transaction_data, field):
    """Get field value from transaction data (supports both lowercase and capitalized)"""
    field_mapping = {
        'device_type': ['device_type', 'Device_Type'],
        'beneficiary': ['beneficiary', 'Beneficiary'],
        'transaction_amount': ['transaction_amount', 'Transaction_Amount'],
        'tx_count_last_hour': ['tx_count_last_hour', 'Transaction_Frequency'],
        'transaction_hour': ['Transaction_Hour', 'transaction_hour'],
        'location': ['location', 'Location'],
        'channel': ['channel', 'Channel'],
        'customer_id': ['customer_id', 'Customer_ID']
    }
    
    if field in field_mapping:
        for key in field_mapping[field]:
            if key in transaction_data:
                return transaction_data[key]
    
    return transaction_data.get(field, 0)

def evaluate_rule(rule, transaction_data):
    """
    Evaluate a single rule against transaction data
    Returns: (triggered: bool, reason: str, risk_points: int)
    """
    try:
        conditions = rule.get('conditions', {})
        field = conditions.get('field')
        operator = conditions.get('operator')
        value = conditions.get('value')
        
        tx_value = get_transaction_field(transaction_data, field)
        
        triggered = False
        reason = None
        risk_points = 0
        
        if operator == 'is_new':
            if field == 'device_type':
                triggered = str(tx_value).lower() in ['unknown', 'new', 'unrecognized']
                reason = 'New/unrecognized device detected'
            elif field == 'beneficiary':
                triggered = 'NEW_BEN' in str(tx_value) or 'new' in str(tx_value).lower()
                reason = 'New beneficiary added'
        
        elif operator == 'greater_than':
            triggered = tx_value > value
            reason = f'{field} ({tx_value}) exceeds threshold ({value})'
        
        elif operator == 'greater_than_multiplier':
            avg = transaction_data.get('avg_transaction_amount', transaction_data.get('Avg_Transaction_Amount', 100000))
            if avg == 0:
                avg = 100000
            threshold = value * avg
            triggered = tx_value > threshold
            reason = f'Amount {tx_value} > {value}x average ({avg})'
        
        elif operator == 'is_new_location':
            normal_location = transaction_data.get('normal_location', 'Kampala')
            triggered = str(tx_value) != normal_location and str(tx_value) not in ['Local', 'Kampala', 'Wakiso']
            reason = f'Unusual location: {tx_value}'
        
        elif operator == 'between':
            if isinstance(value, list) and len(value) == 2:
                triggered = value[0] <= tx_value <= value[1]
                reason = f'{field} between {value[0]} and {value[1]}'
        
        if triggered:
            risk_points = rule.get('action', {}).get('risk_points', 0)
            # Update trigger count in database
            rule['trigger_count'] = rule.get('trigger_count', 0) + 1
            rule['last_triggered'] = get_nairobi_time().isoformat()
            save_rule_to_db(rule)
            invalidate_cache()
        
        return {
            'triggered': triggered,
            'reason': reason,
            'risk_points': risk_points,
            'rule_id': rule.get('id'),
            'rule_name': rule.get('name'),
            'decision': rule.get('action', {}).get('decision', 'CHALLENGE') if triggered else 'APPROVE',
            'alert': rule.get('action', {}).get('alert', True) and triggered
        }
        
    except Exception as e:
        logger.error(f"Error evaluating rule {rule.get('id')}: {e}")
        return {
            'triggered': False,
            'error': str(e)
        }

def evaluate_all_rules(transaction_data):
    """
    Evaluate all active rules against transaction data
    Returns: list of triggered rules and total risk points
    """
    triggered_rules = []
    total_risk_points = 0
    decisions = []
    
    # Get all active rules from database (same pattern as transactions)
    active_rules = get_all_rules_from_db(include_inactive=False)
    
    for rule in active_rules:
        result = evaluate_rule(rule, transaction_data)
        if result.get('triggered'):
            triggered_rules.append(result)
            total_risk_points += result.get('risk_points', 0)
            decisions.append(result.get('decision'))
    
    final_decision = 'APPROVE'
    if decisions:
        if 'BLOCK' in decisions:
            final_decision = 'BLOCK'
        elif 'CHALLENGE' in decisions:
            final_decision = 'CHALLENGE'
        elif 'FLAG' in decisions:
            final_decision = 'FLAG'
    
    return {
        'triggered_rules': triggered_rules,
        'total_risk_points': total_risk_points,
        'final_decision': final_decision,
        'rule_count': len(triggered_rules)
    }

# ============================================
# ROUTES - Using Database
# ============================================

def register_rules_routes(app):
    """Register all rules management routes with the Flask app"""
    
    # Initialize default rules in database
    init_default_rules()
    
    @app.route('/finca/v1/rules', methods=['GET'])
    def get_all_rules_simple():
        """Get all rules (simple GET, no pagination)"""
        try:
            rules = get_all_rules_from_db(include_inactive=True)
            rules.sort(key=lambda x: x.get('priority', 999))
            
            return jsonify({
                'status': 'success',
                'rules': rules,
                'total': len(rules)
            }), 200
            
        except Exception as e:
            logger.error(f"Error getting rules: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @app.route('/finca/v1/rules/list', methods=['POST'])
    def get_all_rules_paginated():
        """Get all rules with pagination (POST with JSON body)"""
        try:
            data = request.json or {}
            
            category = data.get('category')
            is_active = data.get('is_active')
            page = int(data.get('page', 1))
            size = int(data.get('size', 20))
            
            if page < 1:
                page = 1
            if size < 1 or size > 100:
                size = 20
            
            rules = get_all_rules_from_db(include_inactive=True)
            
            if category:
                rules = [r for r in rules if r.get('category') == category]
            
            if is_active is not None:
                is_active_bool = is_active.lower() == 'true'
                rules = [r for r in rules if r.get('is_active') == is_active_bool]
            
            rules.sort(key=lambda x: x.get('priority', 999))
            
            total = len(rules)
            start_idx = (page - 1) * size
            end_idx = start_idx + size
            paginated_rules = rules[start_idx:end_idx]
            
            return jsonify({
                'status': 'success',
                'rules': paginated_rules,
                'pagination': {
                    'page': page,
                    'size': size,
                    'total': total,
                    'total_pages': (total + size - 1) // size,
                    'has_next': end_idx < total,
                    'has_prev': page > 1
                }
            }), 200
            
        except Exception as e:
            logger.error(f"Error getting rules: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @app.route('/finca/v1/rules/<rule_id>', methods=['GET'])
    def get_rule(rule_id):
        """Get a specific rule by ID"""
        try:
            rule = get_rule_from_db(rule_id)
            if not rule:
                return jsonify({
                    'status': 'error',
                    'message': f'Rule {rule_id} not found'
                }), 404
            
            return jsonify({
                'status': 'success',
                'rule': rule
            }), 200
            
        except Exception as e:
            logger.error(f"Error getting rule {rule_id}: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/finca/v1/rules', methods=['POST'])
    def create_rule():
        """Create a new rule"""
        try:
            data = request.json
            
            required = ['name', 'conditions', 'action']
            missing = [f for f in required if f not in data]
            if missing:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required fields: {missing}'
                }), 400
            
            # Get all rules to determine next ID
            existing = get_all_rules_from_db(include_inactive=True)
            rule_id = f"R{len(existing) + 1:03d}"
            
            rule = {
                'id': rule_id,
                'name': data['name'],
                'description': data.get('description', ''),
                'conditions': data['conditions'],
                'action': data['action'],
                'priority': data.get('priority', len(existing) + 1),
                'is_active': data.get('is_active', True),
                'category': data.get('category', 'CUSTOM'),
                'created_at': get_nairobi_time().isoformat(),
                'updated_at': get_nairobi_time().isoformat(),
                'created_by': data.get('created_by', 'User'),
                'version': 1,
                'trigger_count': 0,
                'false_positive_rate': 0.0,
                'last_triggered': None
            }
            
            # Save to database
            save_rule_to_db(rule)
            invalidate_cache()
            
            return jsonify({
                'status': 'success',
                'message': f'Rule {rule_id} created successfully !!!!!!!!!!!!',
                'rule': rule
            }), 200
            
        except Exception as e:
            logger.error(f"Error creating rule: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/finca/v1/rules/<rule_id>', methods=['PUT'])
    def update_rule(rule_id):
        """Update an existing rule"""
        try:
            rule = get_rule_from_db(rule_id)
            if not rule:
                return jsonify({
                    'status': 'error',
                    'message': f'Rule {rule_id} not found'
                }), 404
            
            data = request.json
            
            # Save history before update
            save_rule_history_to_db(rule_id, rule.get('version', 1), dict(rule))
            
            # Update fields
            rule['name'] = data.get('name', rule['name'])
            rule['description'] = data.get('description', rule.get('description', ''))
            rule['conditions'] = data.get('conditions', rule['conditions'])
            rule['action'] = data.get('action', rule['action'])
            rule['priority'] = data.get('priority', rule.get('priority', 999))
            rule['is_active'] = data.get('is_active', rule.get('is_active', True))
            rule['category'] = data.get('category', rule.get('category', 'CUSTOM'))
            rule['updated_at'] = get_nairobi_time().isoformat()
            rule['version'] = rule.get('version', 0) + 1
            
            # Save to database
            save_rule_to_db(rule)
            invalidate_cache()
            
            return jsonify({
                'status': 'success',
                'message': f'Rule {rule_id} updated successfully',
                'rule': rule
            }), 200
            
        except Exception as e:
            logger.error(f"Error updating rule {rule_id}: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/finca/v1/rules/<rule_id>', methods=['DELETE'])
    def delete_rule(rule_id):
        """Soft delete a rule (mark as inactive)"""
        try:
            rule = get_rule_from_db(rule_id)
            if not rule:
                return jsonify({
                    'status': 'error',
                    'message': f'Rule {rule_id} not found'
                }), 404
            
            delete_rule_from_db(rule_id)
            invalidate_cache()
            
            return jsonify({
                'status': 'success',
                'message': f'Rule {rule_id} deleted successfully'
            }), 200
            
        except Exception as e:
            logger.error(f"Error deleting rule {rule_id}: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/finca/v1/rules/<rule_id>/toggle', methods=['POST'])
    def toggle_rule(rule_id):
        """Activate/deactivate a rule"""
        try:
            rule = get_rule_from_db(rule_id)
            if not rule:
                return jsonify({
                    'status': 'error',
                    'message': f'Rule {rule_id} not found'
                }), 404
            
            new_status = toggle_rule_in_db(rule_id)
            invalidate_cache()
            
            status = 'activated' if new_status else 'deactivated'
            
            return jsonify({
                'status': 'success',
                'message': f'Rule {rule_id} {status}',
                'rule': get_rule_from_db(rule_id)
            }), 200
            
        except Exception as e:
            logger.error(f"Error toggling rule {rule_id}: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/finca/v1/rules/simulate', methods=['POST'])
    def simulate_rule():
        """Simulate a rule against a transaction"""
        try:
            data = request.json
            
            if not data:
                return jsonify({
                    'status': 'error',
                    'message': 'No data provided'
                }), 400
            
            if 'rule_id' in data:
                rule = get_rule_from_db(data['rule_id'])
                if not rule:
                    return jsonify({
                        'status': 'error',
                        'message': f'Rule {data["rule_id"]} not found'
                    }), 404
                
                transaction = data.get('transaction', {})
                result = evaluate_rule(rule, transaction)
                
                return jsonify({
                    'status': 'success',
                    'simulation': {
                        'rule': rule,
                        'transaction': transaction,
                        'result': result
                    }
                }), 200
            
            elif 'transaction' in data:
                transaction = data['transaction']
                result = evaluate_all_rules(transaction)
                
                return jsonify({
                    'status': 'success',
                    'simulation': {
                        'transaction': transaction,
                        'result': result
                    }
                }), 200
            
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Missing rule_id or transaction data'
                }), 400
            
        except Exception as e:
            logger.error(f"Error simulating rule: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/finca/v1/rules/categories', methods=['GET'])
    def get_rule_categories():
        """Get all rule categories"""
        try:
            rules = get_all_rules_from_db(include_inactive=True)
            categories = list(set(r.get('category', 'CUSTOM') for r in rules))
            return jsonify({
                'status': 'success',
                'categories': sorted(categories)
            }), 200
            
        except Exception as e:
            logger.error(f"Error getting rule categories: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/finca/v1/rules/stats', methods=['GET'])
    def get_rule_stats():
        """Get rule statistics"""
        try:
            stats = get_rules_stats_from_db()
            
            return jsonify({
                'status': 'success',
                'stats': stats
            }), 200
            
        except Exception as e:
            logger.error(f"Error getting rule stats: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/finca/v1/rules/<rule_id>/history', methods=['GET'])
    def get_rule_history(rule_id):
        """Get rule change history"""
        try:
            rule = get_rule_from_db(rule_id)
            if not rule:
                return jsonify({
                    'status': 'error',
                    'message': f'Rule {rule_id} not found'
                }), 404
            
            history = get_rule_history_from_db(rule_id)
            
            return jsonify({
                'status': 'success',
                'rule_id': rule_id,
                'history': history,
                'total_versions': len(history)
            }), 200
            
        except Exception as e:
            logger.error(f"Error getting rule history {rule_id}: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500