from flask import Blueprint, jsonify, request

from auth.jwt_auth import token_required
from .service import (
    build_customer_360,
    list_customers,
)


customer360_bp = Blueprint(
    "customer360",
    __name__,
)


@customer360_bp.route(
    "/customers",
    methods=["GET"],
)
@token_required
def get_customers(current_user):
    try:
        page = request.args.get(
            "page",
            default=1,
            type=int,
        )

        size = request.args.get(
            "size",
            default=20,
            type=int,
        )

        if page < 1:
            return jsonify({
                "status": "error",
                "message":
                    "page must be greater than 0",
            }), 400

        if size < 1 or size > 100:
            return jsonify({
                "status": "error",
                "message":
                    "size must be between 1 and 100",
            }), 400

        result = list_customers(
            page=page,
            size=size,
            search=request.args.get("search"),
            segment=request.args.get("segment"),
            risk_profile=request.args.get(
                "risk_profile"
            ),
            status=request.args.get("status"),
        )

        return jsonify({
            "status": "success",
            **result,
        }), 200

    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": str(exc),
        }), 500


@customer360_bp.route(
    "/customers/<customer_id>/360",
    methods=["GET"],
)
@token_required
def get_customer_360(
    current_user,
    customer_id,
):
    try:
        profile = build_customer_360(
            customer_id
        )

        if not profile:
            return jsonify({
                "status": "error",
                "message":
                    f"Customer {customer_id} not found",
            }), 404

        return jsonify({
            "status": "success",
            "customer_360": profile,
        }), 200

    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": str(exc),
        }), 500
