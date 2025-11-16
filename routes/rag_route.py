from flask import Blueprint, request, jsonify
from services.rag_pipline import query_with_fallback

query_bp = Blueprint("query", __name__)

@query_bp.route("/query", methods=["POST"])
def query():
    data = request.get_json() or {}
    q = data.get("query")

    if not q:
        return jsonify({"error": "missing 'query' in JSON body"}), 400

    try:
        resp = query_with_fallback(q)

        # resp.text is the clean Gemini output
        return jsonify({
            "text": resp.text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
