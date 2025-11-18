from flask import Blueprint, request, jsonify
from services.rag_pipline import query_with_fallback

rag_bp = Blueprint("rag", __name__)

def clean_text(text):
    text = text.replace("**", "")
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()

@rag_bp.route("/query", methods=["POST"])
def query():
    data = request.get_json() or {}
    q = data.get("query")

    if not q:
        return jsonify({"error": "missing 'query' in JSON body"}), 400

    try:
        resp = query_with_fallback(q)

        cleaned = clean_text(resp.text)

        return jsonify({
            "text": cleaned
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
