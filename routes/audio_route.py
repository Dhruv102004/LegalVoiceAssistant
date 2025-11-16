from flask import Blueprint, request, jsonify
from services.text_and_speech import speech_to_text

audio_bp = Blueprint("audio", __name__)

@audio_bp.route("/stt", methods=["POST"])
def translate_audio():
    if "audio" not in request.files:
        return jsonify({"error": "missing 'audio' file"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    try:
        translated_text = speech_to_text(file.stream, file.filename)
        return jsonify({"translated_text": translated_text})
    except Exception as e:
        return jsonify({"error": "translation failed", "detail": str(e)}), 500
