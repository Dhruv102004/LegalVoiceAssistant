from flask import Blueprint, request, jsonify, send_file
from services.text_and_speech import speech_to_text, text_to_speech
from io import BytesIO

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
    
@audio_bp.route("/tts", methods=["POST"])
def generate_audio():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "JSON body required"}), 400

    text = data.get("text", "").strip()
    lang = data.get("lang", "").strip().lower()

    if not text:
        return jsonify({"error": "'text' field cannot be empty"}), 400

    if lang not in ("hi", "en"):
        return jsonify({"error": "'lang' must be either 'hi' or 'en'"}), 400

    try:
        # Get WAV bytes from TTS
        wav_bytes = text_to_speech(text, lang)

        buf = BytesIO(wav_bytes)
        buf.seek(0)

        return send_file(
            buf,
            mimetype="audio/wav",
            as_attachment=True,
            download_name=f"tts_{lang}.wav"
        )

    except Exception as e:
        return jsonify({
            "error": "synthesis failed",
            "detail": str(e)
        }), 500
