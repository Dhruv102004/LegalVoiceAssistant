import os
import mimetypes
import tempfile
from google import genai
from google.genai import types
from google.cloud import texttospeech

API_KEY = os.environ.get("GENAI_API_KEY", "")

client = genai.Client(api_key=API_KEY)
_client = texttospeech.TextToSpeechClient()

VOICE_MAP = {
    "hi": {
        "language_code": "hi-IN",
        "voice_name": "hi-IN-Standard-F"
    },
    "en": {
        "language_code": "en-US",
        "voice_name": "en-US-Standard-C"
    }
}

def _guess_mime(filename):
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def speech_to_text(file_stream, filename, prompt_text="Translate this audio to English."):
    # ─────────────────────────────────────
    # Step 1: Save temporary file locally
    # ─────────────────────────────────────
    suffix = os.path.splitext(filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name

    data = file_stream.read()
    if isinstance(data, str):
        data = data.encode("utf-8")
    tmp.write(data)
    tmp.close()

    try:
        # ─────────────────────────────────────
        # Step 2: Upload file
        # ─────────────────────────────────────
        mime_type = _guess_mime(filename)
        upload_config = types.UploadFileConfig(
            display_name=filename,
            mime_type=mime_type
        )
        uploaded = client.files.upload(file=tmp_path, config=upload_config)

        # ─────────────────────────────────────
        # Step 3: Wait until file is ACTIVE
        # ─────────────────────────────────────
        while True:
            status = client.files.get(uploaded.name)
            if status.state == "ACTIVE":
                break
            if status.state == "FAILED":
                raise Exception("File processing failed on Gemini.")
            time.sleep(0.5)

        # ─────────────────────────────────────
        # Step 4: Call STT model
        # ─────────────────────────────────────
        response = client.models.generate_content(
            model="gemini-2.5-flash",   # supports audio
            contents=[
                types.Part(text=prompt_text),
                types.Part(
                    audio=types.AudioData(
                        file_uri=status.uri,
                        mime_type=mime_type
                    )
                )
            ]
        )

        return response.text

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def translate_to_hindi(english_text):
 
    if not english_text:
        return ""

    prompt = (
        "Translate the following English text into fluent, natural Hindi. "
        "Do not add explanations or commentary — return only the Hindi translation.\n\n"
        f"English text:\n{english_text}\n\nHindi translation:"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[types.Part(text=prompt)]
    )

    return response.text.strip()


def text_to_speech(text, lang):
    if not text:
        raise ValueError("Empty text provided")

    voice_settings = VOICE_MAP.get(lang, VOICE_MAP["en"])
    language_code = voice_settings["language_code"]
    voice_name = voice_settings["voice_name"]

    if lang == "hi":
        text_to_speak = translate_to_hindi(text)
    else:
        text_to_speak = text

    synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        effects_profile_id=['small-bluetooth-speaker-class-device'],
        speaking_rate=1,
        pitch=1
    )

    response = _client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    return response.audio_content
