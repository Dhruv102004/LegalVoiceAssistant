import os
import mimetypes
import tempfile
from google import genai
from google.genai import types

API_KEY = os.environ.get("GENAI_API_KEY", "")
client = genai.Client(api_key=API_KEY)

def _guess_mime(filename):
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"

def speech_to_text(file_stream, filename, prompt_text="Translate this audio to English."):

    mime_type = _guess_mime(filename)
    suffix = os.path.splitext(filename)[1] or ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    try:
        data = file_stream.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        tmp.write(data)
        tmp.close()

        upload_config = types.UploadFileConfig(display_name=filename, mime_type=mime_type)
        uploaded = client.files.upload(file=tmp_path, config=upload_config)

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                types.Part(text=prompt_text),
                types.Part(
                    file_data=types.FileData(
                        mime_type=mime_type,
                        file_uri=uploaded.uri
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
