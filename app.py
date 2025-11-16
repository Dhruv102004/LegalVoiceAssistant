from dotenv import load_dotenv
load_dotenv()
from flask import Flask
from routes.rag_route import rag_bp
from routes.audio_route import audio_bp
import os

def create_app():
    app = Flask(__name__)
    
    app.register_blueprint(rag_bp, url_prefix="/rag")
    app.register_blueprint(audio_bp, url_prefix="/audio")
    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

