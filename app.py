from dotenv import load_dotenv
load_dotenv()
from flask import Flask
from routes.rag_route import query_bp
import os
def create_app():
    app = Flask(__name__)
    # Load env vars when running locally
    # (you can use python-dotenv in development)
    app.register_blueprint(query_bp, url_prefix="/api")
    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

