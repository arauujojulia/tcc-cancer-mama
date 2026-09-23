from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    APP: Flask = Flask(__name__)

    APP.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:minhasenha123@localhost:5432/tcc_cancer_mama"

    db.init_app(APP)
    CORS(APP)

    @APP.route("/health")
    def health():
        return {"status":"ok"}

    return APP

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)

