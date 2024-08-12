from flask import Flask

def create_app():
    app = Flask(__name__)
    from .routes import main_blueprint
    app.register_blueprint(main_blueprint)
    return app