# app/config.py
import os
import streamlit as st

class BaseConfig:
    """Base configuration shared by all environments."""
    SQLALCHEMY_TRACK_MODIFICATIONS = False  # Disable the SQLAlchemy event system
    DEBUG = False
    TESTING = False

class DevelopmentConfig(BaseConfig):
    """Configuration for the development environment."""
    # SQLite database file path for local development
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the config file
    # SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'dev_app.db')}"
    db_password = st.secrets["database"]["password"]
    SQLALCHEMY_DATABASE_URI = f"postgresql://postgres:{db_password}@db.pvoigvwnytuiwtzkxuyo.supabase.co:5432/postgres"
    DEBUG = True

class TestingConfig(BaseConfig):
    """Configuration for the testing environment."""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'test_app.db')}"
    TESTING = True

class ProductionConfig(BaseConfig):
    """Configuration for production environment."""
    SQLALCHEMY_DATABASE_URI = os.getenv("PROD_DATABASE_URL", "postgresql://user:password@localhost/prod_app")
    DEBUG = False

# Choose the configuration based on the APP_ENV environment variable
ENV = os.getenv("APP_ENV", "development")  # Default to development

if ENV == "development":
    Config = DevelopmentConfig
elif ENV == "testing":
    Config = TestingConfig
elif ENV == "production":
    Config = ProductionConfig
else:
    raise ValueError(f"Unknown APP_ENV: {ENV}")
