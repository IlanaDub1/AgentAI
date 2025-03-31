from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import Config

# Create engine
engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)#, echo=True,pool_pre_ping=True

# Create session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Session = SessionLocal()

# Base class for models
Base = declarative_base()

def create_database():
    print("Creating db...")
    """Utility to create all tables based on the models."""
    from models import user, message, result  # Import all models here to ensure they are registered
    Base.metadata.create_all(bind=engine)
