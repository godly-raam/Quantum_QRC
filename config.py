# config.py
"""
Configuration settings for Q-Fleet QRC Backend
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    API_TITLE: str = "Q-Fleet Quantum Reservoir Computing API"
    API_VERSION: str = "2.0.0"
    API_DESCRIPTION: str = "First QRC system for real-time vehicle routing optimization"
    
    # CORS Settings
    CORS_ORIGINS: list = ["*"]  # In production, restrict to specific domains
    
    # QRC Configuration
    QRC_NUM_QUBITS: int = int(os.getenv("QRC_NUM_QUBITS", "8"))
    QRC_COUPLING_STRENGTH: float = float(os.getenv("QRC_COUPLING_STRENGTH", "0.1"))
    QRC_TRAINING_INSTANCES: int = int(os.getenv("QRC_TRAINING_INSTANCES", "20"))
    QRC_AUTO_TRAIN: bool = os.getenv("QRC_AUTO_TRAIN", "true").lower() == "true"
    
    # Problem Constraints
    MAX_LOCATIONS: int = 8
    MAX_VEHICLES: int = 4
    MIN_LOCATIONS: int = 2
    MIN_VEHICLES: int = 1
    
    # QAOA Configuration
    DEFAULT_QAOA_REPS: int = 4
    MAX_QAOA_REPS: int = 6
    
    # Performance
    MAX_WORKERS: int = 1  # Render free tier limitation
    TIMEOUT_SECONDS: int = 30
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Deployment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Feature flags
FEATURES = {
    "qrc_enabled": True,
    "traffic_jam_simulation": True,
    "priority_delivery": True,
    "benchmarking": True,
    "noise_analysis": False,  # Enable after adding noise_analysis.py
}

# Model configurations
QRC_CONFIG = {
    "small": {"qubits": 6, "training_instances": 15},
    "medium": {"qubits": 8, "training_instances": 20},
    "large": {"qubits": 10, "training_instances": 30},
}

# Select configuration based on environment
ACTIVE_QRC_CONFIG = QRC_CONFIG["medium"]  # Default

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": settings.LOG_LEVEL,
        },
    },
    "root": {
        "level": settings.LOG_LEVEL,
        "handlers": ["console"],
    },
}