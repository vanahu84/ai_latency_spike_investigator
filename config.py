"""
Configuration management for the Latency Spike Investigator.
Handles environment variables and application settings.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Configuration class for managing environment variables and settings."""
    
    # API Configuration
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    NEW_RELIC_API_KEY: Optional[str] = os.getenv("NEW_RELIC_API_KEY")
    DATADOG_API_KEY: Optional[str] = os.getenv("DATADOG_API_KEY")
    DATADOG_APP_KEY: Optional[str] = os.getenv("DATADOG_APP_KEY")
    
    # Database Configuration
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "data/latency_investigator.db")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Application Configuration
    SPIKE_THRESHOLD_MS: float = float(os.getenv("SPIKE_THRESHOLD_MS", "1000"))
    CORRELATION_WINDOW_MINUTES: int = int(os.getenv("CORRELATION_WINDOW_MINUTES", "15"))
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    
    # Deployment Configuration
    PORT: int = int(os.getenv("PORT", "8501"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Rate Limiting
    GEMINI_REQUESTS_PER_MINUTE: int = int(os.getenv("GEMINI_REQUESTS_PER_MINUTE", "60"))
    
    @classmethod
    def validate_required_config(cls) -> list[str]:
        """
        Validate that required configuration is present.
        Returns a list of missing required configuration keys.
        """
        missing_config = []
        
        # Check for required API keys based on deployment
        if not cls.GEMINI_API_KEY:
            missing_config.append("GEMINI_API_KEY")
            
        return missing_config
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get the SQLite database URL."""
        return f"sqlite:///{cls.SQLITE_DB_PATH}"
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment."""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"


# Create a global config instance
config = Config()