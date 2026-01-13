"""
Configuration management using environment variables.

Provides type-safe configuration with validation for deployment to
cloud environments (Digital Ocean, AWS, etc.).
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """
    Application configuration from environment variables.
    
    Attributes:
        redis_url: Redis connection URL
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        environment: Deployment environment (development, staging, production)
        chroma_persist_path: Path to ChromaDB persistence directory
    """
    
    redis_url: str = field(default_factory=lambda: os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    environment: str = field(default_factory=lambda: os.getenv('ENVIRONMENT', 'development'))
    chroma_persist_path: str = field(default_factory=lambda: os.getenv('CHROMA_PERSIST_PATH', './database'))
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_log_levels:
            raise ValueError(f"Invalid LOG_LEVEL: {self.log_level}")


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get application configuration.
    
    Returns singleton Config instance loaded from environment variables.
    
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """
    Reset configuration singleton.
    
    Used for testing to allow environment variable changes to take effect.
    """
    global _config
    _config = None
