# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Configuration loader for SigmaK.

Loads config.yaml with environment variable overrides and exposes typed settings.
Preserves backward-compatibility with existing env-based config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration."""

    sqlite_path: Path


@dataclass(frozen=True)
class ChromaConfig:
    """ChromaDB configuration."""

    persist_directory: Path
    embedding_model: str
    llm_cache_similarity_threshold: float


@dataclass(frozen=True)
class LLMConfig:
    """LLM configuration."""

    model: str
    temperature: float


@dataclass(frozen=True)
class DriftConfig:
    """Drift detection configuration."""

    review_cron: str
    sample_size: int
    low_confidence_threshold: float
    drift_threshold: float


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""

    level: str


@dataclass(frozen=True)
class Config:
    """Root configuration."""

    database: DatabaseConfig
    chroma: ChromaConfig
    llm: LLMConfig
    drift: DriftConfig
    logging: LoggingConfig

    # Backward-compatibility fields (env-only)
    redis_url: str
    environment: str

    @property
    def log_level(self) -> str:
        """Backward-compatibility alias for logging.level."""
        return self.logging.level

    @property
    def chroma_persist_path(self) -> str:
        """Backward-compatibility alias for chroma.persist_directory."""
        return str(self.chroma.persist_directory)


def load_config(path: Optional[Path] = None) -> Config:
    """
    Load configuration from YAML file with environment variable overrides.

    Args:
        path: Path to config.yaml. If None, defaults to repo_root/config.yaml.

    Returns:
        Validated Config object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If required config keys are missing or invalid.
    """
    if path is None:
        # Default to config.yaml in repo root (one level up from src/sigmak)
        path = Path(__file__).parent.parent.parent / "config.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a YAML dictionary")

    # Extract and validate database config
    db_raw = raw.get("database", {})
    if not isinstance(db_raw, dict):
        raise ValueError("'database' must be a dictionary")
    sqlite_path = os.environ.get("SIGMAK_SQLITE_PATH", db_raw.get("sqlite_path"))
    if not sqlite_path:
        raise ValueError("'database.sqlite_path' is required")
    database = DatabaseConfig(sqlite_path=Path(sqlite_path))

    # Extract and validate chroma config
    chroma_raw = raw.get("chroma", {})
    if not isinstance(chroma_raw, dict):
        raise ValueError("'chroma' must be a dictionary")
    persist_dir = os.environ.get("CHROMA_PERSIST_PATH", chroma_raw.get("persist_directory"))
    if not persist_dir:
        raise ValueError("'chroma.persist_directory' is required")
    embedding_model = os.environ.get("SIGMAK_EMBEDDING_MODEL", chroma_raw.get("embedding_model"))
    if not embedding_model:
        raise ValueError("'chroma.embedding_model' is required")
    similarity_threshold = chroma_raw.get("llm_cache_similarity_threshold", 0.8)
    if not isinstance(similarity_threshold, (int, float)):
        raise ValueError("'chroma.llm_cache_similarity_threshold' must be a number")
    chroma = ChromaConfig(
        persist_directory=Path(persist_dir),
        embedding_model=embedding_model,
        llm_cache_similarity_threshold=float(similarity_threshold),
    )

    # Extract and validate LLM config
    llm_raw = raw.get("llm", {})
    if not isinstance(llm_raw, dict):
        raise ValueError("'llm' must be a dictionary")
    llm_model = os.environ.get("SIGMAK_LLM_MODEL", llm_raw.get("model"))
    if not llm_model:
        raise ValueError("'llm.model' is required")
    llm_temp = llm_raw.get("temperature", 0.0)
    if not isinstance(llm_temp, (int, float)):
        raise ValueError("'llm.temperature' must be a number")
    llm = LLMConfig(model=llm_model, temperature=float(llm_temp))

    # Extract and validate drift config
    drift_raw = raw.get("drift", {})
    if not isinstance(drift_raw, dict):
        raise ValueError("'drift' must be a dictionary")
    drift = DriftConfig(
        review_cron=drift_raw.get("review_cron", "0 3 * * *"),
        sample_size=drift_raw.get("sample_size", 100),
        low_confidence_threshold=drift_raw.get("low_confidence_threshold", 0.6),
        drift_threshold=drift_raw.get("drift_threshold", 0.2),
    )

    # Extract and validate logging config
    logging_raw = raw.get("logging", {})
    if not isinstance(logging_raw, dict):
        raise ValueError("'logging' must be a dictionary")
    log_level = os.environ.get("LOG_LEVEL", logging_raw.get("level", "INFO"))
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_log_levels:
        raise ValueError(f"Invalid log level: {log_level}")
    logging = LoggingConfig(level=log_level)

    # Backward-compatibility env vars
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    environment = os.environ.get("ENVIRONMENT", "development")

    return Config(
        database=database,
        chroma=chroma,
        llm=llm,
        drift=drift,
        logging=logging,
        redis_url=redis_url,
        environment=environment,
    )


@lru_cache(maxsize=1)
def get_settings(path: Optional[Path] = None) -> Config:
    """
    Get cached configuration settings.

    Args:
        path: Optional path to config file. If None, uses default.

    Returns:
        Cached Config object.
    """
    return load_config(path)


# Backward-compatibility aliases
def get_config() -> Config:
    """Legacy alias for get_settings()."""
    return get_settings()


def reset_config() -> None:
    """Reset configuration cache (for testing)."""
    get_settings.cache_clear()
