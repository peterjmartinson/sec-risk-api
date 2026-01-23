# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Tests for config loader.

Tests YAML loading, environment overrides, and validation.
"""

import os
from pathlib import Path

import pytest

from sigmak.config import get_settings, load_config, reset_config


def test_load_config_from_yaml(tmp_path: Path) -> None:
    """Test loading config from a YAML file."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
database:
  sqlite_path: "test.db"

chroma:
  persist_directory: "test_chroma"
  embedding_model: "test-model"
  llm_cache_similarity_threshold: 0.75

llm:
  model: "test-llm"
  temperature: 0.5

drift:
  review_cron: "0 4 * * *"
  sample_size: 50
  low_confidence_threshold: 0.5
  drift_threshold: 0.3

logging:
  level: "DEBUG"
"""
    )

    config = load_config(config_file)

    assert config.database.sqlite_path == Path("test.db")
    assert config.chroma.persist_directory == Path("test_chroma")
    assert config.chroma.embedding_model == "test-model"
    assert config.chroma.llm_cache_similarity_threshold == 0.75
    assert config.llm.model == "test-llm"
    assert config.llm.temperature == 0.5
    assert config.drift.review_cron == "0 4 * * *"
    assert config.drift.sample_size == 50
    assert config.logging.level == "DEBUG"


def test_env_override_sqlite_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test SIGMAK_SQLITE_PATH environment variable override."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
database:
  sqlite_path: "default.db"

chroma:
  persist_directory: "test_chroma"
  embedding_model: "test-model"

llm:
  model: "test-llm"
  temperature: 0.0

drift:
  review_cron: "0 3 * * *"
  sample_size: 100
  low_confidence_threshold: 0.6
  drift_threshold: 0.2

logging:
  level: "INFO"
"""
    )

    override_path = str(tmp_path / "override.db")
    monkeypatch.setenv("SIGMAK_SQLITE_PATH", override_path)

    config = load_config(config_file)
    assert config.database.sqlite_path == Path(override_path)


def test_env_override_llm_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test SIGMAK_LLM_MODEL environment variable override."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
database:
  sqlite_path: "test.db"

chroma:
  persist_directory: "test_chroma"
  embedding_model: "test-model"

llm:
  model: "default-model"
  temperature: 0.0

drift:
  review_cron: "0 3 * * *"
  sample_size: 100
  low_confidence_threshold: 0.6
  drift_threshold: 0.2

logging:
  level: "INFO"
"""
    )

    monkeypatch.setenv("SIGMAK_LLM_MODEL", "override-model")

    config = load_config(config_file)
    assert config.llm.model == "override-model"


def test_env_override_embedding_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test SIGMAK_EMBEDDING_MODEL environment variable override."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
database:
  sqlite_path: "test.db"

chroma:
  persist_directory: "test_chroma"
  embedding_model: "default-embedding"

llm:
  model: "test-model"
  temperature: 0.0

drift:
  review_cron: "0 3 * * *"
  sample_size: 100
  low_confidence_threshold: 0.6
  drift_threshold: 0.2

logging:
  level: "INFO"
"""
    )

    monkeypatch.setenv("SIGMAK_EMBEDDING_MODEL", "override-embedding")

    config = load_config(config_file)
    assert config.chroma.embedding_model == "override-embedding"


def test_get_settings_is_cached(tmp_path: Path) -> None:
    """Test that get_settings() caches the result."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
database:
  sqlite_path: "test.db"

chroma:
  persist_directory: "test_chroma"
  embedding_model: "test-model"

llm:
  model: "test-llm"
  temperature: 0.0

drift:
  review_cron: "0 3 * * *"
  sample_size: 100
  low_confidence_threshold: 0.6
  drift_threshold: 0.2

logging:
  level: "INFO"
"""
    )

    # Clear cache first
    reset_config()

    config1 = get_settings(config_file)
    config2 = get_settings(config_file)

    # Should be the same object (cached)
    assert config1 is config2


def test_missing_config_file() -> None:
    """Test that missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(Path("/nonexistent/config.yaml"))


def test_invalid_yaml(tmp_path: Path) -> None:
    """Test that invalid YAML raises an error."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("not: a: valid: yaml:")

    with pytest.raises(Exception):  # yaml.YAMLError or similar
        load_config(config_file)


def test_missing_required_field(tmp_path: Path) -> None:
    """Test that missing required fields raise ValueError."""
    config_file = tmp_path / "incomplete.yaml"
    config_file.write_text(
        """
database:
  sqlite_path: "test.db"

chroma:
  persist_directory: "test_chroma"
  # Missing embedding_model

llm:
  model: "test-llm"
  temperature: 0.0

drift:
  review_cron: "0 3 * * *"
  sample_size: 100
  low_confidence_threshold: 0.6
  drift_threshold: 0.2

logging:
  level: "INFO"
"""
    )

    with pytest.raises(ValueError, match="embedding_model"):
        load_config(config_file)


def test_backward_compatibility_aliases(tmp_path: Path) -> None:
    """Test that legacy properties still work."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
database:
  sqlite_path: "test.db"

chroma:
  persist_directory: "test_chroma"
  embedding_model: "test-model"

llm:
  model: "test-llm"
  temperature: 0.0

drift:
  review_cron: "0 3 * * *"
  sample_size: 100
  low_confidence_threshold: 0.6
  drift_threshold: 0.2

logging:
  level: "WARNING"
"""
    )

    config = load_config(config_file)

    # Test backward-compatible properties
    assert config.log_level == "WARNING"
    assert config.chroma_persist_path == "test_chroma"
    assert config.redis_url == "redis://localhost:6379/0"
    assert config.environment == "development"
