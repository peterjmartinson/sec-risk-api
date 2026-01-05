"""
Authentication and rate limiting for SEC Risk API (Issue #25).

Provides API key management and configurable rate limiting to protect compute costs.
"""

import json
import logging
import secrets
from pathlib import Path
from typing import Dict, Optional, Any

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# API key header name
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Rate limiter instance
limiter = Limiter(key_func=get_remote_address)  # Will override with API key based limiting


class APIKeyManager:
    """
    Manages API keys with user association and rate limits.

    For MVP, uses in-memory storage. In production, replace with database.
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        """
        Initialize key manager.

        Args:
            storage_path: Optional path to JSON file for persistence.
        """
        self.storage_path = storage_path or Path("api_keys.json")
        self.keys: Dict[str, Dict[str, Any]] = {}

        # Load existing keys if file exists
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    self.keys = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load API keys from {self.storage_path}: {e}")

    def create_key(self, user: str, rate_limit: int = 10) -> str:
        """
        Create a new API key for a user.

        Args:
            user: Username or identifier.
            rate_limit: Requests per minute.

        Returns:
            The generated API key.
        """
        key = secrets.token_urlsafe(32)
        self.keys[key] = {"user": user, "rate_limit": rate_limit}
        self._save_keys()
        logger.info(f"Created API key for user: {user}")
        return key

    def delete_key(self, key: str) -> bool:
        """
        Delete an API key.

        Args:
            key: The API key to delete.

        Returns:
            True if deleted, False if not found.
        """
        if key in self.keys:
            user = self.keys[key]["user"]
            del self.keys[key]
            self._save_keys()
            logger.info(f"Deleted API key for user: {user}")
            return True
        return False

    def validate_key(self, key: str) -> Optional[str]:
        """
        Validate an API key and return the associated user.

        Args:
            key: The API key to validate.

        Returns:
            Username if valid, None otherwise.
        """
        return self.keys.get(key, {}).get("user")

    def get_rate_limit(self, key: str) -> Optional[int]:
        """
        Get the rate limit for an API key.

        Args:
            key: The API key.

        Returns:
            Rate limit if key exists, None otherwise.
        """
        return self.keys.get(key, {}).get("rate_limit")

    def _save_keys(self) -> None:
        """Save keys to storage file."""
        try:
            with open(self.storage_path, "w") as f:
                json.dump(self.keys, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")


# Global key manager instance
key_manager = APIKeyManager()


def authenticate_api_key(api_key: str) -> str:
    """
    Authenticate an API key.

    Args:
        api_key: The API key from request header.

    Returns:
        Associated username.

    Raises:
        HTTPException: If key is invalid.
    """
    if not api_key:
        logger.warning("API key required but not provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )

    user = key_manager.validate_key(api_key)
    if not user:
        logger.warning("Invalid API key provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    return user


def get_api_key_from_request(request: Request) -> Optional[str]:
    """
    Extract API key from request headers.

    Args:
        request: FastAPI request object.

    Returns:
        API key if present, None otherwise.
    """
    return request.headers.get("x-api-key")


def rate_limit_key_func(request: Request) -> str:
    """
    Rate limiting key function based on API key.

    Args:
        request: FastAPI request object.

    Returns:
        Key for rate limiting (API key or IP if no key).
    """
    api_key = get_api_key_from_request(request)
    if api_key:
        user = key_manager.validate_key(api_key)
        if user:
            # Use API key as rate limit key
            return api_key
    # Fallback to IP
    return get_remote_address(request)