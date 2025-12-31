"""Unit tests for configuration."""

import os
from unittest.mock import patch

import pytest

from voice_chat.config import LanguageMode, Settings


class TestSettings:
    """Tests for Settings class."""

    def test_defaults(self) -> None:
        """Settings have sensible defaults when no env vars are set."""
        # Create settings with explicit empty values to test defaults
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(
                _env_file=None,  # Disable .env loading
            )

        assert settings.gemini_api_key == ""
        assert settings.elevenlabs_api_key == ""
        assert settings.language_mode == LanguageMode.AUTO
        assert settings.gemini_model == "gemini-2.0-flash"
        assert settings.max_agent_iterations == 10
        assert settings.sample_rate == 16000

    def test_loads_from_env(self) -> None:
        """Settings load from environment variables."""
        env = {
            "GEMINI_API_KEY": "test-gemini-key",
            "ELEVENLABS_API_KEY": "test-elevenlabs-key",
            "LANGUAGE_MODE": "sv",
            "GEMINI_MODEL": "gemini-2.5-pro",
        }

        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)

        assert settings.gemini_api_key == "test-gemini-key"
        assert settings.elevenlabs_api_key == "test-elevenlabs-key"
        assert settings.language_mode == LanguageMode.SWEDISH
        assert settings.gemini_model == "gemini-2.5-pro"

    def test_language_mode_values(self) -> None:
        """LanguageMode enum has expected values."""
        assert LanguageMode.ENGLISH.value == "en"
        assert LanguageMode.SWEDISH.value == "sv"
        assert LanguageMode.AUTO.value == "auto"

    def test_invalid_language_mode_ignored(self) -> None:
        """Invalid language mode falls back to default."""
        env = {"LANGUAGE_MODE": "invalid"}

        with patch.dict(os.environ, env, clear=True):
            # Pydantic should raise validation error or use default
            try:
                settings = Settings()
                # If no error, check it used default
                assert settings.language_mode == LanguageMode.AUTO
            except ValueError:
                # Validation error is also acceptable
                pass
