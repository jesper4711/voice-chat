"""Configuration management for Voice Chat."""

from enum import Enum
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


class LanguageMode(str, Enum):
    """Language mode for the assistant."""

    ENGLISH = "en"
    SWEDISH = "sv"
    AUTO = "auto"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    elevenlabs_api_key: str = Field(default="", description="ElevenLabs API key")
    picovoice_access_key: str = Field(default="", description="Picovoice access key")
    tavily_api_key: str = Field(default="", description="Tavily API key for web search")

    # Language
    language_mode: LanguageMode = Field(
        default=LanguageMode.AUTO,
        description="Language mode: 'en', 'sv', or 'auto'",
    )

    # LLM Settings
    gemini_model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini model to use",
    )
    max_agent_iterations: int = Field(
        default=10,
        description="Maximum iterations for agent loop",
    )

    # TTS Settings
    elevenlabs_model: str = Field(
        default="eleven_multilingual_v2",
        description="ElevenLabs model to use",
    )
    elevenlabs_voice_id: str = Field(
        default="21m00Tcm4TlvDq8ikWAM",  # Rachel - default voice
        description="ElevenLabs voice ID",
    )

    # STT Settings
    whisper_model: str = Field(
        default="base",
        description="Whisper model size: tiny, base, small, medium, large-v3",
    )

    # Audio Settings
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    silence_threshold_seconds: float = Field(
        default=1.5,
        description="Seconds of silence to detect end of speech",
    )

    # Memory Settings
    max_conversation_turns: int = Field(
        default=20,
        description="Maximum conversation turns to keep in memory",
    )
    inactivity_timeout_seconds: float = Field(
        default=1800,
        description="Seconds of inactivity before clearing conversation history (default: 30 min)",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    return Settings()
