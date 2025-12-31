"""Text-to-Speech using ElevenLabs API."""

import asyncio
import io
from dataclasses import dataclass
from typing import AsyncIterator

from elevenlabs import AsyncElevenLabs, VoiceSettings

from voice_chat.config import get_settings


@dataclass
class TTSConfig:
    """Configuration for TTS."""

    voice_id: str
    model: str
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True


class TextToSpeech:
    """ElevenLabs TTS client."""

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize TTS client.

        Args:
            api_key: ElevenLabs API key. If None, uses settings.
            voice_id: Voice ID to use. If None, uses settings.
            model: Model to use. If None, uses settings.
        """
        settings = get_settings()
        self.api_key = api_key or settings.elevenlabs_api_key
        self.voice_id = voice_id or settings.elevenlabs_voice_id
        self.model = model or settings.elevenlabs_model

        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not set")

        self.client = AsyncElevenLabs(api_key=self.api_key)

        self.config = TTSConfig(
            voice_id=self.voice_id,
            model=self.model,
        )

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.

        Returns:
            Audio data as bytes (MP3 format).
        """
        if not text.strip():
            return b""

        audio_chunks: list[bytes] = []

        async for chunk in self.client.text_to_speech.convert(
            voice_id=self.config.voice_id,
            text=text,
            model_id=self.config.model,
            voice_settings=VoiceSettings(
                stability=self.config.stability,
                similarity_boost=self.config.similarity_boost,
                style=self.config.style,
                use_speaker_boost=self.config.use_speaker_boost,
            ),
        ):
            audio_chunks.append(chunk)

        return b"".join(audio_chunks)

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream synthesized audio chunks.

        Args:
            text: Text to synthesize.

        Yields:
            Audio chunks as bytes.
        """
        if not text.strip():
            return

        async for chunk in self.client.text_to_speech.convert(
            voice_id=self.config.voice_id,
            text=text,
            model_id=self.config.model,
            voice_settings=VoiceSettings(
                stability=self.config.stability,
                similarity_boost=self.config.similarity_boost,
                style=self.config.style,
                use_speaker_boost=self.config.use_speaker_boost,
            ),
        ):
            yield chunk

    async def get_voices(self) -> list[dict]:
        """Get available voices.

        Returns:
            List of voice dictionaries with id, name, and labels.
        """
        response = await self.client.voices.get_all()
        return [
            {
                "id": voice.voice_id,
                "name": voice.name,
                "labels": voice.labels,
            }
            for voice in response.voices
        ]


async def speak(text: str) -> bytes:
    """Simple helper for one-off TTS.

    Args:
        text: Text to speak.

    Returns:
        Audio data as bytes.
    """
    tts = TextToSpeech()
    return await tts.synthesize(text)
