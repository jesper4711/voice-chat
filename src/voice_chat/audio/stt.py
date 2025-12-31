"""Speech-to-Text using Faster-Whisper."""

import asyncio
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from faster_whisper import WhisperModel

from voice_chat.config import LanguageMode, get_settings


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text."""

    text: str
    start: float
    end: float
    confidence: float
    language: str | None = None


@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription."""

    text: str
    segments: list[TranscriptionSegment]
    language: str
    language_probability: float
    duration: float


class SpeechToText:
    """Faster-Whisper STT client."""

    def __init__(
        self,
        model_size: str | None = None,
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        """Initialize STT.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3).
            device: Device to use ('cpu', 'cuda', 'auto').
            compute_type: Compute type ('int8', 'float16', 'float32', 'auto').
        """
        settings = get_settings()
        self.model_size = model_size or settings.whisper_model

        # Determine device and compute type
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if compute_type == "auto":
            compute_type = "int8" if device == "cpu" else "float16"

        self.device = device
        self.compute_type = compute_type

        # Load model lazily
        self._model: WhisperModel | None = None

    def _get_model(self) -> WhisperModel:
        """Get or create the Whisper model."""
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (float32, -1 to 1).
            language: Language code (e.g., 'en', 'sv'). None for auto-detect.
            sample_rate: Sample rate of audio.

        Returns:
            TranscriptionResult with text and segments.
        """
        settings = get_settings()

        # Determine language based on settings
        if language is None:
            if settings.language_mode == LanguageMode.ENGLISH:
                language = "en"
            elif settings.language_mode == LanguageMode.SWEDISH:
                language = "sv"
            # AUTO mode: let Whisper detect

        model = self._get_model()

        # Ensure audio is the right format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))

        # Run transcription
        segments_iter, info = model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        # Collect segments
        segments: list[TranscriptionSegment] = []
        full_text_parts: list[str] = []

        for segment in segments_iter:
            segments.append(
                TranscriptionSegment(
                    text=segment.text.strip(),
                    start=segment.start,
                    end=segment.end,
                    confidence=segment.avg_logprob,
                    language=info.language,
                )
            )
            full_text_parts.append(segment.text.strip())

        full_text = " ".join(full_text_parts)

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
        )

    async def transcribe_async(
        self,
        audio: np.ndarray,
        language: str | None = None,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """Async version of transcribe.

        Args:
            audio: Audio data.
            language: Language code or None for auto.
            sample_rate: Sample rate.

        Returns:
            TranscriptionResult.
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self.transcribe, audio, language, sample_rate
        )


async def transcribe(audio: np.ndarray, language: str | None = None) -> str:
    """Simple helper for one-off transcription.

    Args:
        audio: Audio data.
        language: Language code or None for auto.

    Returns:
        Transcribed text.
    """
    stt = SpeechToText()
    result = await stt.transcribe_async(audio, language)
    return result.text
