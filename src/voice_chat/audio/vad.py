"""Voice Activity Detection using Silero VAD."""

import asyncio
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from voice_chat.config import get_settings


@dataclass
class VADResult:
    """Result of voice activity detection."""

    is_speech: bool
    confidence: float
    start_sample: int | None = None
    end_sample: int | None = None


class VoiceActivityDetector:
    """Silero VAD for detecting speech in audio."""

    def __init__(
        self,
        sample_rate: int | None = None,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
    ) -> None:
        """Initialize VAD.

        Args:
            sample_rate: Audio sample rate (8000 or 16000).
            threshold: Speech detection threshold (0-1).
            min_speech_duration_ms: Minimum speech duration to trigger.
            min_silence_duration_ms: Minimum silence to end speech.
        """
        settings = get_settings()
        self.sample_rate = sample_rate or settings.sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms

        # Silero VAD works with 8000 or 16000 Hz
        if self.sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD requires sample_rate of 8000 or 16000 Hz")

        # Load model lazily
        self._model = None
        self._utils = None

    def _get_model(self):
        """Load Silero VAD model."""
        if self._model is None:
            from silero_vad import load_silero_vad, get_speech_timestamps

            self._model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps
        return self._model

    def detect(self, audio: np.ndarray) -> bool:
        """Detect if audio contains speech.

        Args:
            audio: Audio data (float32, -1 to 1).

        Returns:
            True if speech detected.
        """
        model = self._get_model()

        # Ensure correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed
        if len(audio) > 0 and (audio.max() > 1.0 or audio.min() < -1.0):
            audio = audio / max(abs(audio.max()), abs(audio.min()))

        # Get speech timestamps
        import torch
        audio_tensor = torch.from_numpy(audio)

        timestamps = self._get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
        )

        return len(timestamps) > 0

    def get_speech_segments(
        self,
        audio: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Get speech segments from audio.

        Args:
            audio: Audio data.

        Returns:
            List of (start_sample, end_sample) tuples.
        """
        model = self._get_model()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if len(audio) > 0 and (audio.max() > 1.0 or audio.min() < -1.0):
            audio = audio / max(abs(audio.max()), abs(audio.min()))

        import torch
        audio_tensor = torch.from_numpy(audio)

        timestamps = self._get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
        )

        return [(ts["start"], ts["end"]) for ts in timestamps]

    def process_chunk(self, audio: np.ndarray) -> VADResult:
        """Process a single audio chunk.

        Args:
            audio: Audio chunk.

        Returns:
            VADResult with speech detection info.
        """
        is_speech = self.detect(audio)
        segments = self.get_speech_segments(audio) if is_speech else []

        return VADResult(
            is_speech=is_speech,
            confidence=1.0 if is_speech else 0.0,
            start_sample=segments[0][0] if segments else None,
            end_sample=segments[-1][1] if segments else None,
        )


class StreamingVAD:
    """Streaming VAD for real-time speech detection."""

    # Minimum audio duration (in seconds) to accumulate before running VAD
    MIN_PROCESS_DURATION = 0.3  # 300ms

    def __init__(
        self,
        sample_rate: int | None = None,
        threshold: float = 0.5,
        silence_threshold_seconds: float | None = None,
    ) -> None:
        """Initialize streaming VAD.

        Args:
            sample_rate: Audio sample rate.
            threshold: Speech detection threshold.
            silence_threshold_seconds: Seconds of silence to end speech.
        """
        settings = get_settings()
        self.sample_rate = sample_rate or settings.sample_rate
        self.threshold = threshold
        self.silence_threshold = silence_threshold_seconds or settings.silence_threshold_seconds

        self._vad = VoiceActivityDetector(
            sample_rate=self.sample_rate,
            threshold=threshold,
        )

        # State tracking
        self._is_speaking = False
        self._silence_duration = 0.0  # Track in seconds for clarity
        self._speech_samples = 0
        self._audio_buffer: list[np.ndarray] = []
        self._pending_chunks: list[np.ndarray] = []  # Buffer chunks before VAD processing

    def reset(self) -> None:
        """Reset VAD state."""
        self._is_speaking = False
        self._silence_duration = 0.0
        self._speech_samples = 0
        self._audio_buffer.clear()
        self._pending_chunks.clear()

    def process(self, audio: np.ndarray) -> tuple[bool, bool, np.ndarray | None]:
        """Process audio chunk and detect speech boundaries.

        Args:
            audio: Audio chunk.

        Returns:
            Tuple of:
                - speech_started: True if speech just started
                - speech_ended: True if speech just ended
                - complete_audio: If speech_ended, the complete speech audio
        """
        # Accumulate audio in pending buffer
        self._pending_chunks.append(audio.copy())

        # Calculate pending audio duration
        pending_samples = sum(len(chunk) for chunk in self._pending_chunks)
        pending_duration = pending_samples / self.sample_rate

        # Only run VAD when we have enough audio
        if pending_duration < self.MIN_PROCESS_DURATION:
            # Not enough audio yet, just buffer it
            if self._is_speaking:
                # If we're already speaking, still add to main buffer
                self._audio_buffer.append(audio.copy())
            return False, False, None

        # Combine pending chunks for VAD processing
        pending_audio = np.concatenate(self._pending_chunks)
        self._pending_chunks.clear()

        # Run VAD on the accumulated audio
        is_speech = self._vad.detect(pending_audio)

        speech_started = False
        speech_ended = False
        complete_audio = None

        if is_speech:
            self._silence_duration = 0.0
            self._speech_samples += len(pending_audio)

            if not self._is_speaking:
                self._is_speaking = True
                speech_started = True
                self._audio_buffer.clear()

            self._audio_buffer.append(pending_audio)

        else:
            if self._is_speaking:
                self._silence_duration += pending_duration
                self._audio_buffer.append(pending_audio)  # Include trailing silence

                if self._silence_duration >= self.silence_threshold:
                    # Speech ended
                    speech_ended = True
                    self._is_speaking = False
                    complete_audio = np.concatenate(self._audio_buffer)
                    self._audio_buffer.clear()
                    self._speech_samples = 0
                    self._silence_duration = 0.0

        return speech_started, speech_ended, complete_audio

    @property
    def is_speaking(self) -> bool:
        """Check if currently in speech."""
        return self._is_speaking

    @property
    def speech_duration(self) -> float:
        """Get current speech duration in seconds."""
        return self._speech_samples / self.sample_rate

    def get_current_audio(self) -> np.ndarray | None:
        """Get audio accumulated so far.

        Returns:
            Current audio buffer or None if empty.
        """
        if not self._audio_buffer:
            return None
        return np.concatenate(self._audio_buffer)
