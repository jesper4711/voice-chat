"""Wake word detection using Picovoice Porcupine."""

import asyncio
import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import pvporcupine

from voice_chat.config import get_settings


class BuiltinWakeWord(str, Enum):
    """Built-in Porcupine wake words."""

    ALEXA = "alexa"
    AMERICANO = "americano"
    BLUEBERRY = "blueberry"
    BUMBLEBEE = "bumblebee"
    COMPUTER = "computer"
    GRAPEFRUIT = "grapefruit"
    GRASSHOPPER = "grasshopper"
    HEY_GOOGLE = "hey google"
    HEY_SIRI = "hey siri"
    JARVIS = "jarvis"
    OK_GOOGLE = "ok google"
    PICOVOICE = "picovoice"
    PORCUPINE = "porcupine"
    TERMINATOR = "terminator"


@dataclass
class WakeWordDetection:
    """Wake word detection event."""

    keyword: str
    keyword_index: int
    timestamp: float


class WakeWordDetector:
    """Porcupine wake word detector."""

    def __init__(
        self,
        access_key: str | None = None,
        keywords: list[str] | None = None,
        keyword_paths: list[str | Path] | None = None,
        sensitivities: list[float] | None = None,
        on_wake_word: Callable[[WakeWordDetection], None] | None = None,
    ) -> None:
        """Initialize wake word detector.

        Args:
            access_key: Picovoice access key. If None, uses settings.
            keywords: Built-in keywords to detect (e.g., ["jarvis", "computer"]).
            keyword_paths: Paths to custom .ppn files.
            sensitivities: Sensitivity for each keyword (0.0 to 1.0).
            on_wake_word: Callback when wake word detected.
        """
        settings = get_settings()
        self.access_key = access_key or settings.picovoice_access_key

        if not self.access_key:
            raise ValueError("PICOVOICE_ACCESS_KEY not set")

        # Default to "jarvis" if no keywords specified
        self.keywords = keywords or ["jarvis"]
        self.keyword_paths = [str(p) for p in keyword_paths] if keyword_paths else None
        self.sensitivities = sensitivities or [0.5] * len(self.keywords)
        self.on_wake_word = on_wake_word

        # Create Porcupine instance
        self._porcupine: pvporcupine.Porcupine | None = None
        self._is_running = False

    def _get_porcupine(self) -> pvporcupine.Porcupine:
        """Get or create Porcupine instance."""
        if self._porcupine is None:
            if self.keyword_paths:
                self._porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keyword_paths=self.keyword_paths,
                    sensitivities=self.sensitivities,
                )
            else:
                self._porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keywords=self.keywords,
                    sensitivities=self.sensitivities,
                )
        return self._porcupine

    @property
    def sample_rate(self) -> int:
        """Get required sample rate."""
        return self._get_porcupine().sample_rate

    @property
    def frame_length(self) -> int:
        """Get required frame length."""
        return self._get_porcupine().frame_length

    def process(self, audio: np.ndarray) -> int:
        """Process audio frame and detect wake word.

        Args:
            audio: Audio frame (int16, length must match frame_length).

        Returns:
            Keyword index if detected, -1 otherwise.
        """
        porcupine = self._get_porcupine()

        # Convert float32 to int16 if needed
        if audio.dtype == np.float32:
            audio = (audio * 32767).astype(np.int16)
        elif audio.dtype != np.int16:
            audio = audio.astype(np.int16)

        # Ensure correct length
        if len(audio) != porcupine.frame_length:
            # Pad or truncate
            if len(audio) < porcupine.frame_length:
                audio = np.pad(audio, (0, porcupine.frame_length - len(audio)))
            else:
                audio = audio[: porcupine.frame_length]

        result = porcupine.process(audio)
        return result

    def detect(self, audio: np.ndarray) -> WakeWordDetection | None:
        """Detect wake word in audio.

        Args:
            audio: Audio data.

        Returns:
            WakeWordDetection if detected, None otherwise.
        """
        import time

        porcupine = self._get_porcupine()
        frame_length = porcupine.frame_length

        # Convert to int16 if needed
        if audio.dtype == np.float32:
            audio_int16 = (audio * 32767).astype(np.int16)
        else:
            audio_int16 = audio.astype(np.int16)

        # Process in frames
        for i in range(0, len(audio_int16) - frame_length + 1, frame_length):
            frame = audio_int16[i : i + frame_length]
            result = porcupine.process(frame)

            if result >= 0:
                keyword = self.keywords[result] if result < len(self.keywords) else f"custom_{result}"
                detection = WakeWordDetection(
                    keyword=keyword,
                    keyword_index=result,
                    timestamp=time.time(),
                )

                if self.on_wake_word:
                    self.on_wake_word(detection)

                return detection

        return None

    def cleanup(self) -> None:
        """Clean up Porcupine resources."""
        if hasattr(self, "_porcupine") and self._porcupine is not None:
            self._porcupine.delete()
            self._porcupine = None

    def __del__(self) -> None:
        """Destructor to clean up resources."""
        self.cleanup()


class StreamingWakeWordDetector:
    """Streaming wake word detector for real-time audio."""

    def __init__(
        self,
        access_key: str | None = None,
        keywords: list[str] | None = None,
        on_wake_word: Callable[[WakeWordDetection], None] | None = None,
    ) -> None:
        """Initialize streaming detector.

        Args:
            access_key: Picovoice access key.
            keywords: Keywords to detect.
            on_wake_word: Callback for wake word detection.
        """
        self._detector = WakeWordDetector(
            access_key=access_key,
            keywords=keywords,
            on_wake_word=on_wake_word,
        )
        self._buffer = np.array([], dtype=np.int16)
        self._is_running = False

    @property
    def sample_rate(self) -> int:
        """Get required sample rate."""
        return self._detector.sample_rate

    @property
    def frame_length(self) -> int:
        """Get required frame length."""
        return self._detector.frame_length

    def start(self) -> None:
        """Start detection."""
        self._is_running = True
        self._buffer = np.array([], dtype=np.int16)

    def stop(self) -> None:
        """Stop detection."""
        self._is_running = False

    def process_chunk(self, audio: np.ndarray) -> WakeWordDetection | None:
        """Process audio chunk.

        Args:
            audio: Audio chunk (can be any length).

        Returns:
            WakeWordDetection if detected.
        """
        if not self._is_running:
            return None

        # Convert to int16
        if audio.dtype == np.float32:
            audio_int16 = (audio * 32767).astype(np.int16)
        else:
            audio_int16 = audio.astype(np.int16)

        # Add to buffer
        self._buffer = np.concatenate([self._buffer, audio_int16])

        # Process complete frames
        frame_length = self._detector.frame_length

        while len(self._buffer) >= frame_length:
            frame = self._buffer[:frame_length]
            self._buffer = self._buffer[frame_length:]

            result = self._detector.process(frame)
            if result >= 0:
                import time

                keywords = self._detector.keywords
                keyword = keywords[result] if result < len(keywords) else f"custom_{result}"
                detection = WakeWordDetection(
                    keyword=keyword,
                    keyword_index=result,
                    timestamp=time.time(),
                )

                if self._detector.on_wake_word:
                    self._detector.on_wake_word(detection)

                return detection

        return None

    def cleanup(self) -> None:
        """Clean up resources."""
        self._detector.cleanup()
