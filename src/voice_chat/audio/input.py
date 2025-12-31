"""Audio input handling for microphone capture."""

import asyncio
import queue
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np
import sounddevice as sd

from voice_chat.config import get_settings


@dataclass
class AudioChunk:
    """A chunk of audio data."""

    data: np.ndarray
    sample_rate: int
    timestamp: float


class AudioInput:
    """Microphone input handler."""

    def __init__(
        self,
        sample_rate: int | None = None,
        chunk_size: int = 512,
        channels: int = 1,
        on_audio: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        """Initialize audio input.

        Args:
            sample_rate: Sample rate in Hz. If None, uses settings.
            chunk_size: Number of samples per chunk.
            channels: Number of audio channels (1 = mono).
            on_audio: Callback for each audio chunk.
        """
        settings = get_settings()
        self.sample_rate = sample_rate or settings.sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.on_audio = on_audio

        self._stream: sd.InputStream | None = None
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._is_running = False
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start capturing audio from microphone."""
        if self._is_running:
            return

        self._stop_event.clear()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._is_running = True

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for audio input."""
        if status:
            pass  # Could log status flags

        # Copy data and add to queue
        audio = indata.copy().flatten()
        self._audio_queue.put(audio)

        if self.on_audio:
            self.on_audio(audio)

    def stop(self) -> None:
        """Stop audio capture."""
        self._stop_event.set()
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._is_running = False

    def read(self, timeout: float | None = None) -> np.ndarray | None:
        """Read a chunk of audio.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Audio data or None if timeout.
        """
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_all(self) -> np.ndarray:
        """Read all available audio chunks.

        Returns:
            Concatenated audio data.
        """
        chunks = []
        while not self._audio_queue.empty():
            try:
                chunks.append(self._audio_queue.get_nowait())
            except queue.Empty:
                break

        if not chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(chunks)

    def clear(self) -> None:
        """Clear the audio queue."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    @property
    def is_running(self) -> bool:
        """Check if audio capture is running."""
        return self._is_running

    async def read_async(self, timeout: float | None = None) -> np.ndarray | None:
        """Async version of read.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Audio data or None if timeout.
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self.read, timeout
        )


class BufferedAudioInput:
    """Buffered audio input that accumulates audio until stopped."""

    def __init__(
        self,
        sample_rate: int | None = None,
        chunk_size: int = 512,
    ) -> None:
        """Initialize buffered audio input.

        Args:
            sample_rate: Sample rate in Hz.
            chunk_size: Samples per chunk.
        """
        settings = get_settings()
        self.sample_rate = sample_rate or settings.sample_rate
        self.chunk_size = chunk_size

        self._buffer: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._input = AudioInput(
            sample_rate=self.sample_rate,
            chunk_size=chunk_size,
            on_audio=self._on_audio,
        )

    def _on_audio(self, audio: np.ndarray) -> None:
        """Handle incoming audio."""
        with self._lock:
            self._buffer.append(audio.copy())

    def start(self) -> None:
        """Start recording."""
        self.clear()
        self._input.start()

    def stop(self) -> np.ndarray:
        """Stop recording and return all audio.

        Returns:
            Complete audio recording.
        """
        self._input.stop()
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            return np.concatenate(self._buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()

    def get_buffer(self) -> np.ndarray:
        """Get current buffer without stopping.

        Returns:
            Current audio buffer.
        """
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            return np.concatenate(self._buffer)

    @property
    def is_running(self) -> bool:
        """Check if recording."""
        return self._input.is_running

    @property
    def duration(self) -> float:
        """Get current recording duration in seconds."""
        with self._lock:
            total_samples = sum(len(chunk) for chunk in self._buffer)
            return total_samples / self.sample_rate
