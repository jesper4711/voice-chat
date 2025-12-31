"""Audio output and playback."""

import asyncio
import io
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np
import sounddevice as sd

from voice_chat.config import get_settings


@dataclass
class PlaybackState:
    """State of audio playback."""

    is_playing: bool = False
    is_interrupted: bool = False
    current_position: int = 0


class AudioOutput:
    """Audio output handler for playing synthesized speech."""

    def __init__(
        self,
        sample_rate: int | None = None,
        on_playback_start: Callable[[], None] | None = None,
        on_playback_end: Callable[[], None] | None = None,
    ) -> None:
        """Initialize audio output.

        Args:
            sample_rate: Output sample rate. If None, uses settings.
            on_playback_start: Callback when playback starts.
            on_playback_end: Callback when playback ends.
        """
        settings = get_settings()
        self.sample_rate = sample_rate or 24000  # ElevenLabs default
        self.on_playback_start = on_playback_start
        self.on_playback_end = on_playback_end
        self.state = PlaybackState()
        self._stop_event = threading.Event()
        self._current_stream: sd.OutputStream | None = None

    async def play_mp3(self, audio_data: bytes) -> None:
        """Play MP3 audio data.

        Args:
            audio_data: MP3 audio bytes.
        """
        if not audio_data:
            return

        # Decode MP3 to PCM
        pcm_data = await self._decode_mp3(audio_data)
        if pcm_data is None:
            return

        await self.play_pcm(pcm_data)

    async def play_pcm(self, audio_data: np.ndarray) -> None:
        """Play PCM audio data.

        Args:
            audio_data: NumPy array of audio samples.
        """
        if audio_data is None or len(audio_data) == 0:
            return

        self.state.is_playing = True
        self.state.is_interrupted = False
        self._stop_event.clear()

        if self.on_playback_start:
            self.on_playback_start()

        try:
            # Play audio in a thread to not block
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._play_blocking,
                audio_data,
            )
        finally:
            self.state.is_playing = False
            if self.on_playback_end:
                self.on_playback_end()

    def _play_blocking(self, audio_data: np.ndarray) -> None:
        """Blocking audio playback (runs in thread)."""
        try:
            # Ensure audio is float32 and in range [-1, 1]
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))

            # Play with sounddevice
            sd.play(audio_data, samplerate=self.sample_rate)

            # Wait for playback to finish or be interrupted
            while sd.get_stream().active:
                if self._stop_event.is_set():
                    sd.stop()
                    self.state.is_interrupted = True
                    break
                self._stop_event.wait(timeout=0.05)

            sd.wait()  # Ensure playback is complete
        except Exception as e:
            print(f"Audio playback error: {e}")

    def stop(self) -> None:
        """Stop current playback."""
        self._stop_event.set()
        sd.stop()

    async def _decode_mp3(self, mp3_data: bytes) -> np.ndarray | None:
        """Decode MP3 to PCM audio.

        Args:
            mp3_data: MP3 audio bytes.

        Returns:
            NumPy array of audio samples, or None on error.
        """
        try:
            import av

            # Use PyAV to decode MP3
            container = av.open(io.BytesIO(mp3_data))
            audio_stream = container.streams.audio[0]

            frames = []
            for frame in container.decode(audio_stream):
                # Convert to numpy array
                array = frame.to_ndarray()
                # Convert from int16 to float32
                if array.dtype == np.int16:
                    array = array.astype(np.float32) / 32768.0
                frames.append(array)

            container.close()

            if not frames:
                return None

            # Concatenate all frames
            audio = np.concatenate(frames, axis=1 if len(frames[0].shape) > 1 else 0)

            # Convert stereo to mono if needed
            if len(audio.shape) > 1 and audio.shape[0] == 2:
                audio = audio.mean(axis=0)

            # Flatten if needed
            audio = audio.flatten()

            # Update sample rate from the actual audio
            self.sample_rate = audio_stream.rate

            return audio

        except ImportError:
            print("PyAV not available for MP3 decoding")
            return None
        except Exception as e:
            print(f"Error decoding MP3: {e}")
            return None


class StreamingAudioOutput:
    """Streaming audio output for real-time playback."""

    def __init__(self, sample_rate: int = 24000) -> None:
        """Initialize streaming output.

        Args:
            sample_rate: Output sample rate.
        """
        self.sample_rate = sample_rate
        self._stream: sd.OutputStream | None = None
        self._buffer: list[np.ndarray] = []
        self._lock = threading.Lock()
        self.is_playing = False

    def start(self) -> None:
        """Start the audio stream."""
        if self._stream is not None:
            return

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self._audio_callback,
        )
        self._stream.start()
        self.is_playing = True

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Audio callback for streaming playback."""
        with self._lock:
            if self._buffer:
                chunk = self._buffer.pop(0)
                if len(chunk) >= frames:
                    outdata[:, 0] = chunk[:frames]
                    if len(chunk) > frames:
                        self._buffer.insert(0, chunk[frames:])
                else:
                    outdata[: len(chunk), 0] = chunk
                    outdata[len(chunk) :, 0] = 0
            else:
                outdata.fill(0)

    def write(self, audio_data: np.ndarray) -> None:
        """Write audio data to the buffer.

        Args:
            audio_data: Audio samples to play.
        """
        with self._lock:
            self._buffer.append(audio_data)

    def stop(self) -> None:
        """Stop the audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.is_playing = False
        with self._lock:
            self._buffer.clear()
