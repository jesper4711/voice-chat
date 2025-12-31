"""Voice Assistant - Main orchestrator combining all components."""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import numpy as np

from voice_chat.agent.loop import AgentLoop
from voice_chat.audio.input import AudioInput
from voice_chat.audio.output import AudioOutput
from voice_chat.audio.stt import SpeechToText
from voice_chat.audio.tts import TextToSpeech
from voice_chat.audio.vad import StreamingVAD
from voice_chat.audio.wake_word import StreamingWakeWordDetector, WakeWordDetection
from voice_chat.config import get_settings


class AssistantState(Enum):
    """State of the voice assistant."""

    IDLE = auto()  # Waiting for wake word
    LISTENING = auto()  # Recording user speech
    PROCESSING = auto()  # Transcribing and thinking
    SPEAKING = auto()  # Speaking response
    INTERRUPTED = auto()  # User interrupted during response


@dataclass
class AssistantConfig:
    """Configuration for the assistant."""

    wake_words: list[str] | None = None
    enable_wake_word: bool = True
    enable_vad: bool = True
    auto_listen_after_response: bool = True
    follow_up_timeout_seconds: float = 5.0
    max_speech_duration_seconds: float = 30.0


class VoiceAssistant:
    """Main voice assistant orchestrating all components."""

    def __init__(
        self,
        config: AssistantConfig | None = None,
        on_state_change: Callable[[AssistantState], None] | None = None,
        on_transcription: Callable[[str], None] | None = None,
        on_response: Callable[[str], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """Initialize voice assistant.

        Args:
            config: Assistant configuration.
            on_state_change: Callback when state changes.
            on_transcription: Callback when user speech transcribed.
            on_response: Callback when assistant responds.
            on_error: Callback on error.
        """
        settings = get_settings()
        self.config = config or AssistantConfig()

        # Callbacks
        self.on_state_change = on_state_change
        self.on_transcription = on_transcription
        self.on_response = on_response
        self.on_error = on_error

        # State
        self._state = AssistantState.IDLE
        self._is_running = False
        self._last_interaction_time = 0.0

        # Components (initialized lazily)
        self._agent: AgentLoop | None = None
        self._audio_input: AudioInput | None = None
        self._audio_output: AudioOutput | None = None
        self._tts: TextToSpeech | None = None
        self._stt: SpeechToText | None = None
        self._vad: StreamingVAD | None = None
        self._wake_word: StreamingWakeWordDetector | None = None

    @property
    def state(self) -> AssistantState:
        """Get current state."""
        return self._state

    def _set_state(self, state: AssistantState) -> None:
        """Set state and notify callback."""
        if state != self._state:
            self._state = state
            if self.on_state_change:
                self.on_state_change(state)

    def _init_components(self) -> None:
        """Initialize all components."""
        settings = get_settings()

        # Agent
        self._agent = AgentLoop()

        # Audio I/O
        self._audio_input = AudioInput(
            sample_rate=settings.sample_rate,
            chunk_size=512,
        )
        self._audio_output = AudioOutput()

        # TTS/STT
        try:
            self._tts = TextToSpeech()
        except ValueError:
            self._tts = None  # No API key

        self._stt = SpeechToText()

        # VAD
        if self.config.enable_vad:
            self._vad = StreamingVAD(
                sample_rate=settings.sample_rate,
                silence_threshold_seconds=settings.silence_threshold_seconds,
            )

        # Wake word
        if self.config.enable_wake_word:
            try:
                self._wake_word = StreamingWakeWordDetector(
                    keywords=self.config.wake_words,
                    on_wake_word=self._on_wake_word,
                )
            except ValueError:
                self._wake_word = None  # No API key

    def _on_wake_word(self, detection: WakeWordDetection) -> None:
        """Handle wake word detection."""
        if self._state == AssistantState.IDLE:
            self._set_state(AssistantState.LISTENING)
            self._last_interaction_time = time.time()

    async def start(self) -> None:
        """Start the voice assistant."""
        if self._is_running:
            return

        self._init_components()
        self._is_running = True
        self._set_state(AssistantState.IDLE)

        # Start audio input
        if self._audio_input:
            self._audio_input.start()

        # Start wake word detection
        if self._wake_word:
            self._wake_word.start()

        # Main loop
        await self._run_loop()

    async def stop(self) -> None:
        """Stop the voice assistant."""
        self._is_running = False

        if self._audio_input:
            self._audio_input.stop()

        if self._audio_output:
            self._audio_output.stop()

        if self._wake_word:
            self._wake_word.stop()
            self._wake_word.cleanup()

        self._set_state(AssistantState.IDLE)

    async def _run_loop(self) -> None:
        """Main processing loop."""
        audio_buffer: list[np.ndarray] = []

        while self._is_running:
            try:
                # Read audio chunk
                audio = await self._audio_input.read_async(timeout=0.1)
                if audio is None:
                    continue

                # Process based on state
                if self._state == AssistantState.IDLE:
                    await self._process_idle(audio)

                elif self._state == AssistantState.LISTENING:
                    await self._process_listening(audio, audio_buffer)

                elif self._state == AssistantState.SPEAKING:
                    # Check for interruption
                    if self._vad and self._vad._vad.detect(audio):
                        self._audio_output.stop()
                        self._set_state(AssistantState.INTERRUPTED)
                        audio_buffer = [audio]

                elif self._state == AssistantState.INTERRUPTED:
                    # Collect interrupted speech
                    await self._process_listening(audio, audio_buffer)

            except Exception as e:
                if self.on_error:
                    self.on_error(e)
                await asyncio.sleep(0.1)

    async def _process_idle(self, audio: np.ndarray) -> None:
        """Process audio while idle (waiting for wake word)."""
        # Check for wake word
        if self._wake_word:
            detection = self._wake_word.process_chunk(audio)
            if detection:
                self._set_state(AssistantState.LISTENING)
                if self._vad:
                    self._vad.reset()
                return

        # Check for follow-up within timeout
        if self.config.auto_listen_after_response:
            elapsed = time.time() - self._last_interaction_time
            if elapsed < self.config.follow_up_timeout_seconds:
                # Treat as potential follow-up
                if self._vad and self._vad._vad.detect(audio):
                    self._set_state(AssistantState.LISTENING)
                    if self._vad:
                        self._vad.reset()

    async def _process_listening(
        self, audio: np.ndarray, audio_buffer: list[np.ndarray]
    ) -> None:
        """Process audio while listening for user speech."""
        audio_buffer.append(audio)

        # Check speech duration limit
        total_samples = sum(len(chunk) for chunk in audio_buffer)
        duration = total_samples / get_settings().sample_rate

        if duration > self.config.max_speech_duration_seconds:
            # Max duration reached, process what we have
            await self._process_speech(audio_buffer)
            audio_buffer.clear()
            return

        # Use VAD to detect end of speech
        if self._vad:
            speech_started, speech_ended, complete_audio = self._vad.process(audio)

            if speech_ended and complete_audio is not None:
                # Use VAD's complete audio instead of buffer
                await self._process_speech_audio(complete_audio)
                audio_buffer.clear()
                return

    async def _process_speech(self, audio_buffer: list[np.ndarray]) -> None:
        """Process collected speech audio."""
        if not audio_buffer:
            self._set_state(AssistantState.IDLE)
            return

        audio = np.concatenate(audio_buffer)
        await self._process_speech_audio(audio)

    async def _process_speech_audio(self, audio: np.ndarray) -> None:
        """Process speech audio through STT and agent."""
        self._set_state(AssistantState.PROCESSING)

        try:
            # Transcribe
            result = await self._stt.transcribe_async(audio)
            text = result.text.strip()

            if not text:
                self._set_state(AssistantState.IDLE)
                return

            if self.on_transcription:
                self.on_transcription(text)

            # Get agent response
            response = await self._agent.run(text)

            if self.on_response:
                self.on_response(response)

            # Speak response
            if self._tts and self._audio_output:
                self._set_state(AssistantState.SPEAKING)
                audio_data = await self._tts.synthesize(response)
                await self._audio_output.play_mp3(audio_data)

            self._last_interaction_time = time.time()
            self._set_state(AssistantState.IDLE)

        except Exception as e:
            if self.on_error:
                self.on_error(e)
            self._set_state(AssistantState.IDLE)

    async def process_text(self, text: str) -> str:
        """Process text input directly (for testing or text mode).

        Args:
            text: User input text.

        Returns:
            Assistant response.
        """
        if self._agent is None:
            self._agent = AgentLoop()

        self._set_state(AssistantState.PROCESSING)

        try:
            response = await self._agent.run(text)
            self._last_interaction_time = time.time()
            return response
        finally:
            self._set_state(AssistantState.IDLE)

    def interrupt(self) -> None:
        """Interrupt current speech."""
        if self._state == AssistantState.SPEAKING:
            if self._audio_output:
                self._audio_output.stop()
            self._set_state(AssistantState.INTERRUPTED)

    def clear_history(self) -> None:
        """Clear conversation history."""
        if self._agent:
            self._agent.clear_history()
