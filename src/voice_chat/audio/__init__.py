"""Audio module for Voice Chat - input, output, STT, TTS, VAD, wake word."""

from voice_chat.audio.input import AudioInput, BufferedAudioInput
from voice_chat.audio.output import AudioOutput, StreamingAudioOutput
from voice_chat.audio.stt import SpeechToText, transcribe
from voice_chat.audio.tts import TextToSpeech, speak
from voice_chat.audio.vad import StreamingVAD, VoiceActivityDetector
from voice_chat.audio.wake_word import (
    StreamingWakeWordDetector,
    WakeWordDetection,
    WakeWordDetector,
)

__all__ = [
    # Input
    "AudioInput",
    "BufferedAudioInput",
    # Output
    "AudioOutput",
    "StreamingAudioOutput",
    # TTS
    "TextToSpeech",
    "speak",
    # STT
    "SpeechToText",
    "transcribe",
    # VAD
    "VoiceActivityDetector",
    "StreamingVAD",
    # Wake word
    "WakeWordDetector",
    "StreamingWakeWordDetector",
    "WakeWordDetection",
]
