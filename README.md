# Voice Chat

A voice-activated AI assistant with wake word detection and extensible tool system.

## Features

- Voice-activated with Porcupine wake word detection
- Speech-to-text with Faster-Whisper
- Text-to-speech with ElevenLabs
- Gemini-powered AI agent with tool calling
- Extensible tool registry system
- Bilingual support (English/Swedish)

## Installation

```bash
uv sync --all-extras
```

## Usage

```bash
# Text chat mode
voice-chat chat

# With voice output (TTS)
voice-chat chat --speak

# Single command
voice-chat run "What time is it?"

# Full voice mode with wake word ("Jarvis")
voice-chat voice

# Voice mode without wake word (always listening)
voice-chat voice --no-wake-word

# Voice mode with custom wake word
voice-chat voice --wake-word computer
```

## Conversation Memory

The assistant maintains conversation context with automatic memory management:

- **Sliding window**: Keeps the last 20 conversation turns (user + assistant pairs)
- **Inactivity timeout**: Clears history after 30 minutes of silence

Configure in `.env`:
```bash
MAX_CONVERSATION_TURNS=20        # Increase for longer memory, 0 = unlimited
INACTIVITY_TIMEOUT_SECONDS=1800  # 1800 = 30 min, 3600 = 1 hour
```

This prevents context from growing unbounded while maintaining enough history for natural conversations.

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required API keys
GEMINI_API_KEY=your_key
ELEVENLABS_API_KEY=your_key
PICOVOICE_ACCESS_KEY=your_key

# Optional: Memory settings
MAX_CONVERSATION_TURNS=20        # Number of turns to keep (0 = unlimited)
INACTIVITY_TIMEOUT_SECONDS=1800  # Clear history after 30 min of silence

# Optional: Audio settings
SILENCE_THRESHOLD_SECONDS=1.5    # Silence duration to detect end of speech
WHISPER_MODEL=base               # STT model: tiny, base, small, medium, large-v3
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov

# Integration tests (requires API keys)
pytest -m integration
```
