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

## Environment Variables

Copy `.env.example` to `.env` and configure:

```
GEMINI_API_KEY=your_key
ELEVENLABS_API_KEY=your_key
PICOVOICE_ACCESS_KEY=your_key
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
