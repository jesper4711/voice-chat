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

# With voice output
voice-chat chat --speak

# With voice input and output
voice-chat chat --speak --listen

# Single command
voice-chat run "What time is it?"
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
