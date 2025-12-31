# Architectural Decisions and Concerns

This document tracks key architectural decisions made during development and any concerns or issues discovered.

## Decisions

### D1: Python Version (3.13)

**Decision**: Require Python 3.13 specifically, not 3.14+

**Rationale**:
- `onnxruntime` (required by `silero-vad`) does not yet have wheels for Python 3.14
- This is a temporary constraint until onnxruntime releases 3.14 builds
- Specified in `pyproject.toml` as `requires-python = ">=3.13,<3.14"`

**Trade-offs**: Limits to Python 3.13.x only; will need to update when onnxruntime supports 3.14

---

### D2: google-genai SDK for Gemini Integration

**Decision**: Use the `google-genai` SDK instead of `google-generativeai`

**Rationale**:
- `google-genai` is the newer, more modern SDK
- Better async support with `client.aio.models.generate_content()`
- Cleaner API for tool calling via `GenerateContentConfig.tools`
- First-class Pydantic schema support via `response_schema`

**Implementation Notes**:
- Tools must be passed inside `GenerateContentConfig`, not as separate parameter
- Structured output uses `response_mime_type="application/json"` + `response_schema`

---

### D3: Agent Loop Design

**Decision**: Simple iterative loop with max iterations limit

**Rationale**:
- Prevents infinite loops when LLM keeps requesting tools
- Default max of 10 iterations is sufficient for most tasks
- Clear state management with ConversationState

**Trade-offs**:
- May terminate complex multi-step tasks early
- Could add more sophisticated loop termination logic later

---

### D4: Tool Registry Pattern

**Decision**: Central registry with handler functions, not classes

**Rationale**:
- Simpler than class-based tools
- Easy to register new tools with decorators
- Schema generation from ToolParameter definitions
- Async handlers throughout

**Trade-offs**: Less encapsulation than class-based approach

---

### D5: Silero VAD for End-of-Speech Detection

**Decision**: Use Silero VAD instead of simple energy-based detection

**Rationale**:
- Much more accurate than energy thresholds
- Works well with varying noise levels
- Built-in speech padding and minimum duration settings
- Runs on CPU efficiently

**Implementation Notes**:
- Requires sample rate of 8000 or 16000 Hz
- Using streaming wrapper for real-time detection
- Configurable silence threshold (default 1.5s)

---

### D6: Porcupine for Wake Word Detection

**Decision**: Use Picovoice Porcupine for wake word

**Rationale**:
- Industry-leading accuracy for wake word detection
- Runs entirely on-device (privacy)
- Built-in keywords available (jarvis, computer, etc.)
- Support for custom wake words

**Trade-offs**:
- Requires API key (free tier has limits)
- Commercial use requires license

---

### D7: State Machine for Voice Assistant

**Decision**: Simple enum-based state machine (IDLE → LISTENING → PROCESSING → SPEAKING)

**Rationale**:
- Clear state transitions
- Easy to understand and debug
- Callback-based notifications for UI
- Handles interruption via INTERRUPTED state

**States**:
- `IDLE`: Waiting for wake word or follow-up
- `LISTENING`: Recording user speech
- `PROCESSING`: Running STT and agent
- `SPEAKING`: Playing TTS response
- `INTERRUPTED`: User interrupted during playback

---

### D8: MP3 for TTS Audio Format

**Decision**: ElevenLabs outputs MP3, decode to PCM for playback

**Rationale**:
- MP3 is ElevenLabs' default/recommended format
- Smaller than raw PCM for network transfer
- PyAV handles decoding efficiently

**Trade-offs**: Extra decoding step; could use raw PCM if ElevenLabs supports it

---

### D9: Async-First Architecture

**Decision**: Use async/await throughout the codebase

**Rationale**:
- Better handling of concurrent I/O (audio, network)
- Non-blocking audio playback and recording
- Cleaner integration with asyncio event loop

**Implementation Notes**:
- Blocking operations wrapped with `run_in_executor`
- Audio callbacks use threading internally (sounddevice requirement)

---

## Concerns and Known Issues

### C1: Audio Thread Safety

**Concern**: Audio callbacks from sounddevice run in separate threads

**Mitigation**:
- Using thread-safe queues for audio data
- Locks for shared state in VAD and buffer classes

**Status**: Implemented but needs more testing under load

---

### C2: Wake Word Sample Rate Mismatch

**Concern**: Porcupine requires specific sample rate (16000 Hz)

**Mitigation**:
- Default sample rate set to 16000 Hz in settings
- Wake word detector validates sample rate on init

**Status**: Resolved by using consistent 16000 Hz throughout

---

### C3: ElevenLabs API Latency

**Concern**: TTS synthesis adds noticeable latency

**Potential Solutions**:
- Implement streaming TTS with chunked playback
- Pre-generate common responses
- Use local TTS as fallback

**Status**: Not yet optimized; streaming TTS is a future enhancement

---

### C4: Whisper Model Loading Time

**Concern**: First transcription is slow due to model loading

**Mitigation**:
- Lazy loading of model
- Model stays in memory after first load
- Could add eager loading option

**Status**: Acceptable for now; eager loading could be added

---

### C5: Gemini Tool Calling Reliability

**Concern**: LLM sometimes doesn't call tools when expected

**Observations**:
- Works well with clear tool descriptions
- System prompt helps guide tool usage
- Integration tests verify basic functionality

**Status**: Working; may need prompt engineering for edge cases

---

### C6: Missing Tests for Audio Components

**Concern**: Audio input/output/VAD/wake word have no unit tests

**Rationale**:
- These components require actual hardware (microphone, speakers)
- Mocking audio would require complex simulation
- Integration testing is more valuable here

**Plan**:
- Add integration tests that can be run with `--audio` flag
- Create test fixtures with pre-recorded audio files

**Status**: Acknowledged; audio tests deferred

---

## Future Considerations

### F1: Streaming Response

Could implement streaming agent responses with incremental TTS for lower latency.

### F2: Multi-Language TTS Voices

ElevenLabs supports multiple languages; could auto-select voice based on detected language.

### F3: Conversation Persistence

Could save conversation history to disk/database for resuming sessions.

### F4: Custom Tool Loading

Could load custom tools from a plugins directory or Python entry points.

### F5: Web Interface

Could add a web-based UI using FastAPI + WebSockets for browser access.

---

## Test Coverage

Current test coverage focuses on:
- Agent loop logic (mocked LLM)
- Tool registry and execution
- Conversation state management
- Configuration loading
- Gemini API integration (requires API key)

Not yet covered:
- Audio input/output
- TTS/STT modules
- VAD/Wake word detection
- VoiceAssistant orchestration
