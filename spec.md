# Voice Chat - Specification

## Overview

Voice Chat is a hybrid AI assistant that combines the always-on, voice-activated nature of Alexa with the conversational depth and extensibility of ChatGPT. Unlike commercial assistants, it runs locally on your hardware, offers unlimited customization, and can control your entire computer.

**Key differentiators:**
- Always listening, wake-word activated
- Extremely extensible tool system (can execute any CLI command)
- Full computer control (not sandboxed like Alexa/Siri)
- Local knowledge base with persistent memory
- Can spawn headless Claude Code sessions for complex tasks

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Voice Chat                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Wake    │───▶│  Speech  │───▶│   LLM    │───▶│  Text    │  │
│  │  Word    │    │  to Text │    │ (Gemini) │    │ to Speech│  │
│  │Detection │    │  (STT)   │    │          │    │  (TTS)   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │                               │                          │
│       │                               ▼                          │
│       │                        ┌──────────┐                     │
│       │                        │  Tool    │                     │
│       │                        │  System  │                     │
│       │                        └──────────┘                     │
│       │                               │                          │
│       ▼                               ▼                          │
│  ┌──────────┐                  ┌──────────┐                     │
│  │  Audio   │                  │ Knowledge│                     │
│  │  Input   │                  │   Base   │                     │
│  │  (Mic)   │                  │(Postgres)│                     │
│  └──────────┘                  └──────────┘                     │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                        Web UI (Testing)                          │
│                      Text chat interface                         │
└─────────────────────────────────────────────────────────────────┘
```

## Technical Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Language | Python 3.13 | Latest stable, improved REPL, JIT compiler |
| LLM | Google Gemini API | Gemini 3 Flash as primary model |
| Wake Word | Porcupine (Picovoice) | Free tier available, custom wake words |
| STT | Faster-Whisper | Local, 4x faster than OpenAI Whisper |
| TTS | ElevenLabs | Eleven v3 or Multilingual v2 |
| VAD | Silero VAD | Determines when user stops speaking |
| Database | PostgreSQL + pgvector | Vector search for RAG |
| Web Framework | FastAPI | Simple REST API + WebSocket |
| Audio | sounddevice | Cross-platform audio I/O, NumPy arrays |

## Versions (as of December 2025)

### Python & Runtime

| Component | Version | Released | Notes |
|-----------|---------|----------|-------|
| Python | 3.13.11 | Dec 5, 2025 | Stable, new REPL, experimental JIT |

Python 3.13 features:
- New interactive shell with multiline editing and syntax highlighting
- Experimental free-threaded mode (GIL disabled) - `python3.13t`
- Experimental JIT compiler (disabled by default)
- Improved error messages with color
- Support until October 2029

### LLM Models

| Model | Model ID | Input | Output | Context | Notes |
|-------|----------|-------|--------|---------|-------|
| Gemini 3 Flash | `gemini-3-flash-preview` | $0.50/M | $3.00/M | 1M tokens | Primary choice, Dec 17 2025 |
| Gemini 2.5 Pro | `gemini-2.5-pro` | $1.25/M | $10/M | 1M tokens | Best for coding tasks |
| Gemini 2.5 Flash | `gemini-2.5-flash` | Lower | Lower | 1M tokens | Budget option |

### TTS Models

| Model | Model ID | Notes |
|-------|----------|-------|
| Eleven v3 | `eleven_v3` | Latest, most expressive, 70+ languages |
| Eleven Multilingual v2 | `eleven_multilingual_v2` | Stable, Swedish support |
| Eleven Flash v2.5 | `eleven_flash_v2_5` | 75ms latency, real-time |

Note: Eleven v3 is not recommended for real-time/agent applications.

### Python Packages

| Package | Version | Released | Python | Notes |
|---------|---------|----------|--------|-------|
| google-genai | 1.56.0 | Dec 17, 2025 | >=3.10 | New SDK, replaces google-generativeai |
| faster-whisper | 1.2.1 | Oct 31, 2025 | >=3.9 | Built-in Silero VAD support |
| pvporcupine | 4.0.1 | Dec 18, 2025 | >=3.9 | Wake word detection |
| silero-vad | 6.2.0 | Nov 6, 2025 | >=3.8 | Voice activity detection |
| elevenlabs | 2.27.0 | Dec 15, 2025 | >=3.8 | TTS SDK |
| fastapi | 0.128.0 | Dec 27, 2025 | >=3.9 | Web framework |
| sounddevice | 0.5.3 | Oct 19, 2025 | >=3.7 | Audio I/O |
| pgvector | 0.4.2 | Dec 5, 2025 | >=3.9 | PostgreSQL vector extension |

### requirements.txt

```
# Core
python>=3.13

# LLM
google-genai>=1.56.0
pydantic-ai>=0.1.0            # Optional: agent framework
google-adk>=0.5.0             # Optional: Google's agent framework

# Audio
faster-whisper>=1.2.1
pvporcupine>=4.0.1
silero-vad>=6.2.0
sounddevice>=0.5.3
numpy>=2.0.0

# TTS
elevenlabs>=2.27.0

# CLI
typer>=0.15.0
rich>=13.9.0                 # Pretty terminal output

# Testing
pytest>=8.3.0
pytest-asyncio>=0.24.0
pytest-cov>=6.0.0
respx>=0.22.0                # Mock HTTP requests

# Database (future)
pgvector>=0.4.2
psycopg>=3.2.0
sqlalchemy>=2.0.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.10.0
```

### Important Notes

1. **google-genai vs google-generativeai**: The old `google-generativeai` package is deprecated as of Nov 30, 2025. Use `google-genai` instead.

2. **CUDA compatibility**: faster-whisper's ctranslate2 dependency requires CUDA 12 + cuDNN 9. For older CUDA versions:
   - CUDA 11 + cuDNN 8: Use ctranslate2==3.24.0
   - CUDA 12 + cuDNN 8: Use ctranslate2==4.4.0

3. **Apple Silicon**: All packages support macOS arm64 natively. faster-whisper runs efficiently on M1/M2/M3.

## Language Support

The assistant supports both English and Swedish, with automatic language detection.

### Configuration

```python
LANGUAGE_MODE = "auto"  # Options: "en", "sv", "auto"
```

| Mode | Behavior |
|------|----------|
| `en` | Always respond in English, regardless of input language |
| `sv` | Always respond in Swedish, regardless of input language |
| `auto` | Detect user's language and respond in the same language |

### Component Support by Language

| Component | English | Swedish | Notes |
|-----------|---------|---------|-------|
| Wake Word | Excellent | Limited | Use English wake word for reliability |
| STT (Whisper) | Excellent | Excellent | Auto-detects language, handles code-switching |
| LLM (Gemini) | Excellent | Very Good | Responds naturally in both languages |
| TTS (ElevenLabs) | Excellent | Very Good | Multilingual v2 model supports Swedish |
| TTS (Azure) | Excellent | Excellent | Best Swedish neural voices |

### Speech-to-Text Language Handling

Faster-Whisper handles multilingual input automatically:
- Auto-detects language per utterance
- Handles "Swenglish" (mixed Swedish/English) reasonably well
- Can be forced to specific language if needed

```python
# Auto-detect (recommended)
result = model.transcribe(audio)

# Force Swedish
result = model.transcribe(audio, language="sv")
```

### LLM Language Handling

System prompt instructs the LLM on language behavior:

```
# For auto mode:
"Respond in the same language the user speaks. If they speak Swedish,
respond in Swedish. If they speak English, respond in English."

# For forced Swedish:
"Always respond in Swedish, regardless of what language the user speaks."
```

### TTS Language Handling

ElevenLabs Multilingual v2 automatically speaks the detected language:
- Same voice works for both English and Swedish
- No voice switching needed
- Natural accent in both languages

### Wake Word Consideration

For reliability, use an English wake word even for Swedish conversations:
- Porcupine's pre-trained models are English-optimized
- Names that work in both languages: "Hey Nova", "OK Computer", "Hey Jarvis"
- Custom Swedish wake words possible but require more training data

### Future: Full Swedish Wake Word

If a native Swedish wake word is desired:
1. Use Picovoice Console to train custom wake word
2. Record multiple speakers with Swedish pronunciation
3. Test extensively for false positive/negative rates

## Core Components

### 1. Wake Word Detection

Continuously listens for a trigger phrase to activate the assistant.

**Options evaluated:**
- **Porcupine (Picovoice)** - Recommended. Low CPU, custom wake words, good accuracy
- **OpenWake** - Open source alternative, requires more setup
- **Snowboy** - Deprecated but still functional

**Configuration:**
- Wake word: TBD (placeholder: "Hey Jarvis" or custom)
- Sensitivity: Adjustable to reduce false positives
- Always running in background with minimal CPU usage

### 2. Speech-to-Text (STT)

Converts user speech to text after wake word activation.

**Primary choice: Faster-Whisper (local)**
- Runs locally, no API costs
- Good accuracy, especially with `large-v3` model
- ~1-2 second latency on Apple Silicon
- Excellent multilingual support (English + Swedish)
- Auto-detects language, handles code-switching

**Alternative: Google Speech-to-Text API**
- Lower latency, streaming capable
- Costs money per minute
- Better for real-time applications
- Also supports Swedish

### 3. Voice Activity Detection (VAD)

Determines when the user has finished speaking. Critical for natural conversation flow.

**Implementation:**
- Use Silero VAD or WebRTC VAD
- Configurable silence threshold (default: 1.5 seconds)
- Detects speech vs. background noise
- Prevents premature interruption during pauses

### 4. LLM Integration (Gemini)

The conversational brain of the assistant.

**Primary model: Gemini 3 Flash** (`gemini-3-flash-preview`)
- Released Dec 17, 2025
- $0.50/M input, $3.00/M output tokens
- 1M token context window, 64k output
- Function calling, streaming, system instructions

**Setup:**
- Use `google-genai` SDK (not deprecated `google-generativeai`)
- System prompt defines assistant personality and capabilities
- Tool definitions registered for function calling
- Conversation history maintained for context

```python
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Hello, world!"
)
```

**Future considerations:**
- Local LLM fallback (Ollama with Llama/Mistral)
- Multi-model routing based on task complexity
- Gemini 2.5 Pro for complex coding tasks

### 4.1 Structured Output & Pydantic Integration

Getting reliable JSON from LLMs is critical for tool calling. Gemini supports structured outputs natively.

**Two approaches:**

#### Option A: Native google-genai with Pydantic (Recommended for simple cases)

```python
from google import genai
from pydantic import BaseModel, Field
from typing import List, Optional

class ToolCall(BaseModel):
    tool_name: str = Field(description="Name of the tool to call")
    arguments: dict = Field(description="Arguments to pass to the tool")
    reasoning: Optional[str] = Field(description="Why this tool was chosen")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Turn on the lights in the living room",
    config={
        "response_mime_type": "application/json",
        "response_schema": ToolCall,  # Pydantic model directly
    },
)

# Parse response
tool_call = ToolCall.model_validate_json(response.text)
```

#### Option B: Pydantic AI Framework (Recommended for agents)

```python
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic import BaseModel

class AssistantResponse(BaseModel):
    message: str
    tool_calls: list[ToolCall] | None
    needs_clarification: bool

agent = Agent(
    GoogleModel("gemini-3-flash-preview"),
    result_type=AssistantResponse,
    system_prompt="You are a helpful voice assistant..."
)

result = await agent.run("What's on my calendar today?")
print(result.data)  # Typed AssistantResponse
```

**Pydantic AI benefits:**
- Type-safe agent framework (like FastAPI for agents)
- Built-in tool registration with type hints
- Automatic retry on malformed responses
- Supports Gemini, OpenAI, Anthropic, and more

**Known limitations & workarounds:**

| Issue | Workaround |
|-------|------------|
| Default values in Pydantic fields cause errors | Use `Optional[T]` instead of `T = default` |
| `additionalProperties` rejected by SDK | Use `response_json_schema` instead of `response_schema` |
| Complex nested schemas rejected | Flatten structure, shorten property names |
| Nullable fields (`str | None`) not supported | Use `Optional[str]` and omit from `required` |
| Conditional schemas (`if/then`) not supported | Validate manually after generation |

**Best practices:**

1. **Use Field descriptions** - Helps the model understand intent
   ```python
   name: str = Field(description="The user's full name")
   ```

2. **Keep schemas simple** - Flat is better than nested
   ```python
   # Good
   class Response(BaseModel):
       action: str
       target: str

   # Avoid deep nesting
   class Response(BaseModel):
       action: Action
           sub_action: SubAction
               details: Details  # Too deep
   ```

3. **Validate semantics, not just syntax** - Structured output guarantees valid JSON, not correct values
   ```python
   tool_call = ToolCall.model_validate_json(response.text)
   if tool_call.tool_name not in REGISTERED_TOOLS:
       raise ValueError(f"Unknown tool: {tool_call.tool_name}")
   ```

4. **Use enums for fixed choices**
   ```python
   from enum import Enum

   class Intent(str, Enum):
       PLAY_MUSIC = "play_music"
       SET_TIMER = "set_timer"
       SEARCH_WEB = "search_web"
       GENERAL_CHAT = "general_chat"

   class Response(BaseModel):
       intent: Intent
       confidence: float
   ```

**Package versions for Pydantic integration:**

```
pydantic>=2.10.0
pydantic-ai>=0.1.0  # Optional, for agent framework
```

### 5. Text-to-Speech (TTS)

Converts LLM responses to natural speech.

**Model options:**

| Model | Model ID | Best for |
|-------|----------|----------|
| Eleven v3 | `eleven_v3` | Highest quality, expressive, 70+ languages |
| Multilingual v2 | `eleven_multilingual_v2` | Stable, good Swedish, real-time capable |
| Flash v2.5 | `eleven_flash_v2_5` | Lowest latency (75ms), real-time agents |

**Recommendation:** Start with Multilingual v2 for real-time voice chat. Use v3 for pre-generated content.

**Key features:**
- High-quality, natural voices
- Low latency streaming
- ~$5/month for hobby use (Starter plan)
- Same voice works for English and Swedish
- Auto-detects language from text

**Alternatives:**
- Azure TTS - Excellent Swedish neural voices, $16/M characters
- OpenAI TTS - Good quality, simple API, multilingual
- Google Cloud TTS - Reliable, Swedish Neural2 voices available
- macOS `say` command - Free, robotic, English only

### 5.1 Agent Loop Architecture

The core of the assistant is an agent loop that processes user requests through iterative reasoning and action cycles.

#### The Basic Agent Loop

```
┌─────────┐      ┌──────────┐      ┌─────────────┐
│  Human  │─────▶│ LLM Call │─────▶│ Environment │
│ (Voice) │      │          │      │  (Tools)    │
└─────────┘      └────┬─────┘      └──────┬──────┘
                      │                    │
                      │    ◀───Feedback────┘
                      │         Action────▶
                      ▼
                 ┌────────┐
                 │  Stop  │ (task complete or max iterations)
                 └────────┘
```

**The loop in pseudocode:**

```python
async def agent_loop(user_message: str, max_iterations: int = 10) -> str:
    messages = [{"role": "user", "content": user_message}]

    for i in range(max_iterations):
        # 1. Call LLM with current context
        response = await llm.generate(
            model="gemini-3-flash-preview",
            messages=messages,
            tools=registered_tools,
        )

        # 2. Check if LLM wants to use a tool
        if response.tool_calls:
            for tool_call in response.tool_calls:
                # 3. Execute tool (Action)
                result = await execute_tool(
                    tool_call.name,
                    tool_call.arguments
                )
                # 4. Add result to context (Feedback)
                messages.append({
                    "role": "tool",
                    "name": tool_call.name,
                    "content": result
                })
        else:
            # 5. No tool calls = task complete
            return response.text

    return "Max iterations reached"
```

#### Agent Patterns

| Pattern | Description | Best For |
|---------|-------------|----------|
| **ReAct** | Thought → Action → Observation loop | General tasks, debugging |
| **Plan-and-Execute** | Plan all steps first, then execute | Complex multi-step tasks |
| **Agentic RAG** | Iterative retrieval with validation | Knowledge-intensive queries |

**ReAct Pattern (Recommended for Voice Chat):**

```python
class ReActResponse(BaseModel):
    thought: str = Field(description="Reasoning about what to do next")
    action: Optional[str] = Field(description="Tool to call, or null if done")
    action_input: Optional[dict] = Field(description="Arguments for the tool")
    final_answer: Optional[str] = Field(description="Response to user if done")
```

The LLM explicitly reasons before acting, making debugging easier.

#### Framework Options

| Framework | Pros | Cons | Best For |
|-----------|------|------|----------|
| **Native loop** | Simple, full control | More code to maintain | Learning, simple agents |
| **Pydantic AI** | Type-safe, FastAPI-like | Newer, less docs | Production with types |
| **Google ADK** | Official Google, Gemini-optimized | Google ecosystem lock-in | Gemini-first projects |
| **LangGraph** | Powerful, stateful, visual | Complex, heavy | Complex multi-agent |

**Recommendation:** Start with **Pydantic AI** for type safety, or **native loop** to understand the fundamentals. Consider **Google ADK** if staying in Google ecosystem.

#### Voice-Specific Considerations

For a voice assistant, the agent loop has special requirements:

1. **Streaming responses** - Start TTS before full response is ready
   ```python
   async for chunk in llm.stream_generate(...):
       if chunk.text:
           await tts.speak_chunk(chunk.text)
   ```

2. **Interruptibility** - User can interrupt mid-response
   ```python
   while speaking:
       if wake_word_detected():
           cancel_current_speech()
           break
   ```

3. **Progress feedback** - Tell user what's happening
   ```python
   # Before long-running tool
   await speak("Let me check your calendar...")
   result = await calendar_tool.get_events()
   ```

4. **Graceful failures** - Don't leave user hanging
   ```python
   try:
       result = await tool.execute(timeout=10)
   except TimeoutError:
       await speak("Sorry, that's taking too long. Let me try another way.")
   ```

#### Stopping Conditions

The loop must know when to stop:

```python
class StopCondition(Enum):
    TASK_COMPLETE = "task_complete"      # LLM says done
    MAX_ITERATIONS = "max_iterations"    # Safety limit (e.g., 10)
    USER_INTERRUPT = "user_interrupt"    # Wake word detected
    ERROR = "error"                      # Unrecoverable failure
    TIMEOUT = "timeout"                  # Overall time limit
```

#### Agentic RAG for Knowledge Queries

When the user asks knowledge-intensive questions:

```
User: "What was discussed in last week's team meeting?"

Loop:
  1. LLM decides: need to search knowledge base
  2. Action: search_documents("team meeting last week")
  3. Feedback: [3 documents found, relevance scores]
  4. LLM evaluates: document 1 looks relevant but incomplete
  5. Action: search_documents("team meeting action items")
  6. Feedback: [2 more documents]
  7. LLM synthesizes: combines information
  8. Stop: returns summary to user
```

Unlike simple RAG (one-shot retrieval), agentic RAG iterates until confident.

#### Hybrid Retrieval Strategy

**Key insight:** Semantic search isn't always best. The LLM should choose the retrieval method.

| Query Type | Best Tool | Example |
|------------|-----------|---------|
| Conceptual | Semantic search | "What do we know about customer onboarding?" |
| Exact match | Keyword/SQL | "Find emails from John Smith" |
| Filtered | Tag + SQL | "Meeting notes from December" |
| Combined | Hybrid | "What did John say about the Q4 budget?" |

**Retrieval tools for the LLM:**

```python
@tool(description="Search by meaning/concept. Best for questions about topics, themes, or 'things like X'.")
async def semantic_search(query: str, limit: int = 5) -> list[Document]:
    embedding = await embed(query)
    return await db.vector_search(embedding, limit=limit)

@tool(description="Search by exact keywords. Best for names, IDs, specific terms, or quoted phrases.")
async def keyword_search(
    keywords: list[str],
    match_all: bool = False,
    limit: int = 10
) -> list[Document]:
    return await db.fulltext_search(keywords, match_all=match_all, limit=limit)

@tool(description="Filter by metadata tags. Best for date ranges, categories, sources, or people.")
async def filter_search(
    tags: Optional[list[str]] = None,
    source: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    person: Optional[str] = None,
    limit: int = 20
) -> list[Document]:
    return await db.metadata_filter(
        tags=tags, source=source,
        date_from=date_from, date_to=date_to,
        person=person, limit=limit
    )

@tool(description="Combine semantic search with filters. Use when you need both concept matching AND metadata constraints.")
async def hybrid_search(
    query: str,
    tags: Optional[list[str]] = None,
    person: Optional[str] = None,
    date_from: Optional[str] = None,
    limit: int = 10
) -> list[Document]:
    # Pre-filter by metadata, then semantic search within results
    candidates = await db.metadata_filter(tags=tags, person=person, date_from=date_from, limit=100)
    embedding = await embed(query)
    return await rerank_by_similarity(candidates, embedding, limit=limit)
```

**Example: LLM choosing the right tool**

```
User: "What did Erik say about the API redesign?"

LLM reasoning:
- "Erik" is a specific name → needs keyword/filter, not semantic
- "API redesign" is a concept → could use semantic
- Best approach: hybrid search with person filter

Action: hybrid_search(
    query="API redesign",
    person="Erik",
    limit=10
)
```

**Document tagging schema:**

```python
class DocumentMetadata(BaseModel):
    source: str          # "email", "meeting", "slack", "document"
    tags: list[str]      # ["project-x", "api", "technical"]
    people: list[str]    # ["Erik", "Anna"] - mentioned or involved
    date: datetime
    title: Optional[str]
    summary: Optional[str]  # LLM-generated summary for quick scanning
```

**Why this matters:**

| Search Type | "Find emails from Erik" | Score |
|-------------|-------------------------|-------|
| Semantic only | Finds emails *about* Erik, misses emails *from* Erik | Poor |
| Keyword only | Finds exact match "Erik" anywhere | Good |
| Filter (person=Erik, source=email) | Exact match on metadata | Best |

The LLM's job is to decompose the query and pick the right tool(s).

### 6. Tool System

Extensible architecture for adding capabilities.

**Core design principles:**
- Tools are Python functions with JSON schema definitions
- LLM decides when to call tools based on user intent
- Results fed back to LLM for natural response
- Easy to add new tools without modifying core code

**Tool registration example:**
```python
@tool(
    name="execute_command",
    description="Execute a CLI command on the system",
    parameters={
        "command": {"type": "string", "description": "The command to execute"}
    }
)
async def execute_command(command: str) -> str:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout or result.stderr
```

**MVP Tools:**
- `execute_command` - Run any CLI command
- `get_current_time` - Current date/time
- `web_search` - Search the web (via API)
- `control_volume` - System volume control (AppleScript)
- `open_application` - Launch macOS apps

### 7. Knowledge Base

PostgreSQL with pgvector for hybrid retrieval (semantic + keyword + metadata).

**Schema design:**
```sql
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For fuzzy text search

-- Conversation history
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    role TEXT NOT NULL,  -- 'user', 'assistant', 'tool'
    content TEXT NOT NULL,
    tool_name TEXT,      -- if role='tool'
    embedding VECTOR(1536)
);

-- Knowledge documents with hybrid search support
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Content
    content TEXT NOT NULL,
    chunk_index INTEGER,
    parent_id INTEGER REFERENCES documents(id),

    -- Metadata for filtering
    source TEXT NOT NULL,        -- 'email', 'meeting', 'slack', 'document', 'calendar'
    title TEXT,
    summary TEXT,                -- LLM-generated summary
    date TIMESTAMPTZ,            -- When the content is from (not when ingested)

    -- Hybrid search fields
    embedding VECTOR(1536),      -- For semantic search
    search_vector TSVECTOR,      -- For full-text keyword search

    -- Structured metadata
    metadata JSONB DEFAULT '{}'
);

-- People mentioned/involved (many-to-many)
CREATE TABLE document_people (
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    person_name TEXT NOT NULL,
    role TEXT,  -- 'author', 'recipient', 'mentioned', 'attendee'
    PRIMARY KEY (document_id, person_name)
);

-- Tags (many-to-many)
CREATE TABLE document_tags (
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (document_id, tag)
);

-- Indexes for each search type
CREATE INDEX idx_documents_embedding ON documents
    USING hnsw (embedding vector_cosine_ops);           -- Semantic search

CREATE INDEX idx_documents_search_vector ON documents
    USING gin (search_vector);                          -- Full-text search

CREATE INDEX idx_documents_source ON documents (source);
CREATE INDEX idx_documents_date ON documents (date);
CREATE INDEX idx_document_people_name ON document_people (person_name);
CREATE INDEX idx_document_tags_tag ON document_tags (tag);
CREATE INDEX idx_documents_metadata ON documents USING gin (metadata);

-- Auto-update search_vector on insert/update
CREATE OR REPLACE FUNCTION update_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', COALESCE(NEW.title, '') || ' ' || NEW.content);
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER documents_search_vector_trigger
    BEFORE INSERT OR UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_search_vector();
```

**Query examples:**

```sql
-- Semantic search (conceptual queries)
SELECT id, title, content, 1 - (embedding <=> $1) AS similarity
FROM documents
ORDER BY embedding <=> $1
LIMIT 10;

-- Keyword search (exact terms, names)
SELECT id, title, content, ts_rank(search_vector, query) AS rank
FROM documents, plainto_tsquery('english', $1) query
WHERE search_vector @@ query
ORDER BY rank DESC
LIMIT 10;

-- Filter by person
SELECT d.* FROM documents d
JOIN document_people dp ON d.id = dp.document_id
WHERE dp.person_name ILIKE '%erik%'
ORDER BY d.date DESC;

-- Hybrid: filter first, then semantic search
WITH filtered AS (
    SELECT d.* FROM documents d
    JOIN document_people dp ON d.id = dp.document_id
    WHERE dp.person_name ILIKE '%erik%'
    AND d.date > NOW() - INTERVAL '30 days'
)
SELECT id, title, content, 1 - (embedding <=> $1) AS similarity
FROM filtered
ORDER BY embedding <=> $1
LIMIT 10;
```

**Agentic RAG pipeline:**
1. LLM analyzes query to determine search strategy
2. LLM selects appropriate tool(s): semantic, keyword, filter, or hybrid
3. Execute search, return results with relevance scores
4. LLM evaluates results - may iterate with refined search
5. LLM synthesizes final answer from retrieved context

### 8. CLI Interface

Simple command-line interface for testing and interaction.

**Commands:**
```bash
# Text chat only
uv run python -m voice_chat chat

# Text chat with voice output
uv run python -m voice_chat chat --speak

# Full voice mode (push-to-talk)
uv run python -m voice_chat chat --speak --listen

# Always-on with wake word
uv run python -m voice_chat

# Run a single command
uv run python -m voice_chat run "what time is it"
```

**Implementation:**
- Simple input/output loop using `input()` for MVP
- Rich/Textual for better formatting (optional)
- Typer for CLI argument parsing

## Hardware

### Development (MacBook)
- Built-in microphone for testing
- Built-in speakers
- Local PostgreSQL via Homebrew/Docker

### Production (Mac Mini)
- External USB microphone (recommended: conference mic with noise cancellation)
- External speakers or audio output
- Always-on, headless operation
- PostgreSQL running locally

**Mac Mini considerations:**
- M1/M2 Mac Mini has no built-in microphone
- Recommend: Jabra Speak 510 or similar USB speakerphone
- Or: USB microphone array + separate speakers

## MVP Scope (v1)

### MVP Phases

Build incrementally, test each phase before moving on.

#### Phase 0: Project Skeleton
**Goal:** Basic project structure, configs, dependencies install correctly.

```
voice-chat/
├── pyproject.toml
├── .env.example
├── src/
│   └── voice_chat/
│       ├── __init__.py
│       └── main.py
└── README.md
```

**Success:** `uv run python -m voice_chat` runs without errors.

---

#### Phase 1: CLI Chat + Tools
**Goal:** Text conversation with Gemini via CLI, basic tool calling works.

**Components:**
- [ ] Gemini API integration (`google-genai`)
- [ ] Basic agent loop (no frameworks yet)
- [ ] Tool registry with `execute_command`
- [ ] CLI chat interface (simple input/output loop)
- [ ] Conversation history (in-memory)

**Success criteria:**
```
$ uv run python -m voice_chat chat
You: What time is it?
Assistant: [calls get_current_time tool] It's 14:32.
You: List files in current directory
Assistant: [calls execute_command tool] Here are the files: ...
```

---

#### Phase 2: Voice Output (TTS)
**Goal:** Assistant speaks responses through speakers.

**Components:**
- [ ] ElevenLabs TTS integration
- [ ] Audio playback (sounddevice)
- [ ] Streaming TTS (start speaking before full response)

**Success criteria:**
```
$ uv run python -m voice_chat chat --speak
You: Hello
Assistant: [speaks] "Hello! How can I help you?"
```

---

#### Phase 3: Voice Input (STT)
**Goal:** You speak, it transcribes to text.

**Components:**
- [ ] Microphone input (sounddevice)
- [ ] Faster-Whisper STT integration
- [ ] Voice Activity Detection (Silero VAD)
- [ ] Language detection (English/Swedish)

**Success criteria:**
```
$ uv run python -m voice_chat chat --speak --listen
[Press Enter to speak]
[Recording... speak now]
[Detected: "What's the weather like?"]
Assistant: [speaks response]
```

---

#### Phase 4: Wake Word
**Goal:** Always-on listening, wake word activates assistant.

**Components:**
- [ ] Porcupine wake word detection
- [ ] Background audio monitoring
- [ ] State machine: IDLE → LISTENING → PROCESSING → SPEAKING → IDLE

**Success criteria:**
```
$ uv run python -m voice_chat
Listening for wake word...
[Say "Hey Jarvis"]
[Wake word detected!]
[Recording your message...]
[You said: "Turn on the lights"]
Assistant: [speaks] "I'll run that command for you..."
[Back to listening for wake word...]
```

---

#### Phase 5: Polish & Robustness
**Goal:** Natural conversation flow, error handling, interrupts.

**Components:**
- [ ] Interrupt handling (wake word cancels current speech)
- [ ] Graceful error recovery
- [ ] Conversation persistence (save to file)
- [ ] Progress feedback ("Let me check...")
- [ ] Timeout handling

**Success criteria:**
- Can interrupt assistant mid-sentence with wake word
- Recovers gracefully from network errors
- Conversation survives restart

---

### MVP Feature Summary

| Feature | Phase | Status |
|---------|-------|--------|
| Project setup | 0 | ⬜ |
| Gemini LLM integration | 1 | ⬜ |
| Basic agent loop | 1 | ⬜ |
| Tool: `execute_command` | 1 | ⬜ |
| Tool: `get_current_time` | 1 | ⬜ |
| CLI chat interface | 1 | ⬜ |
| ElevenLabs TTS | 2 | ⬜ |
| Audio playback | 2 | ⬜ |
| Microphone input | 3 | ⬜ |
| Faster-Whisper STT | 3 | ⬜ |
| Silero VAD | 3 | ⬜ |
| Language detection | 3 | ⬜ |
| Porcupine wake word | 4 | ⬜ |
| State machine | 4 | ⬜ |
| Interrupt handling | 5 | ⬜ |
| Error recovery | 5 | ⬜ |
| Conversation persistence | 5 | ⬜ |

### Excluded from MVP (future phases)
- Web UI (CLI is sufficient)
- PostgreSQL knowledge base
- RAG functionality
- Google Calendar integration
- Gmail integration
- Spotify integration
- Claude Code spawning
- Remote access
- Custom wake words
- Local LLM fallback

## Future Roadmap

### Phase 2: Integrations
- Google Calendar (view/create events)
- Gmail (read/summarize/draft emails)
- Spotify (play music, podcasts, control playback)

### Phase 3: Claude Code
- Spawn headless Claude Code sessions
- Background task execution
- Notification when tasks complete
- Voice-controlled coding assistance

### Phase 4: Knowledge & Memory
- PostgreSQL with pgvector
- Document ingestion pipeline
- Long-term memory across sessions
- Personal knowledge base building

### Phase 5: Advanced Features
- Local LLM for simple queries (cost optimization)
- Multi-room audio (multiple Mac Minis)
- Mobile app for remote access
- Custom wake word training
- Proactive notifications

## Security Considerations

### Current (v1)
- Runs on local network only
- No authentication (single user, trusted network)
- API keys stored in environment variables

### Future (if remote access enabled)
- HTTPS with valid certificate
- Authentication (API key or OAuth)
- Rate limiting
- Audit logging
- Consider VPN instead of exposing to internet

### Privacy
- All conversations stored locally
- Cloud APIs see conversation content (Gemini, ElevenLabs)
- For sensitive queries, future local LLM option

## Configuration

Environment variables (`.env` file):
```bash
GEMINI_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
PICOVOICE_ACCESS_KEY=your_key_here

# Language settings
LANGUAGE_MODE=auto  # "en", "sv", or "auto"

# Optional
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
POSTGRES_URL=postgresql://localhost/voicechat
```

## Directory Structure

```
voice-chat/
├── spec.md
├── README.md
├── pyproject.toml           # Project config, dependencies
├── .env                     # API keys (not in git)
├── .env.example             # Template for .env
│
├── src/voice_chat/          # Main package
│   ├── __init__.py
│   ├── __main__.py          # Entry point: python -m voice_chat
│   ├── cli.py               # CLI commands (Typer)
│   ├── config.py            # Settings from env vars
│   │
│   ├── agent/               # Core agent logic
│   │   ├── __init__.py
│   │   ├── loop.py          # Main agent loop
│   │   ├── state.py         # Conversation state
│   │   └── prompts.py       # System prompts
│   │
│   ├── llm/                 # LLM integration
│   │   ├── __init__.py
│   │   └── gemini.py        # Gemini API client
│   │
│   ├── audio/               # Voice I/O
│   │   ├── __init__.py
│   │   ├── input.py         # Microphone recording
│   │   ├── output.py        # Speaker playback
│   │   ├── stt.py           # Speech-to-text (Whisper)
│   │   ├── tts.py           # Text-to-speech (ElevenLabs)
│   │   ├── vad.py           # Voice activity detection
│   │   └── wake_word.py     # Porcupine wake word
│   │
│   └── tools/               # Tool implementations
│       ├── __init__.py
│       ├── registry.py      # Tool registration system
│       ├── system.py        # execute_command, open_app, etc.
│       └── time.py          # get_current_time, etc.
│
├── tests/
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_tools.py
│   └── test_audio.py
│
└── scripts/
    └── setup.sh             # Initial setup helper
```

## Testing Strategy

### Testing Pyramid

```
        ┌─────────────┐
        │   Manual    │  ← User verification at phase completion
        │  (you)      │
        ├─────────────┤
        │    E2E      │  ← Full voice flow (Phase 4+)
        ├─────────────┤
        │ Integration │  ← Real API calls (Gemini, ElevenLabs)
        ├─────────────┤
        │    Unit     │  ← Fast, mocked, run on every change
        └─────────────┘
```

### Test Types

| Type | Speed | API Keys | When to Run | Who Runs |
|------|-------|----------|-------------|----------|
| Unit | <1s each | No | Every change | Claude (autonomous) |
| Integration | 2-10s each | Yes | Before phase completion | Claude (autonomous) |
| E2E | 30s+ | Yes | Phase completion | Claude + User verification |
| Manual | - | Yes | Phase completion | User |

### Unit Tests (Mocked)

Fast tests that don't need real APIs. Mock external dependencies.

```python
# tests/unit/test_agent_loop.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from voice_chat.agent.loop import agent_loop

@pytest.mark.asyncio
async def test_agent_loop_simple_response():
    """LLM returns text without tool calls → returns response"""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MockResponse(text="Hello!", tool_calls=[])

    result = await agent_loop("Hi", llm=mock_llm, tools=[])

    assert result == "Hello!"
    mock_llm.generate.assert_called_once()

@pytest.mark.asyncio
async def test_agent_loop_with_tool_call():
    """LLM requests tool → executes tool → feeds result back"""
    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = [
        MockResponse(text="", tool_calls=[ToolCall("get_time", {})]),
        MockResponse(text="It's 2pm", tool_calls=[]),
    ]
    mock_tool = AsyncMock(return_value="14:00")

    result = await agent_loop("What time?", llm=mock_llm, tools={"get_time": mock_tool})

    assert "2pm" in result
    assert mock_llm.generate.call_count == 2
    mock_tool.assert_called_once()

@pytest.mark.asyncio
async def test_agent_loop_max_iterations():
    """Safety: stops after max iterations"""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MockResponse(
        text="", tool_calls=[ToolCall("loop_forever", {})]
    )

    result = await agent_loop("Loop", llm=mock_llm, max_iterations=3)

    assert mock_llm.generate.call_count == 3  # Stopped at limit
```

### Integration Tests (Real APIs)

Test actual API integrations. Require API keys. Run less frequently.

```python
# tests/integration/test_gemini.py
import pytest
from voice_chat.llm.gemini import GeminiClient

@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_simple_chat():
    """Verify Gemini API connection and basic response"""
    client = GeminiClient()  # Uses GEMINI_API_KEY from env

    response = await client.generate("Say 'test successful' and nothing else.")

    assert "test successful" in response.text.lower()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_structured_output():
    """Verify Pydantic schema enforcement works"""
    from pydantic import BaseModel

    class TestResponse(BaseModel):
        message: str
        number: int

    client = GeminiClient()
    response = await client.generate(
        "Return a greeting and the number 42",
        response_schema=TestResponse
    )

    parsed = TestResponse.model_validate_json(response.text)
    assert parsed.number == 42

@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_tool_calling():
    """Verify tool calling works end-to-end"""
    client = GeminiClient()

    tools = [{
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {"city": {"type": "string"}}
    }]

    response = await client.generate(
        "What's the weather in Stockholm?",
        tools=tools
    )

    assert response.tool_calls
    assert response.tool_calls[0].name == "get_weather"
    assert "stockholm" in response.tool_calls[0].arguments["city"].lower()
```

### Running Tests

```bash
# Run all unit tests (fast, no API keys needed)
uv run pytest tests/unit -v

# Run integration tests (needs API keys in .env)
uv run pytest tests/integration -v -m integration

# Run all tests
uv run pytest -v

# Run with coverage
uv run pytest --cov=voice_chat --cov-report=term-missing

# Run specific test file
uv run pytest tests/unit/test_agent_loop.py -v
```

### Test Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "integration: marks tests that require API keys (deselect with '-m not integration')",
    "slow: marks tests as slow (deselect with '-m not slow')",
]
testpaths = ["tests"]

[tool.coverage.run]
source = ["src/voice_chat"]
omit = ["*/tests/*"]
```

### Autonomous Development Workflow

**Claude works autonomously:**
1. Write code for a component
2. Write unit tests (mocked)
3. Run unit tests → fix until passing
4. Write integration test
5. Run integration test → fix until passing
6. Commit with test results

**User verifies at checkpoints:**
- End of each phase
- Review code + test coverage
- Manual testing (especially audio in Phase 3-4)

### Verification Checkpoints

| Phase | Claude Delivers | User Verifies |
|-------|-----------------|---------------|
| 0 | Project runs, tests pass | `uv run pytest` works |
| 1 | Chat works, tool tests pass | Try chatting, run a command |
| 2 | TTS tests pass | Listen to audio output |
| 3 | STT tests pass | Speak and verify transcription |
| 4 | Wake word tests pass | Test wake word activation |
| 5 | All tests pass, coverage >80% | Full conversation flow |

### Test Files Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── test_agent_loop.py
│   ├── test_tool_registry.py
│   ├── test_config.py
│   └── test_prompts.py
├── integration/
│   ├── __init__.py
│   ├── test_gemini.py
│   ├── test_elevenlabs.py   # Phase 2
│   ├── test_whisper.py      # Phase 3
│   └── test_porcupine.py    # Phase 4
└── e2e/
    ├── __init__.py
    └── test_full_flow.py    # Phase 5
```

### Fixtures (conftest.py)

```python
# tests/conftest.py
import pytest
from unittest.mock import AsyncMock

@pytest.fixture
def mock_llm():
    """Pre-configured mock LLM for unit tests"""
    llm = AsyncMock()
    llm.generate.return_value = MockResponse(text="Mock response", tool_calls=[])
    return llm

@pytest.fixture
def sample_tools():
    """Sample tool registry for testing"""
    return {
        "get_time": AsyncMock(return_value="14:00:00"),
        "execute_command": AsyncMock(return_value="command output"),
    }

@pytest.fixture
def gemini_client():
    """Real Gemini client for integration tests"""
    pytest.importorskip("google.genai")
    from voice_chat.llm.gemini import GeminiClient
    return GeminiClient()
```

### Environment for Testing

```bash
# .env.example
GEMINI_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here      # Phase 2+
PICOVOICE_ACCESS_KEY=your_key_here    # Phase 4+

# Test-specific (optional)
TEST_VERBOSE=true
TEST_SKIP_SLOW=false
```

## Open Questions

1. **Wake word** - Final choice TBD (custom via Picovoice console)
2. **TTS voice** - Test ElevenLabs Multilingual v2 voices to find preferred one
3. **STT model size** - Balance accuracy vs. latency (test `base`, `small`, `medium`, `large`)
4. **Conversation buffer** - How many turns to keep in context?
5. **Tool permissions** - Should some tools require confirmation? (e.g., `execute_command` with `rm`)
6. **Default language mode** - Start with `auto` (mirror user) or fixed language?
