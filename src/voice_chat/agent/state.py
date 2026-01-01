"""Conversation state management."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class MessageRole(str, Enum):
    """Role of a message in the conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    thought_signature: str | None = None  # Required for Gemini 3


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""

    tool_call_id: str
    name: str
    result: str
    success: bool = True


@dataclass
class Message:
    """A message in the conversation."""

    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None


@dataclass
class ConversationState:
    """Maintains the state of a conversation."""

    session_id: UUID = field(default_factory=uuid4)
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)

    # Memory settings (can be overridden)
    max_turns: int = 20  # 1 turn = user + assistant exchange
    inactivity_timeout_seconds: float = 1800  # 30 minutes

    def _check_inactivity_timeout(self) -> None:
        """Clear history if inactivity timeout exceeded."""
        elapsed = (datetime.now() - self.last_interaction).total_seconds()
        if elapsed > self.inactivity_timeout_seconds and self.messages:
            self.clear()

    def _trim_to_max_turns(self) -> None:
        """Trim messages to keep only the last max_turns."""
        if self.max_turns <= 0:
            return

        # Count turns (user messages)
        user_count = sum(1 for m in self.messages if m.role == MessageRole.USER)

        if user_count <= self.max_turns:
            return

        # Find how many turns to remove
        turns_to_remove = user_count - self.max_turns

        # Remove oldest turns (user + following assistant/tool messages)
        removed_turns = 0
        new_messages: list[Message] = []
        skip_until_next_user = True  # Skip messages until we've removed enough turns

        for msg in self.messages:
            if msg.role == MessageRole.USER:
                if removed_turns < turns_to_remove:
                    removed_turns += 1
                    skip_until_next_user = True
                    continue
                else:
                    skip_until_next_user = False

            if not skip_until_next_user:
                new_messages.append(msg)

        self.messages = new_messages

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self._check_inactivity_timeout()
        self.last_interaction = datetime.now()
        self.messages.append(Message(role=MessageRole.USER, content=content))
        self._trim_to_max_turns()

    def add_assistant_message(
        self,
        content: str,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        """Add an assistant message to the conversation."""
        self.messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=tool_calls,
            )
        )

    def add_tool_results(self, results: list[ToolResult]) -> None:
        """Add tool results to the conversation."""
        self.messages.append(
            Message(
                role=MessageRole.TOOL,
                content="",
                tool_results=results,
            )
        )

    def get_messages_for_llm(self) -> list[dict[str, Any]]:
        """Convert messages to format suitable for LLM API."""
        result = []
        for msg in self.messages:
            if msg.role == MessageRole.USER:
                result.append({"role": "user", "content": msg.content})
            elif msg.role == MessageRole.ASSISTANT:
                if msg.tool_calls:
                    # Assistant message with tool calls
                    result.append({
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "name": tc.name,
                                "args": tc.arguments,
                                "id": tc.id,
                                "thought_signature": tc.thought_signature,
                            }
                            for tc in msg.tool_calls
                        ],
                    })
                else:
                    result.append({"role": "assistant", "content": msg.content})
            elif msg.role == MessageRole.TOOL and msg.tool_results:
                for tr in msg.tool_results:
                    result.append({
                        "role": "tool",
                        "tool_call_id": tr.tool_call_id,
                        "name": tr.name,
                        "content": tr.result,
                    })
        return result

    def clear(self) -> None:
        """Clear all messages from the conversation."""
        self.messages.clear()

    @property
    def turn_count(self) -> int:
        """Count the number of user turns in the conversation."""
        return sum(1 for m in self.messages if m.role == MessageRole.USER)
