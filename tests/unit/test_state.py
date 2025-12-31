"""Unit tests for conversation state."""

import pytest

from voice_chat.agent.state import (
    ConversationState,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
)


class TestConversationState:
    """Tests for ConversationState class."""

    def test_new_state_is_empty(self) -> None:
        """New conversation state has no messages."""
        state = ConversationState()
        assert len(state.messages) == 0
        assert state.turn_count == 0

    def test_add_user_message(self) -> None:
        """Can add user messages."""
        state = ConversationState()
        state.add_user_message("Hello")

        assert len(state.messages) == 1
        assert state.messages[0].role == MessageRole.USER
        assert state.messages[0].content == "Hello"
        assert state.turn_count == 1

    def test_add_assistant_message(self) -> None:
        """Can add assistant messages."""
        state = ConversationState()
        state.add_assistant_message("Hi there!")

        assert len(state.messages) == 1
        assert state.messages[0].role == MessageRole.ASSISTANT
        assert state.messages[0].content == "Hi there!"
        assert state.turn_count == 0  # Only user messages count as turns

    def test_add_assistant_message_with_tool_calls(self) -> None:
        """Can add assistant message with tool calls."""
        state = ConversationState()
        tool_calls = [
            ToolCall(name="get_time", arguments={}),
            ToolCall(name="get_date", arguments={}),
        ]
        state.add_assistant_message("Let me check", tool_calls=tool_calls)

        assert len(state.messages) == 1
        assert state.messages[0].tool_calls is not None
        assert len(state.messages[0].tool_calls) == 2

    def test_add_tool_results(self) -> None:
        """Can add tool results."""
        state = ConversationState()
        results = [
            ToolResult(tool_call_id="1", name="get_time", result="14:00"),
            ToolResult(tool_call_id="2", name="get_date", result="2024-01-01"),
        ]
        state.add_tool_results(results)

        assert len(state.messages) == 1
        assert state.messages[0].role == MessageRole.TOOL
        assert state.messages[0].tool_results is not None
        assert len(state.messages[0].tool_results) == 2

    def test_get_messages_for_llm_user_messages(self) -> None:
        """get_messages_for_llm formats user messages correctly."""
        state = ConversationState()
        state.add_user_message("Hello")
        state.add_user_message("How are you?")

        messages = state.get_messages_for_llm()

        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "user", "content": "How are you?"}

    def test_get_messages_for_llm_assistant_messages(self) -> None:
        """get_messages_for_llm formats assistant messages correctly."""
        state = ConversationState()
        state.add_assistant_message("Hello!")

        messages = state.get_messages_for_llm()

        assert len(messages) == 1
        assert messages[0] == {"role": "assistant", "content": "Hello!"}

    def test_get_messages_for_llm_with_tool_calls(self) -> None:
        """get_messages_for_llm formats tool calls correctly."""
        state = ConversationState()
        tool_call = ToolCall(name="get_time", arguments={"tz": "UTC"}, id="abc123")
        state.add_assistant_message("Checking time", tool_calls=[tool_call])

        messages = state.get_messages_for_llm()

        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Checking time"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["name"] == "get_time"
        assert msg["tool_calls"][0]["args"] == {"tz": "UTC"}

    def test_get_messages_for_llm_with_tool_results(self) -> None:
        """get_messages_for_llm formats tool results correctly."""
        state = ConversationState()
        result = ToolResult(tool_call_id="abc123", name="get_time", result="14:00")
        state.add_tool_results([result])

        messages = state.get_messages_for_llm()

        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "abc123"
        assert msg["name"] == "get_time"
        assert msg["content"] == "14:00"

    def test_clear(self) -> None:
        """clear() removes all messages."""
        state = ConversationState()
        state.add_user_message("Hello")
        state.add_assistant_message("Hi")

        assert len(state.messages) == 2

        state.clear()

        assert len(state.messages) == 0
        assert state.turn_count == 0

    def test_turn_count_only_counts_user_messages(self) -> None:
        """turn_count only counts user messages."""
        state = ConversationState()
        state.add_user_message("1")
        state.add_assistant_message("response 1")
        state.add_user_message("2")
        state.add_assistant_message("response 2")
        state.add_user_message("3")

        assert state.turn_count == 3

    def test_session_id_is_unique(self) -> None:
        """Each conversation has a unique session ID."""
        state1 = ConversationState()
        state2 = ConversationState()

        assert state1.session_id != state2.session_id


class TestMessage:
    """Tests for Message class."""

    def test_message_has_timestamp(self) -> None:
        """Messages have a timestamp."""
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.timestamp is not None

    def test_message_defaults(self) -> None:
        """Message has sensible defaults."""
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.tool_calls is None
        assert msg.tool_results is None


class TestToolCall:
    """Tests for ToolCall class."""

    def test_tool_call_has_id(self) -> None:
        """Tool calls have an auto-generated ID."""
        tc1 = ToolCall(name="test", arguments={})
        tc2 = ToolCall(name="test", arguments={})

        assert tc1.id is not None
        assert tc2.id is not None
        assert tc1.id != tc2.id

    def test_tool_call_explicit_id(self) -> None:
        """Can provide explicit ID."""
        tc = ToolCall(name="test", arguments={}, id="custom-id")
        assert tc.id == "custom-id"


class TestToolResult:
    """Tests for ToolResult class."""

    def test_tool_result_success_default(self) -> None:
        """Tool result defaults to success=True."""
        result = ToolResult(tool_call_id="1", name="test", result="ok")
        assert result.success is True

    def test_tool_result_failure(self) -> None:
        """Can mark tool result as failure."""
        result = ToolResult(tool_call_id="1", name="test", result="error", success=False)
        assert result.success is False
