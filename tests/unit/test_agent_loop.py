"""Unit tests for agent loop."""

import pytest
from unittest.mock import AsyncMock

from voice_chat.agent.loop import AgentLoop, StopReason
from voice_chat.agent.state import ToolCall
from voice_chat.llm.gemini import LLMResponse
from voice_chat.tools.registry import ToolRegistry


class TestAgentLoop:
    """Tests for AgentLoop class."""

    @pytest.mark.asyncio
    async def test_simple_response_no_tools(self, mock_llm: AsyncMock, empty_tools: ToolRegistry) -> None:
        """LLM returns text without tool calls → returns response directly."""
        agent = AgentLoop(llm=mock_llm, tools=empty_tools, max_iterations=5)

        result = await agent.run_with_details("Hello")

        assert result.response == "Mock response"
        assert result.stop_reason == StopReason.TASK_COMPLETE
        assert result.iterations == 1
        assert result.tool_calls_made == []
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_call_and_response(
        self, mock_llm_with_tool_call: AsyncMock, sample_tools: ToolRegistry
    ) -> None:
        """LLM requests tool → executes tool → gets final response."""
        agent = AgentLoop(llm=mock_llm_with_tool_call, tools=sample_tools, max_iterations=5)

        result = await agent.run_with_details("What time is it?")

        assert "14:00" in result.response
        assert result.stop_reason == StopReason.TASK_COMPLETE
        assert result.iterations == 2
        assert "get_current_time" in result.tool_calls_made
        assert mock_llm_with_tool_call.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_max_iterations_reached(
        self, sample_tools: ToolRegistry
    ) -> None:
        """Agent stops after max iterations with tool call loop."""
        llm = AsyncMock()
        # LLM always returns a tool call, never a final response
        llm.generate.return_value = LLMResponse(
            text="",
            tool_calls=[ToolCall(name="get_current_time", arguments={})],
        )

        agent = AgentLoop(llm=llm, tools=sample_tools, max_iterations=3)

        result = await agent.run_with_details("Keep checking the time")

        assert result.stop_reason == StopReason.MAX_ITERATIONS
        assert result.iterations == 3
        assert llm.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(
        self, empty_tools: ToolRegistry
    ) -> None:
        """Unknown tool call returns error in tool result."""
        llm = AsyncMock()
        llm.generate.side_effect = [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(name="nonexistent_tool", arguments={})],
            ),
            LLMResponse(
                text="I couldn't find that tool",
                tool_calls=[],
            ),
        ]

        agent = AgentLoop(llm=llm, tools=empty_tools, max_iterations=5)

        result = await agent.run_with_details("Use a fake tool")

        assert result.stop_reason == StopReason.TASK_COMPLETE
        assert "nonexistent_tool" in result.tool_calls_made

    @pytest.mark.asyncio
    async def test_conversation_history_maintained(
        self, mock_llm: AsyncMock, empty_tools: ToolRegistry
    ) -> None:
        """Conversation history is maintained across calls."""
        agent = AgentLoop(llm=mock_llm, tools=empty_tools)

        await agent.run("First message")
        await agent.run("Second message")

        history = agent.get_history()
        user_messages = [m for m in history if m.get("role") == "user"]
        assert len(user_messages) == 2
        assert user_messages[0]["content"] == "First message"
        assert user_messages[1]["content"] == "Second message"

    @pytest.mark.asyncio
    async def test_clear_history(
        self, mock_llm: AsyncMock, empty_tools: ToolRegistry
    ) -> None:
        """clear_history() removes all messages."""
        agent = AgentLoop(llm=mock_llm, tools=empty_tools)

        await agent.run("Test message")
        assert len(agent.get_history()) > 0

        agent.clear_history()
        assert len(agent.get_history()) == 0

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_response(
        self, sample_tools: ToolRegistry
    ) -> None:
        """LLM can request multiple tools in one response."""
        llm = AsyncMock()
        llm.generate.side_effect = [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(name="get_current_time", arguments={}),
                    ToolCall(name="execute_command", arguments={"command": "echo test"}),
                ],
            ),
            LLMResponse(
                text="Here's the time and command output",
                tool_calls=[],
            ),
        ]

        agent = AgentLoop(llm=llm, tools=sample_tools, max_iterations=5)

        result = await agent.run_with_details("Time and run echo")

        assert result.stop_reason == StopReason.TASK_COMPLETE
        assert len(result.tool_calls_made) == 2
        assert "get_current_time" in result.tool_calls_made
        assert "execute_command" in result.tool_calls_made
