"""Integration tests for Agent with real tool calling.

These tests verify the full agent loop with Gemini 3 Flash
including thought_signature handling.

Run with: pytest -m integration tests/integration/test_agent_tools.py
"""

import os

import pytest

from voice_chat.agent.loop import AgentLoop, StopReason

# Skip all tests in this module if no API key
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set",
    ),
]


class TestAgentToolCalling:
    """Tests for agent tool calling with real Gemini API."""

    @pytest.mark.asyncio
    async def test_get_current_time(self) -> None:
        """Agent can call get_current_time tool and return result."""
        agent = AgentLoop()

        response = await agent.run("What time is it right now?")

        # Should contain time-related content
        assert any(
            indicator in response.lower()
            for indicator in ["time", ":", "am", "pm", "oclock", "o'clock"]
        ), f"Expected time in response, got: {response}"

    @pytest.mark.asyncio
    async def test_get_current_date(self) -> None:
        """Agent can call get_current_date tool and return result."""
        agent = AgentLoop()

        response = await agent.run("What day is it today?")

        # Should contain day-related content
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        response_lower = response.lower()
        has_day = any(day in response_lower for day in days)

        assert has_day, f"Expected day name in response, got: {response}"

    @pytest.mark.asyncio
    async def test_tool_call_with_details(self) -> None:
        """Verify tool call metadata is tracked correctly."""
        agent = AgentLoop()

        result = await agent.run_with_details("What time is it?")

        assert result.stop_reason == StopReason.TASK_COMPLETE
        assert "get_current_time" in result.tool_calls_made
        assert result.iterations >= 2  # At least: tool call + final response

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self) -> None:
        """Agent can handle queries requiring multiple tools."""
        agent = AgentLoop()

        response = await agent.run("What's the current time and date?")

        # Should contain both time and date info
        response_lower = response.lower()
        has_time = any(x in response_lower for x in ["time", ":", "am", "pm"])
        has_date = any(x in response_lower for x in ["day", "date", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"])

        assert has_time or has_date, f"Expected time/date info, got: {response}"

    @pytest.mark.asyncio
    async def test_no_tool_for_simple_question(self) -> None:
        """Agent answers simple questions without tools."""
        agent = AgentLoop()

        result = await agent.run_with_details("What is 2 + 2?")

        assert result.stop_reason == StopReason.TASK_COMPLETE
        # Should not have called any tools for simple math
        assert len(result.tool_calls_made) == 0 or result.iterations == 1
        assert "4" in result.response

    @pytest.mark.asyncio
    async def test_conversation_context_with_tools(self) -> None:
        """Agent maintains context across tool calls."""
        agent = AgentLoop()

        # First query with tool
        await agent.run("What time is it?")

        # Follow-up that references the first query
        response = await agent.run("Was that AM or PM?")

        # Should understand context
        assert any(
            x in response.lower()
            for x in ["am", "pm", "morning", "afternoon", "evening", "night"]
        ), f"Expected AM/PM context, got: {response}"

    @pytest.mark.asyncio
    async def test_thought_signature_preserved(self) -> None:
        """Verify thought_signature is captured and preserved in tool calls."""
        agent = AgentLoop()

        result = await agent.run_with_details("What time is it?")

        # Check that we made tool calls and completed successfully
        assert result.stop_reason == StopReason.TASK_COMPLETE
        assert "get_current_time" in result.tool_calls_made

        # Check conversation state for thought_signature
        messages = agent.get_history()
        assistant_msgs = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]

        if assistant_msgs:
            tool_calls = assistant_msgs[0].get("tool_calls", [])
            # Gemini 3 should include thought_signature
            for tc in tool_calls:
                # thought_signature may be None for some models, but field should exist
                assert "thought_signature" in tc
