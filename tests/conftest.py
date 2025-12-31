"""Pytest configuration and fixtures."""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import pytest

from voice_chat.agent.state import ToolCall
from voice_chat.llm.gemini import LLMResponse
from voice_chat.tools.registry import Tool, ToolParameter, ToolRegistry


@dataclass
class MockLLMResponse:
    """Mock LLM response for testing."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "STOP"
    usage: dict[str, int] | None = None


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Pre-configured mock LLM for unit tests."""
    llm = AsyncMock()
    llm.generate.return_value = LLMResponse(
        text="Mock response",
        tool_calls=[],
        finish_reason="STOP",
    )
    return llm


@pytest.fixture
def mock_llm_with_tool_call() -> AsyncMock:
    """Mock LLM that requests a tool call then returns final response."""
    llm = AsyncMock()
    llm.generate.side_effect = [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(name="get_current_time", arguments={})],
        ),
        LLMResponse(
            text="The current time is 14:00",
            tool_calls=[],
        ),
    ]
    return llm


@pytest.fixture
def sample_tools() -> ToolRegistry:
    """Sample tool registry for testing."""
    registry = ToolRegistry()

    async def mock_get_time(timezone: str = "local") -> str:
        return "14:00:00 (mock)"

    async def mock_execute_command(command: str, timeout: int = 30) -> str:
        return f"Executed: {command}"

    registry.register(
        Tool(
            name="get_current_time",
            description="Get the current time",
            parameters=[
                ToolParameter(
                    name="timezone",
                    type="string",
                    description="Timezone",
                    required=False,
                )
            ],
            handler=mock_get_time,
        )
    )

    registry.register(
        Tool(
            name="execute_command",
            description="Execute a shell command",
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    description="Command to execute",
                    required=True,
                )
            ],
            handler=mock_execute_command,
        )
    )

    return registry


@pytest.fixture
def empty_tools() -> ToolRegistry:
    """Empty tool registry for testing."""
    return ToolRegistry()


# Skip markers for different test types
def pytest_configure(config: Any) -> None:
    """Configure custom markers."""
    config.addinivalue_line("markers", "integration: marks tests that require API keys")
    config.addinivalue_line("markers", "slow: marks tests as slow")
