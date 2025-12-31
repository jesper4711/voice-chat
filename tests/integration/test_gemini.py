"""Integration tests for Gemini API.

These tests require GEMINI_API_KEY to be set in environment.
Run with: pytest -m integration
"""

import os

import pytest
from pydantic import BaseModel, Field

from voice_chat.llm.gemini import GeminiClient

# Skip all tests in this module if no API key
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set",
    ),
]


class TestGeminiBasic:
    """Basic Gemini API tests."""

    @pytest.mark.asyncio
    async def test_simple_chat(self) -> None:
        """Verify Gemini API connection and basic response."""
        client = GeminiClient()

        response = await client.generate(
            messages=[{"role": "user", "content": "Say 'test successful' and nothing else."}],
        )

        assert response.text
        assert "test" in response.text.lower()
        assert response.usage is not None
        assert response.usage.get("total_tokens", 0) > 0

    @pytest.mark.asyncio
    async def test_conversation_context(self) -> None:
        """Verify conversation context is maintained."""
        client = GeminiClient()

        messages = [
            {"role": "user", "content": "My name is TestUser123."},
            {"role": "assistant", "content": "Nice to meet you, TestUser123!"},
            {"role": "user", "content": "What is my name?"},
        ]

        response = await client.generate(messages=messages)

        assert "TestUser123" in response.text

    @pytest.mark.asyncio
    async def test_system_prompt(self) -> None:
        """Verify system prompt is used."""
        client = GeminiClient()

        response = await client.generate(
            messages=[{"role": "user", "content": "What language should you respond in?"}],
            system_prompt="You must always respond in Swedish. Svara alltid på svenska.",
        )

        # Check for common Swedish words
        swedish_indicators = ["på", "och", "att", "är", "jag", "du", "ska", "svenska"]
        text_lower = response.text.lower()
        has_swedish = any(word in text_lower for word in swedish_indicators)

        assert has_swedish, f"Expected Swedish response, got: {response.text}"


class TestGeminiToolCalling:
    """Tests for Gemini tool calling."""

    @pytest.mark.asyncio
    async def test_tool_calling_triggers(self) -> None:
        """Verify LLM requests tool call when appropriate."""
        client = GeminiClient()

        tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "city": {
                        "type": "string",
                        "description": "The city name",
                        "required": True,
                    }
                },
            }
        ]

        response = await client.generate(
            messages=[{"role": "user", "content": "What's the weather in Stockholm?"}],
            tools=tools,
        )

        assert response.tool_calls, "Expected tool call but got none"
        assert len(response.tool_calls) >= 1
        assert response.tool_calls[0].name == "get_weather"
        assert "stockholm" in response.tool_calls[0].arguments.get("city", "").lower()

    @pytest.mark.asyncio
    async def test_no_tool_call_when_unnecessary(self) -> None:
        """Verify LLM doesn't call tools for simple questions."""
        client = GeminiClient()

        tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "city": {"type": "string", "description": "City", "required": True}
                },
            }
        ]

        response = await client.generate(
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
            tools=tools,
        )

        # Should answer directly without tool call
        assert not response.tool_calls or len(response.tool_calls) == 0
        assert "4" in response.text

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self) -> None:
        """Verify LLM can request multiple tools."""
        client = GeminiClient()

        tools = [
            {
                "name": "get_time",
                "description": "Get current time in a timezone",
                "parameters": {
                    "timezone": {"type": "string", "description": "Timezone", "required": True}
                },
            },
            {
                "name": "get_date",
                "description": "Get today's date",
                "parameters": {},
            },
        ]

        response = await client.generate(
            messages=[{"role": "user", "content": "What time is it in Tokyo and what's today's date?"}],
            tools=tools,
            system_prompt="Use the available tools to answer. Call multiple tools if needed.",
        )

        # Should call at least one tool (ideally both)
        assert response.tool_calls
        tool_names = [tc.name for tc in response.tool_calls]
        assert "get_time" in tool_names or "get_date" in tool_names


class TestGeminiStructuredOutput:
    """Tests for structured output with Pydantic schemas."""

    @pytest.mark.asyncio
    async def test_simple_structured_output(self) -> None:
        """Verify Pydantic schema enforcement works."""

        class SimpleResponse(BaseModel):
            message: str = Field(description="A greeting message")
            number: int = Field(description="A number between 1 and 100")

        client = GeminiClient()

        response = await client.generate(
            messages=[
                {"role": "user", "content": "Return a greeting and the number 42"}
            ],
            response_schema=SimpleResponse,
        )

        # Should be valid JSON that parses to our model
        parsed = SimpleResponse.model_validate_json(response.text)
        assert parsed.number == 42
        assert len(parsed.message) > 0

    @pytest.mark.asyncio
    async def test_structured_output_with_list(self) -> None:
        """Verify structured output works with lists."""

        class Item(BaseModel):
            name: str
            quantity: int

        class ShoppingList(BaseModel):
            items: list[Item] = Field(description="List of items to buy")
            store: str = Field(description="Store name")

        client = GeminiClient()

        response = await client.generate(
            messages=[
                {
                    "role": "user",
                    "content": "Create a shopping list with 3 items for a grocery store called 'FreshMart'",
                }
            ],
            response_schema=ShoppingList,
        )

        parsed = ShoppingList.model_validate_json(response.text)
        assert len(parsed.items) == 3
        assert parsed.store == "FreshMart"

    @pytest.mark.asyncio
    async def test_structured_output_with_enum(self) -> None:
        """Verify structured output works with enums."""
        from enum import Enum

        class Priority(str, Enum):
            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"

        class Task(BaseModel):
            title: str = Field(description="Task title")
            priority: Priority = Field(description="Task priority level")

        client = GeminiClient()

        response = await client.generate(
            messages=[
                {"role": "user", "content": "Create a high priority task called 'Fix critical bug'"}
            ],
            response_schema=Task,
        )

        parsed = Task.model_validate_json(response.text)
        assert parsed.title == "Fix critical bug"
        assert parsed.priority == Priority.HIGH

    @pytest.mark.asyncio
    async def test_structured_output_optional_fields(self) -> None:
        """Verify structured output handles optional fields."""
        from typing import Optional

        class Person(BaseModel):
            name: str = Field(description="Person's name")
            age: Optional[int] = Field(default=None, description="Person's age if known")
            occupation: Optional[str] = Field(default=None, description="Person's job if known")

        client = GeminiClient()

        response = await client.generate(
            messages=[
                {"role": "user", "content": "Create a person named 'Alice' who is 30 years old (occupation unknown)"}
            ],
            response_schema=Person,
        )

        parsed = Person.model_validate_json(response.text)
        assert parsed.name == "Alice"
        assert parsed.age == 30
        # Occupation should be None or missing

    @pytest.mark.asyncio
    async def test_structured_output_nested_models(self) -> None:
        """Verify structured output works with nested Pydantic models."""

        class Address(BaseModel):
            street: str
            city: str
            country: str

        class Company(BaseModel):
            name: str = Field(description="Company name")
            address: Address = Field(description="Company headquarters address")
            employee_count: int = Field(description="Number of employees")

        client = GeminiClient()

        response = await client.generate(
            messages=[
                {
                    "role": "user",
                    "content": "Create a company called 'TechCorp' located at '123 Main St, Stockholm, Sweden' with 100 employees",
                }
            ],
            response_schema=Company,
        )

        parsed = Company.model_validate_json(response.text)
        assert parsed.name == "TechCorp"
        assert parsed.address.city == "Stockholm"
        assert parsed.address.country == "Sweden"
        assert parsed.employee_count == 100


class TestGeminiEdgeCases:
    """Edge case and error handling tests."""

    @pytest.mark.asyncio
    async def test_empty_message(self) -> None:
        """Verify handling of empty message."""
        client = GeminiClient()

        response = await client.generate(
            messages=[{"role": "user", "content": ""}],
        )

        # Should still return something (even if confused)
        assert response is not None

    @pytest.mark.asyncio
    async def test_very_long_message(self) -> None:
        """Verify handling of long messages."""
        client = GeminiClient()

        long_text = "Hello. " * 1000  # About 7000 chars

        response = await client.generate(
            messages=[{"role": "user", "content": f"Summarize this: {long_text}"}],
        )

        assert response.text
        assert response.usage is not None
