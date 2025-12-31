"""Unit tests for tool registry and tools."""

import pytest

from voice_chat.tools.registry import Tool, ToolParameter, ToolRegistry


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_register_and_get_tool(self) -> None:
        """Can register and retrieve a tool."""
        registry = ToolRegistry()

        async def handler() -> str:
            return "result"

        tool = Tool(
            name="test_tool",
            description="A test tool",
            handler=handler,
        )

        registry.register(tool)

        retrieved = registry.get("test_tool")
        assert retrieved is not None
        assert retrieved.name == "test_tool"
        assert retrieved.description == "A test tool"

    def test_get_nonexistent_tool_returns_none(self) -> None:
        """Getting a nonexistent tool returns None."""
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_list_tools(self) -> None:
        """list_tools returns all registered tools."""
        registry = ToolRegistry()

        tool1 = Tool(name="tool1", description="First")
        tool2 = Tool(name="tool2", description="Second")

        registry.register(tool1)
        registry.register(tool2)

        tools = registry.list_tools()
        assert len(tools) == 2
        names = [t.name for t in tools]
        assert "tool1" in names
        assert "tool2" in names

    def test_get_schemas(self) -> None:
        """get_schemas returns schemas for all tools."""
        registry = ToolRegistry()

        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters=[
                ToolParameter(
                    name="arg1",
                    type="string",
                    description="First argument",
                    required=True,
                ),
                ToolParameter(
                    name="arg2",
                    type="integer",
                    description="Second argument",
                    required=False,
                ),
            ],
        )

        registry.register(tool)

        schemas = registry.get_schemas()
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["name"] == "test_tool"
        assert "arg1" in schema["parameters"]
        assert "arg2" in schema["parameters"]
        assert schema["parameters"]["arg1"]["required"] is True
        assert schema["parameters"]["arg2"]["required"] is False

    @pytest.mark.asyncio
    async def test_execute_tool(self) -> None:
        """Can execute a registered tool."""
        registry = ToolRegistry()

        async def handler(message: str) -> str:
            return f"Received: {message}"

        tool = Tool(
            name="echo",
            description="Echo a message",
            parameters=[
                ToolParameter(name="message", type="string", description="Message")
            ],
            handler=handler,
        )

        registry.register(tool)

        result = await registry.execute("echo", {"message": "Hello"})
        assert result == "Received: Hello"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self) -> None:
        """Executing unknown tool returns error."""
        registry = ToolRegistry()

        result = await registry.execute("unknown", {})
        assert "Error" in result
        assert "unknown" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_without_handler(self) -> None:
        """Executing tool without handler returns error."""
        registry = ToolRegistry()

        tool = Tool(name="no_handler", description="No handler")
        registry.register(tool)

        result = await registry.execute("no_handler", {})
        assert "Error" in result
        assert "no handler" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_with_invalid_args(self) -> None:
        """Executing tool with invalid args returns error."""
        registry = ToolRegistry()

        async def handler(required_arg: str) -> str:
            return required_arg

        tool = Tool(
            name="needs_arg",
            description="Needs an argument",
            parameters=[
                ToolParameter(name="required_arg", type="string", description="Required")
            ],
            handler=handler,
        )

        registry.register(tool)

        result = await registry.execute("needs_arg", {})  # Missing required arg
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_execute_tool_with_exception(self) -> None:
        """Tool that raises exception returns error message."""
        registry = ToolRegistry()

        async def failing_handler() -> str:
            raise RuntimeError("Something went wrong")

        tool = Tool(
            name="failing",
            description="Always fails",
            handler=failing_handler,
        )

        registry.register(tool)

        result = await registry.execute("failing", {})
        assert "Error" in result
        assert "Something went wrong" in result

    def test_len(self) -> None:
        """len() returns number of registered tools."""
        registry = ToolRegistry()
        assert len(registry) == 0

        registry.register(Tool(name="t1", description=""))
        assert len(registry) == 1

        registry.register(Tool(name="t2", description=""))
        assert len(registry) == 2


class TestTool:
    """Tests for Tool class."""

    def test_to_schema_simple(self) -> None:
        """to_schema works for simple tool."""
        tool = Tool(name="simple", description="A simple tool")
        schema = tool.to_schema()

        assert schema["name"] == "simple"
        assert schema["description"] == "A simple tool"
        assert schema["parameters"] == {}

    def test_to_schema_with_parameters(self) -> None:
        """to_schema includes parameters."""
        tool = Tool(
            name="with_params",
            description="Has parameters",
            parameters=[
                ToolParameter(
                    name="text",
                    type="string",
                    description="Text input",
                    required=True,
                ),
                ToolParameter(
                    name="count",
                    type="integer",
                    description="Number of times",
                    required=False,
                ),
            ],
        )

        schema = tool.to_schema()

        assert "text" in schema["parameters"]
        assert schema["parameters"]["text"]["type"] == "string"
        assert schema["parameters"]["text"]["required"] is True

        assert "count" in schema["parameters"]
        assert schema["parameters"]["count"]["type"] == "integer"
        assert schema["parameters"]["count"]["required"] is False

    def test_to_schema_with_enum(self) -> None:
        """to_schema includes enum values."""
        tool = Tool(
            name="with_enum",
            description="Has enum",
            parameters=[
                ToolParameter(
                    name="color",
                    type="string",
                    description="Color choice",
                    enum=["red", "green", "blue"],
                ),
            ],
        )

        schema = tool.to_schema()

        assert schema["parameters"]["color"]["enum"] == ["red", "green", "blue"]
