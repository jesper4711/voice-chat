"""Tool registry system for managing available tools."""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Coroutine


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str  # "string", "integer", "boolean", "number", "array", "object"
    description: str
    required: bool = True
    enum: list[str] | None = None
    default: Any = None


@dataclass
class Tool:
    """Definition of a tool that can be called by the LLM."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable[..., Coroutine[Any, Any, str]] | None = None
    dangerous: bool = False  # If True, may require user confirmation

    def to_schema(self) -> dict[str, Any]:
        """Convert to JSON schema format for LLM."""
        params = {}
        for p in self.parameters:
            params[p.name] = {
                "type": p.type,
                "description": p.description,
                "required": p.required,
            }
            if p.enum:
                params[p.name]["enum"] = p.enum

        return {
            "name": self.name,
            "description": self.description,
            "parameters": params,
        }


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get schemas for all tools."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name with given arguments.

        Args:
            name: Tool name.
            arguments: Arguments to pass to the tool.

        Returns:
            Tool execution result as string.

        Raises:
            ValueError: If tool not found or no handler.
        """
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Unknown tool '{name}'"

        if not tool.handler:
            return f"Error: Tool '{name}' has no handler"

        try:
            result = await tool.handler(**arguments)
            return result
        except TypeError as e:
            return f"Error calling {name}: Invalid arguments - {e}"
        except Exception as e:
            return f"Error executing {name}: {e}"

    def __len__(self) -> int:
        return len(self._tools)


def tool(
    name: str,
    description: str,
    parameters: list[ToolParameter] | None = None,
    dangerous: bool = False,
) -> Callable[[Callable[..., Coroutine[Any, Any, str]]], Tool]:
    """Decorator for registering a tool.

    Example:
        @tool(
            name="get_time",
            description="Get current time",
            parameters=[],
        )
        async def get_time() -> str:
            return datetime.now().isoformat()
    """

    def decorator(func: Callable[..., Coroutine[Any, Any, str]]) -> Tool:
        return Tool(
            name=name,
            description=description,
            parameters=parameters or [],
            handler=func,
            dangerous=dangerous,
        )

    return decorator


@lru_cache
def get_default_registry() -> ToolRegistry:
    """Get the default tool registry with built-in tools."""
    from voice_chat.tools.spotify import get_spotify_tools
    from voice_chat.tools.system import get_system_tools
    from voice_chat.tools.time import get_time_tools
    from voice_chat.tools.web import get_web_tools

    registry = ToolRegistry()

    # Register all built-in tools
    for tool in get_time_tools():
        registry.register(tool)

    for tool in get_system_tools():
        registry.register(tool)

    for tool in get_spotify_tools():
        registry.register(tool)

    for tool in get_web_tools():
        registry.register(tool)

    return registry
