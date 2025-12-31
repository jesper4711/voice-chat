"""Main agent loop for processing user requests."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from voice_chat.agent.prompts import get_system_prompt
from voice_chat.agent.state import ConversationState, ToolCall, ToolResult
from voice_chat.config import get_settings
from voice_chat.llm.gemini import GeminiClient, LLMResponse
from voice_chat.tools.registry import ToolRegistry, get_default_registry


class StopReason(str, Enum):
    """Reason why the agent loop stopped."""

    TASK_COMPLETE = "task_complete"
    MAX_ITERATIONS = "max_iterations"
    ERROR = "error"


@dataclass
class AgentResult:
    """Result of running the agent loop."""

    response: str
    stop_reason: StopReason
    iterations: int
    tool_calls_made: list[str]


class AgentLoop:
    """Main agent loop that processes user requests through LLM and tools."""

    def __init__(
        self,
        llm: GeminiClient | None = None,
        tools: ToolRegistry | None = None,
        max_iterations: int | None = None,
    ) -> None:
        """Initialize the agent loop.

        Args:
            llm: LLM client to use. If None, creates default GeminiClient.
            tools: Tool registry to use. If None, uses default registry.
            max_iterations: Maximum loop iterations. If None, uses settings.
        """
        settings = get_settings()
        self.llm = llm or GeminiClient()
        self.tools = tools or get_default_registry()
        self.max_iterations = max_iterations or settings.max_agent_iterations
        self.state = ConversationState()
        self.system_prompt = get_system_prompt(settings.language_mode)

    async def run(self, user_message: str) -> str:
        """Run the agent loop for a user message.

        Args:
            user_message: The user's input message.

        Returns:
            The assistant's final response.
        """
        result = await self.run_with_details(user_message)
        return result.response

    async def run_with_details(self, user_message: str) -> AgentResult:
        """Run the agent loop and return detailed result.

        Args:
            user_message: The user's input message.

        Returns:
            AgentResult with response and metadata.
        """
        # Add user message to state
        self.state.add_user_message(user_message)

        tool_calls_made: list[str] = []
        iterations = 0

        for iterations in range(1, self.max_iterations + 1):
            # Get LLM response
            response = await self._get_llm_response()

            # Check if LLM wants to call tools
            if response.tool_calls:
                # Execute tools and collect results
                results = await self._execute_tools(response.tool_calls)
                tool_calls_made.extend(tc.name for tc in response.tool_calls)

                # Add assistant message with tool calls
                self.state.add_assistant_message(
                    content=response.text,
                    tool_calls=response.tool_calls,
                )

                # Add tool results
                self.state.add_tool_results(results)

                # Continue loop to get LLM's interpretation of results
                continue

            # No tool calls - task complete
            self.state.add_assistant_message(content=response.text)
            return AgentResult(
                response=response.text,
                stop_reason=StopReason.TASK_COMPLETE,
                iterations=iterations,
                tool_calls_made=tool_calls_made,
            )

        # Max iterations reached
        return AgentResult(
            response="I apologize, but I wasn't able to complete this task. Let me know if you'd like me to try a different approach.",
            stop_reason=StopReason.MAX_ITERATIONS,
            iterations=iterations,
            tool_calls_made=tool_calls_made,
        )

    async def _get_llm_response(self) -> LLMResponse:
        """Get response from LLM with current conversation state."""
        messages = self.state.get_messages_for_llm()
        tool_schemas = self.tools.get_schemas()

        return await self.llm.generate(
            messages=messages,
            system_prompt=self.system_prompt,
            tools=tool_schemas if tool_schemas else None,
        )

    async def _execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute a list of tool calls and return results."""
        results = []

        for tc in tool_calls:
            result_str = await self.tools.execute(tc.name, tc.arguments)
            results.append(
                ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    result=result_str,
                    success=not result_str.startswith("Error"),
                )
            )

        return results

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.state.clear()

    def get_history(self) -> list[dict[str, Any]]:
        """Get conversation history in serializable format."""
        return self.state.get_messages_for_llm()
