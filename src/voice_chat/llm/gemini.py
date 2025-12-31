"""Gemini LLM client using google-genai SDK."""

from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel

from voice_chat.agent.state import ToolCall
from voice_chat.config import get_settings


@dataclass
class LLMResponse:
    """Response from the LLM."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


class GeminiClient:
    """Client for Google Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize Gemini client.

        Args:
            api_key: Gemini API key. If None, uses settings.
            model: Model name. If None, uses settings.
        """
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")

        self.client = genai.Client(api_key=self.api_key)

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: Conversation history in standard format.
            system_prompt: System instructions for the model.
            tools: Tool definitions for function calling.
            response_schema: Pydantic model for structured output.

        Returns:
            LLMResponse with text and/or tool calls.
        """
        # Build contents from messages
        contents = self._build_contents(messages)

        # Build config
        config: dict[str, Any] = {}
        if system_prompt:
            config["system_instruction"] = system_prompt

        if response_schema:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = response_schema

        # Build tools if provided
        if tools:
            config["tools"] = self._build_tools(tools)

        # Make the API call
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config) if config else None,
        )

        return self._parse_response(response)

    def _build_contents(
        self,
        messages: list[dict[str, Any]],
    ) -> list[types.Content]:
        """Convert messages to Gemini content format."""
        contents: list[types.Content] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Map roles to Gemini format
            if role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=content)],
                    )
                )
            elif role == "assistant":
                parts: list[types.Part] = []
                if content:
                    parts.append(types.Part(text=content))

                # Handle tool calls from assistant
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                name=tc["name"],
                                args=tc["args"],
                            )
                        )
                    )

                contents.append(types.Content(role="model", parts=parts))
            elif role == "tool":
                # Tool results
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=msg.get("name", "unknown"),
                                    response={"result": content},
                                )
                            )
                        ],
                    )
                )

        return contents

    def _build_tools(
        self,
        tools: list[dict[str, Any]],
    ) -> list[types.Tool]:
        """Convert tool definitions to Gemini format."""
        function_declarations = []

        for tool in tools:
            # Build parameters schema
            params = tool.get("parameters", {})
            properties = {}
            required = []

            for param_name, param_def in params.items():
                prop: dict[str, Any] = {"type": param_def.get("type", "string").upper()}
                if "description" in param_def:
                    prop["description"] = param_def["description"]
                if "enum" in param_def:
                    prop["enum"] = param_def["enum"]
                properties[param_name] = prop

                if param_def.get("required", False):
                    required.append(param_name)

            parameters: dict[str, Any] = {
                "type": "OBJECT",
                "properties": properties,
            }
            if required:
                parameters["required"] = required

            function_declarations.append(
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=parameters if properties else None,
                )
            )

        return [types.Tool(function_declarations=function_declarations)]

    def _parse_response(self, response: types.GenerateContentResponse) -> LLMResponse:
        """Parse Gemini response into LLMResponse."""
        text = ""
        tool_calls: list[ToolCall] = []

        if response.candidates:
            candidate = response.candidates[0]

            for part in candidate.content.parts:
                if part.text:
                    text += part.text
                elif part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        )
                    )

        # Get usage stats if available
        usage = None
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                "total_tokens": response.usage_metadata.total_token_count or 0,
            }

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=candidate.finish_reason.name if candidate else None,
            usage=usage,
        )


async def generate_simple(prompt: str, system_prompt: str | None = None) -> str:
    """Simple helper for one-off generations."""
    client = GeminiClient()
    response = await client.generate(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=system_prompt,
    )
    return response.text
