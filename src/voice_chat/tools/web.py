"""Web search tools using Tavily API."""

from voice_chat.config import get_settings
from voice_chat.tools.registry import Tool, ToolParameter


async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information.

    Args:
        query: Search query.
        max_results: Maximum number of results (1-10).

    Returns:
        Search results formatted as text.
    """
    from tavily import TavilyClient

    settings = get_settings()

    if not settings.tavily_api_key:
        return "Error: TAVILY_API_KEY not set. Get one free at https://app.tavily.com"

    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query,
            max_results=min(max_results, 10),
            include_answer=True,
        )

        # Build response
        parts = []

        # Include AI-generated answer if available
        if response.get("answer"):
            parts.append(f"Summary: {response['answer']}\n")

        # Include top results
        parts.append("Sources:")
        for r in response.get("results", []):
            title = r.get("title", "No title")
            content = r.get("content", "")[:200]
            url = r.get("url", "")
            parts.append(f"- {title}: {content}...")
            parts.append(f"  URL: {url}")

        return "\n".join(parts)

    except Exception as e:
        return f"Error searching web: {e}"


async def get_webpage_content(url: str) -> str:
    """Get the content of a specific webpage.

    Args:
        url: URL of the webpage to fetch.

    Returns:
        Webpage content as text.
    """
    from tavily import TavilyClient

    settings = get_settings()

    if not settings.tavily_api_key:
        return "Error: TAVILY_API_KEY not set"

    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.extract(urls=[url])

        if response.get("results"):
            result = response["results"][0]
            content = result.get("raw_content", "")[:3000]
            return f"Content from {url}:\n\n{content}"
        else:
            return f"Could not extract content from {url}"

    except Exception as e:
        return f"Error fetching webpage: {e}"


def get_web_tools() -> list[Tool]:
    """Get all web search tools."""
    return [
        Tool(
            name="web_search",
            description=(
                "Search the web for current information. Use this for questions about "
                "recent events, facts you're unsure about, or anything requiring up-to-date information."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The search query",
                    required=True,
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results (1-10, default 5)",
                    required=False,
                ),
            ],
            handler=web_search,
        ),
        Tool(
            name="get_webpage_content",
            description="Fetch and read the content of a specific webpage URL.",
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="The URL of the webpage to read",
                    required=True,
                ),
            ],
            handler=get_webpage_content,
        ),
    ]
