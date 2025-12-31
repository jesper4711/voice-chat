"""Time-related tools."""

from datetime import datetime
from zoneinfo import ZoneInfo

from voice_chat.tools.registry import Tool, ToolParameter


async def get_current_time(timezone: str = "local") -> str:
    """Get the current time.

    Args:
        timezone: Timezone name (e.g., 'Europe/Stockholm', 'UTC', 'local').

    Returns:
        Current time as a formatted string.
    """
    if timezone == "local":
        now = datetime.now()
        tz_name = now.astimezone().tzname()
    else:
        try:
            tz = ZoneInfo(timezone)
            now = datetime.now(tz)
            tz_name = timezone
        except Exception:
            now = datetime.now()
            tz_name = f"local (unknown timezone: {timezone})"

    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({tz_name})"


async def get_current_date() -> str:
    """Get the current date."""
    now = datetime.now()
    weekday = now.strftime("%A")
    return f"Today is {weekday}, {now.strftime('%B %d, %Y')}"


def get_time_tools() -> list[Tool]:
    """Get all time-related tools."""
    return [
        Tool(
            name="get_current_time",
            description="Get the current time. Optionally specify a timezone.",
            parameters=[
                ToolParameter(
                    name="timezone",
                    type="string",
                    description="Timezone name (e.g., 'Europe/Stockholm', 'America/New_York', 'UTC'). Use 'local' for local time.",
                    required=False,
                ),
            ],
            handler=get_current_time,
        ),
        Tool(
            name="get_current_date",
            description="Get the current date including the day of the week.",
            parameters=[],
            handler=get_current_date,
        ),
    ]
