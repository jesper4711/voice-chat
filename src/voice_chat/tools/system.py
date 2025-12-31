"""System control tools."""

import asyncio
import subprocess
import sys

from voice_chat.tools.registry import Tool, ToolParameter


async def execute_command(command: str, timeout: int = 30) -> str:
    """Execute a shell command.

    Args:
        command: The shell command to execute.
        timeout: Timeout in seconds.

    Returns:
        Command output (stdout + stderr).
    """
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )

        output = ""
        if stdout:
            output += stdout.decode("utf-8", errors="replace")
        if stderr:
            if output:
                output += "\n"
            output += stderr.decode("utf-8", errors="replace")

        if not output.strip():
            output = f"Command completed with exit code {process.returncode}"

        # Truncate very long outputs
        if len(output) > 5000:
            output = output[:5000] + "\n... (output truncated)"

        return output.strip()

    except asyncio.TimeoutError:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {e}"


async def open_application(app_name: str) -> str:
    """Open an application (macOS only).

    Args:
        app_name: Name of the application to open.

    Returns:
        Result message.
    """
    if sys.platform != "darwin":
        return "Error: open_application is only supported on macOS"

    try:
        # Try opening by name
        result = subprocess.run(
            ["open", "-a", app_name],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            return f"Opened {app_name}"
        else:
            return f"Error: Could not open {app_name}. {result.stderr.strip()}"

    except subprocess.TimeoutExpired:
        return "Error: Timed out opening application"
    except Exception as e:
        return f"Error: {e}"


async def control_volume(action: str, level: int | None = None) -> str:
    """Control system volume (macOS only).

    Args:
        action: One of 'get', 'set', 'mute', 'unmute'.
        level: Volume level 0-100 (only for 'set' action).

    Returns:
        Result message.
    """
    if sys.platform != "darwin":
        return "Error: control_volume is only supported on macOS"

    try:
        if action == "get":
            result = subprocess.run(
                ["osascript", "-e", "output volume of (get volume settings)"],
                capture_output=True,
                text=True,
            )
            volume = result.stdout.strip()
            return f"Current volume: {volume}%"

        elif action == "set":
            if level is None:
                return "Error: 'level' is required for 'set' action"
            level = max(0, min(100, level))
            subprocess.run(
                ["osascript", "-e", f"set volume output volume {level}"],
                capture_output=True,
            )
            return f"Volume set to {level}%"

        elif action == "mute":
            subprocess.run(
                ["osascript", "-e", "set volume output muted true"],
                capture_output=True,
            )
            return "Volume muted"

        elif action == "unmute":
            subprocess.run(
                ["osascript", "-e", "set volume output muted false"],
                capture_output=True,
            )
            return "Volume unmuted"

        else:
            return f"Error: Unknown action '{action}'. Use 'get', 'set', 'mute', or 'unmute'."

    except Exception as e:
        return f"Error controlling volume: {e}"


def get_system_tools() -> list[Tool]:
    """Get all system tools."""
    return [
        Tool(
            name="execute_command",
            description=(
                "Execute a shell command on the system. Use for running CLI tools, "
                "checking system info, managing files, etc. Be careful with destructive commands."
            ),
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    description="The shell command to execute",
                    required=True,
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Timeout in seconds (default: 30)",
                    required=False,
                ),
            ],
            handler=execute_command,
            dangerous=True,  # Can execute arbitrary commands
        ),
        Tool(
            name="open_application",
            description="Open an application by name on macOS.",
            parameters=[
                ToolParameter(
                    name="app_name",
                    type="string",
                    description="Name of the application (e.g., 'Safari', 'Terminal', 'Spotify')",
                    required=True,
                ),
            ],
            handler=open_application,
        ),
        Tool(
            name="control_volume",
            description="Control system volume on macOS.",
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="Action to perform",
                    required=True,
                    enum=["get", "set", "mute", "unmute"],
                ),
                ToolParameter(
                    name="level",
                    type="integer",
                    description="Volume level 0-100 (required for 'set' action)",
                    required=False,
                ),
            ],
            handler=control_volume,
        ),
    ]
