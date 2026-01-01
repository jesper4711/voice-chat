"""Spotify control tools (macOS only via AppleScript)."""

import subprocess
import sys

from voice_chat.tools.registry import Tool, ToolParameter


async def spotify_control(action: str) -> str:
    """Control Spotify playback.

    Args:
        action: One of 'play', 'pause', 'toggle', 'next', 'previous'.

    Returns:
        Result message.
    """
    if sys.platform != "darwin":
        return "Error: Spotify control is only supported on macOS"

    commands = {
        "play": 'tell application "Spotify" to play',
        "pause": 'tell application "Spotify" to pause',
        "toggle": 'tell application "Spotify" to playpause',
        "next": 'tell application "Spotify" to next track',
        "previous": 'tell application "Spotify" to previous track',
    }

    if action not in commands:
        return f"Error: Unknown action '{action}'. Use: play, pause, toggle, next, previous"

    try:
        subprocess.run(
            ["osascript", "-e", commands[action]],
            capture_output=True,
            timeout=5,
        )
        return f"Spotify: {action}"
    except Exception as e:
        return f"Error controlling Spotify: {e}"


async def spotify_now_playing() -> str:
    """Get currently playing track info."""
    if sys.platform != "darwin":
        return "Error: Spotify control is only supported on macOS"

    script = '''
    tell application "Spotify"
        if player state is playing then
            set trackName to name of current track
            set artistName to artist of current track
            set albumName to album of current track
            return trackName & " by " & artistName & " (" & albumName & ")"
        else
            return "Nothing is currently playing"
        end if
    end tell
    '''

    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return f"Now playing: {result.stdout.strip()}"
    except Exception as e:
        return f"Error getting track info: {e}"


async def spotify_search_and_play(query: str, type: str = "track") -> str:
    """Search and play on Spotify.

    Args:
        query: Search query (song name, artist, playlist).
        type: Type of search - 'track', 'album', 'playlist', 'artist'.

    Returns:
        Result message.
    """
    if sys.platform != "darwin":
        return "Error: Spotify control is only supported on macOS"

    # Spotify URI search format
    search_uri = f"spotify:search:{query}"

    # For direct playback, we use the search URI
    script = f'''
    tell application "Spotify"
        activate
        play track "{search_uri}"
    end tell
    '''

    try:
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=5,
        )
        return f"Searching and playing: {query}"
    except Exception as e:
        return f"Error: {e}"


async def spotify_set_volume(level: int) -> str:
    """Set Spotify volume.

    Args:
        level: Volume level 0-100.

    Returns:
        Result message.
    """
    if sys.platform != "darwin":
        return "Error: Spotify control is only supported on macOS"

    level = max(0, min(100, level))
    script = f'tell application "Spotify" to set sound volume to {level}'

    try:
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=5,
        )
        return f"Spotify volume set to {level}%"
    except Exception as e:
        return f"Error setting volume: {e}"


def get_spotify_tools() -> list[Tool]:
    """Get all Spotify tools."""
    return [
        Tool(
            name="spotify_control",
            description="Control Spotify playback: play, pause, toggle (play/pause), next track, or previous track.",
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="Playback action",
                    required=True,
                    enum=["play", "pause", "toggle", "next", "previous"],
                ),
            ],
            handler=spotify_control,
        ),
        Tool(
            name="spotify_now_playing",
            description="Get information about the currently playing track on Spotify.",
            parameters=[],
            handler=spotify_now_playing,
        ),
        Tool(
            name="spotify_search_and_play",
            description="Search for and play music on Spotify. Can search for songs, artists, albums, or playlists.",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query (e.g., 'Bohemian Rhapsody', 'Taylor Swift', 'chill playlist')",
                    required=True,
                ),
            ],
            handler=spotify_search_and_play,
        ),
        Tool(
            name="spotify_set_volume",
            description="Set Spotify's volume level (0-100). This is separate from system volume.",
            parameters=[
                ToolParameter(
                    name="level",
                    type="integer",
                    description="Volume level from 0 to 100",
                    required=True,
                ),
            ],
            handler=spotify_set_volume,
        ),
    ]
