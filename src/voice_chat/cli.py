"""CLI interface for Voice Chat."""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from voice_chat.config import get_settings

app = typer.Typer(
    name="voice-chat",
    help="Voice-activated AI assistant with extensible tool system",
    no_args_is_help=True,
)
console = Console()


@app.command()
def chat(
    speak: bool = typer.Option(False, "--speak", "-s", help="Enable voice output (TTS)"),
    listen: bool = typer.Option(False, "--listen", "-l", help="Enable voice input (STT)"),
) -> None:
    """Start interactive chat session."""
    settings = get_settings()

    if not settings.gemini_api_key:
        console.print("[red]Error:[/red] GEMINI_API_KEY not set in environment")
        raise typer.Exit(1)

    console.print(
        Panel(
            "[bold blue]Voice Chat[/bold blue]\n"
            f"Model: {settings.gemini_model}\n"
            f"Language: {settings.language_mode.value}\n"
            f"Voice output: {'enabled' if speak else 'disabled'}\n"
            f"Voice input: {'enabled' if listen else 'disabled'}",
            title="Session Info",
        )
    )

    asyncio.run(_chat_loop(speak_enabled=speak, listen=listen))


async def _chat_loop(speak_enabled: bool = False, listen: bool = False) -> None:
    """Main chat loop."""
    from voice_chat.agent.loop import AgentLoop

    agent = AgentLoop()

    # Initialize TTS and audio output if speaking is enabled
    tts = None
    audio_output = None
    if speak_enabled:
        try:
            from voice_chat.audio import AudioOutput, TextToSpeech

            tts = TextToSpeech()
            audio_output = AudioOutput()
            console.print("[dim]TTS initialized[/dim]")
        except Exception as e:
            console.print(f"[yellow]TTS initialization failed: {e}[/yellow]")
            speak_enabled = False

    console.print("\n[dim]Type 'exit' or 'quit' to end the session[/dim]\n")

    while True:
        try:
            # Get user input
            if listen:
                # TODO: Implement voice input in Phase 3
                console.print("[yellow]Voice input not yet implemented. Using text input.[/yellow]")

            user_input = console.input("[bold green]You:[/bold green] ")

            if user_input.lower() in ("exit", "quit"):
                console.print("[dim]Goodbye![/dim]")
                break

            if not user_input.strip():
                continue

            # Process through agent
            console.print("[bold blue]Assistant:[/bold blue] ", end="")
            response = await agent.run(user_input)
            console.print(response)

            # Speak response if enabled
            if speak_enabled and tts and audio_output:
                try:
                    console.print("[dim]Speaking...[/dim]", end="")
                    audio_data = await tts.synthesize(response)
                    await audio_output.play_mp3(audio_data)
                    console.print("\r" + " " * 20 + "\r", end="")  # Clear "Speaking..."
                except Exception as e:
                    console.print(f"\n[yellow]TTS error: {e}[/yellow]")

        except KeyboardInterrupt:
            if audio_output:
                audio_output.stop()
            console.print("\n[dim]Interrupted. Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


@app.command()
def run(
    message: str = typer.Argument(..., help="Message to process"),
    speak: bool = typer.Option(False, "--speak", "-s", help="Speak the response"),
) -> None:
    """Run a single command and exit."""
    settings = get_settings()

    if not settings.gemini_api_key:
        console.print("[red]Error:[/red] GEMINI_API_KEY not set in environment")
        raise typer.Exit(1)

    async def _run() -> None:
        from voice_chat.agent.loop import AgentLoop

        agent = AgentLoop()
        response = await agent.run(message)
        console.print(response)

        if speak:
            try:
                from voice_chat.audio import AudioOutput, TextToSpeech

                tts = TextToSpeech()
                audio_output = AudioOutput()
                audio_data = await tts.synthesize(response)
                await audio_output.play_mp3(audio_data)
            except Exception as e:
                console.print(f"[yellow]TTS error: {e}[/yellow]")

    asyncio.run(_run())


@app.command()
def voice(
    wake_word: str = typer.Option("jarvis", "--wake-word", "-w", help="Wake word to listen for"),
    no_wake_word: bool = typer.Option(False, "--no-wake-word", help="Disable wake word, always listen"),
) -> None:
    """Start voice-activated assistant with wake word detection."""
    settings = get_settings()

    if not settings.gemini_api_key:
        console.print("[red]Error:[/red] GEMINI_API_KEY not set in environment")
        raise typer.Exit(1)

    console.print(
        Panel(
            "[bold blue]Voice Assistant[/bold blue]\n"
            f"Model: {settings.gemini_model}\n"
            f"Language: {settings.language_mode.value}\n"
            f"Wake word: {wake_word if not no_wake_word else 'disabled'}",
            title="Voice Mode",
        )
    )

    if not no_wake_word:
        console.print(f"\n[dim]Say '{wake_word}' to activate, Ctrl+C to exit[/dim]\n")
    else:
        console.print("\n[dim]Listening continuously, Ctrl+C to exit[/dim]\n")

    asyncio.run(_voice_loop(wake_word=wake_word, enable_wake_word=not no_wake_word))


async def _voice_loop(wake_word: str = "jarvis", enable_wake_word: bool = True) -> None:
    """Voice-activated assistant loop."""
    from voice_chat.assistant import AssistantConfig, AssistantState, VoiceAssistant

    def on_state_change(state: AssistantState) -> None:
        state_icons = {
            AssistantState.IDLE: "[dim]Idle[/dim]",
            AssistantState.LISTENING: "[green]Listening...[/green]",
            AssistantState.PROCESSING: "[yellow]Thinking...[/yellow]",
            AssistantState.SPEAKING: "[blue]Speaking...[/blue]",
            AssistantState.INTERRUPTED: "[red]Interrupted[/red]",
        }
        console.print(f"\r{state_icons.get(state, state.name)}", end="")

    def on_transcription(text: str) -> None:
        console.print(f"\n[bold green]You:[/bold green] {text}")

    def on_response(text: str) -> None:
        console.print(f"[bold blue]Assistant:[/bold blue] {text}")

    def on_error(error: Exception) -> None:
        console.print(f"\n[red]Error:[/red] {error}")

    config = AssistantConfig(
        wake_words=[wake_word],
        enable_wake_word=enable_wake_word,
        enable_vad=True,
        auto_listen_after_response=True,
    )

    assistant = VoiceAssistant(
        config=config,
        on_state_change=on_state_change,
        on_transcription=on_transcription,
        on_response=on_response,
        on_error=on_error,
    )

    try:
        await assistant.start()
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")
    finally:
        await assistant.stop()


@app.command()
def version() -> None:
    """Show version information."""
    from voice_chat import __version__

    console.print(f"voice-chat version {__version__}")


@app.command()
def test_wake_word(
    wake_word: str = typer.Option("jarvis", "--wake-word", "-w", help="Wake word to test"),
) -> None:
    """Test wake word detection in isolation."""
    import time

    import numpy as np

    from voice_chat.audio.input import AudioInput
    from voice_chat.audio.wake_word import StreamingWakeWordDetector, WakeWordDetection
    from voice_chat.config import get_settings

    settings = get_settings()

    console.print(f"[bold]Testing wake word detection[/bold]")
    console.print(f"Wake word: {wake_word}")
    console.print(f"Sample rate: {settings.sample_rate} Hz")

    detections = []

    def on_wake_word(detection: WakeWordDetection) -> None:
        detections.append(detection)
        console.print(f"\n[bold green]DETECTED: {detection.keyword}![/bold green]")

    try:
        detector = StreamingWakeWordDetector(
            keywords=[wake_word],
            on_wake_word=on_wake_word,
        )
        console.print(f"Porcupine frame length: {detector.frame_length}")
        console.print(f"Porcupine sample rate: {detector.sample_rate}")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Make sure PICOVOICE_ACCESS_KEY is set in .env")
        raise typer.Exit(1)

    audio_input = AudioInput(
        sample_rate=settings.sample_rate,
        chunk_size=detector.frame_length,  # Match Porcupine's frame length
    )

    console.print(f"\n[dim]Say '{wake_word}' - Ctrl+C to stop[/dim]")
    console.print("[dim]Audio levels will show as dots[/dim]\n")

    detector.start()
    audio_input.start()

    chunks_processed = 0
    try:
        while True:
            audio = audio_input.read(timeout=0.1)
            if audio is not None:
                chunks_processed += 1
                # Show audio level indicator
                level = np.abs(audio).mean()
                if level > 0.01:
                    console.print(".", end="", style="green")
                elif level > 0.001:
                    console.print(".", end="", style="yellow")

                detection = detector.process_chunk(audio)
                if detection:
                    console.print(f"\n[bold green]Wake word detected![/bold green]")

    except KeyboardInterrupt:
        console.print(f"\n\nProcessed {chunks_processed} audio chunks")
        console.print(f"Detections: {len(detections)}")
    finally:
        audio_input.stop()
        detector.cleanup()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Voice Chat - A voice-activated AI assistant."""
    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")


if __name__ == "__main__":
    app()
