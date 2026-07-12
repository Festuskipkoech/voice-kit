import os
import subprocess
import shutil
from pathlib import Path
 
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
 
from voicekit.config import load_config
from voicekit.providers.registry import STT_REGISTRY, TTS_REGISTRY, LLM_REGISTRY

app = typer.Typer(
    name="voicekit",
    help="Self-hosted voice agent infrastructure.",
    add_completion=False,
)
 
console = Console()
 
RUNTIME_DIR = Path(__file__).parent.parent / "runtime"
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

@app.command()
def init(
    name: str = typer.Argument(..., help="Name of your voice agent project"),
):

    """
    Scaffold a new voice agent project.
 
    Creates a project directory with a voice.config.yaml
    you edit to configure your models and system prompt.
    """
    project_path = Path.cwd() / name

    if project_path.exists():
        console.print(f"[red]Error:[/red] Directory '{name}' already exists.")
        raise typer.Exit(code=1)

    project_path.mkdir()

    template_config = TEMPLATES_DIR / "voice.config.yaml"
    dest_config = project_path / "voice.config.yaml"

    if template_config.exists():
        shutil.copy(template_config , dest_config)

    else:
        dest_config.write_text(_default_config(name))
        (project_path / "prompt.txt").write_text(
        "You are a helpful voice assistant. "
        "Keep responses concise and natural. "
        "Never use markdown — you are speaking out loud.\n"
    )
 
    console.print(Panel(
        f"[green]Created project:[/green] [bold]{name}[/bold]\n\n"
        f"Next steps:\n"
        f"  [dim]cd {name}[/dim]\n"
        f"  [dim]voicekit setup[/dim]\n"
        f"  [dim]voicekit dev[/dim]",
        title="voicekit init",
        border_style="green",
    ))

@app.command()
def setup():
    """
    Pull Docker images and validate configuration.
 
    Run this once after init, or after changing voice.config.yaml
    to a model that has not been pulled yet.
    """
    try:
        config = load_config()
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"[red]Config error:[/red] {e}")
        raise typer.Exit(code=1)
    
    console.print(f"[blue]Setting up[/blue] [bold]{config.project}[/bold]")
    console.print(f"  STT: [cyan]{config.stt.model}[/cyan] ({config.stt.variant})")
    console.print(f"  TTS: [cyan]{config.tts.model}[/cyan]")
    console.print(f"  LLM: [cyan]{config.llm.provider}[/cyan] / {config.llm.model}")

    compose_file = RUNTIME_DIR / "docker-compose.yml"
    if not compose_file.exists():
        console.print(f"[red]Error:[/red] runtime/docker-compose.yml not found at {compose_file}")
        raise typer.Exit(code=1)
    
    console.print("\n[blue]Pulling Docker images...[/blue]")
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "pull"],
        capture_output=False,
    )
    if result.returncode != 0:
        console.print("[red]Docker pull failed.[/red] Is Docker running?")
        raise typer.Exit(code=1)
    
    console.print("\n[green]Setup complete.[/green] Run [bold]voicekit dev[/bold] to start.")

@app.command()
def dev():
    """
    Start the full voice stack locally for development.
 
    Runs all services — STT, TTS, gateway — with logs
    streaming to your terminal. Stop with Ctrl+C.
    """

    try:
        config = load_config()
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    compose_file = RUNTIME_DIR / "docker-compose.yml"

    console.print(Panel(
        f"[green]Starting[/green] [bold]{config.project}[/bold]\n\n"
        f"  STT [cyan]http://localhost:8001[/cyan]\n"
        f"  TTS  [cyan]http://localhost:8002[/cyan]\n"
        f"  Gate [cyan]http://localhost:8000[/cyan]\n\n"
        f"[dim]Press Ctrl+C to stop[/dim]",
        title="voicekit dev",
        border_style="blue",
    ))

    env = _build_env(config)

    try: 
        subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "up", "--build"],
            env=env,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
        subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "down"],
            env=env,
        )
        console.print("[green]Stopped.[/green]")

@app.command()
def stop():
    """
    Stop all running voice stack services.
    """
    compose_file =RUNTIME_DIR / "docker-compose.yml"

    console.print("[yellow]Stopping voicekit services...[/yellow]")
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "down"],
    )
    console.print("[green]All services stopped.[/green]")

@app.command()
def status():
    """
    Show the status of all running services.
    """
    compose_file = RUNTIME_DIR / "docker-compose.yml"
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "ps"]
    )

@app.command()
def models():
    """
    List all available STT, TTS, and LLM models.
    """
    table = Table(title="Available Models", border_style="blue")
    table.add_column("Type", style="dim")
    table.add_column("Model ID", style="cyan")
    table.aadd_column("Status")

    for name in STT_REGISTRY:
        table.add_row("STT", name, "[green]available[/green]")    
    
    for name in TTS_REGISTRY:
        table.add_row("TTS", name, "[green]available[/green]")
 
    for name in LLM_REGISTRY:
        table.add_row("LLM", name, "[green]available[/green]")
    console.print(table)
    console.print(
        "\n[dim]Set models in [bold]voice.config.yaml[/bold] in your project directory.[/dim]"
    )

def _build_env(config) -> dict:
    env = os.environ.copy()
    env["VOICEKIT_STT_MODEL"] = config.stt.model
    env["VOICEKIT_STT_VARIANT"] = config.stt.variant
    env["VOICEKIT_TTS_MODEL"] = config.tts.model
    env["VOICEKIT_TTS_VOICE"] = config.tts.voice
    env["VOICEKIT_LLM_PROVIDER"] = config.llm.provider
    env["VOICEKIT_LLM_MODEL"] = config.llm.model
    if config.llm.api_key:
        env["VOICEKIT_LLM_API_KEY"] = config.llm.api_key
    return env 
 
def _default_config(name: str) -> str:
    return f"""project: {name}
 
stt:
  model: simulated
  variant: small
 
tts:
  model: simulated
  voice: default
 
vad:
  enabled: true
  sensitivity: 0.5
 
llm:
  provider: simulated
  model: simulated
  api_key: ""
 
system_prompt: >
  You are a helpful voice assistant.
  Keep responses concise and natural.
  Never use markdown — you are speaking out loud.
"""
 
 

    