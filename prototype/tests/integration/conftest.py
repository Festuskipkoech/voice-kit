"""
Integration test fixtures.

Manages Docker service lifecycle for integration tests.
Services start once per test session and shut down after all tests complete.
Tests never call docker manually — these fixtures handle everything.

Usage:
    pytest tests/integration/ -v

Requirements:
    Docker must be running.
    Ports 8011 and 8012 must be free (test ports, separate from dev ports).
"""

import subprocess
import time
from pathlib import Path

import httpx
import pytest

COMPOSE_FILE = Path(__file__).parent.parent.parent / "runtime" / "docker-compose.test.yml"

STT_URL = "http://localhost:8011"
TTS_URL = "http://localhost:8012"

STT_WS_URL = "ws://localhost:8011"
TTS_WS_URL = "ws://localhost:8012"

def _wait_for_service(url: str, name: str, timeout_seconds: int = 60) -> None:
    """
    Poll the health endpoint until the service is ready.
    Raises if the service does not become healthy within timeout.
    """
    deadline = time.time() + timeout_seconds
    last_error = None

    while time.time() < deadline:
        try:
            response = httpx.get(f"{url}/health", timeout=2.0)
            if response.status_code == 200:
                return
        except Exception as e:
            last_error = e
        time.sleep(1.0)

    raise RuntimeError(
        f"{name} did not become healthy within {timeout_seconds}s. "
        f"Last error: {last_error}"
    )

def _compose(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE)] + args,
        capture_output=True,
        text=True,
    )

@pytest.fixture(scope="session")
def stt_service():
    """
    Start the STT service in Docker for the test session.
    Yields the base URL. Tears down after all tests complete.
    """
    _compose(["up", "-d", "--build", "stt-test"])

    try:
        _wait_for_service(STT_URL, "STT service", timeout_seconds=60)
    except RuntimeError:
        _compose(["logs", "stt-test"])
        _compose(["down"])
        raise

    yield STT_URL

    _compose(["down", "--remove-orphans"])

@pytest.fixture(scope="session")
def tts_service():
    """
    Start the TTS service in Docker for the test session.
    Yields the base URL. Tears down after all tests complete.
    """
    _compose(["up", "-d", "--build", "tts-test"])

    try:
        _wait_for_service(TTS_URL, "TTS service", timeout_seconds=60)
    except RuntimeError:
        _compose(["logs", "tts-test"])
        _compose(["down"])
        raise

    yield TTS_URL

    _compose(["down", "--remove-orphans"])

@pytest.fixture
def stt_ws_url():
    return STT_WS_URL

@pytest.fixture
def tts_ws_url():
    return TTS_WS_URL