import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

@dataclass
class STTConfig:
    model: str
    variant: str

@dataclass
class TTSConfig:
    model: str
    voice: str

@dataclass
class VADConfig:
    enabled: bool
    sensitivity: float

@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str

@dataclass
class VoiceConfig:
    project: str
    stt: STTConfig
    tts: TTSConfig
    vad: VADConfig
    llm: LLMConfig
    system_prompt: str

def _resolve_env(value: str) -> str:
    """
    Replace ${VAR_NAME} patterns with environment variable values.
    Raises clearly if a referenced variable is not set.
    """
    pattern = re.compile(r"\$\{([^}]+)\}")

    def replacer(match):
        var_name = match.group(1)
        val = os.environ.get(var_name)
        if val is None:
            raise EnvironmentError(
                f"Environment variable '{var_name}' is referenced in "
                f"voice.config.yaml but is not set."
            )
        return val

    return pattern.sub(replacer, value)

def load_config(path: Path = None) -> VoiceConfig:
    """
    Load and validate voice.config.yaml from the given path,
    or from the current working directory if no path given.
    """
    if path is None:
        path = Path.cwd() / "voice.config.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"voice.config.yaml not found at {path}.\n"
            "Are you inside a voicekit project directory?\n"
            "Run: voicekit init <project-name>"
        )

    with open(path) as f:
        raw = yaml.safe_load(f)

    _validate_required_keys(raw, path)

    llm_raw = raw.get("llm", {})
    api_key_raw = llm_raw.get("api_key", "")
    api_key = _resolve_env(str(api_key_raw)) if api_key_raw else ""

    vad_raw = raw.get("vad", {})
    sensitivity = float(vad_raw.get("sensitivity", 0.5))
    if not 0.0 <= sensitivity <= 1.0:
        raise ValueError(
            f"vad.sensitivity must be between 0.0 and 1.0, got {sensitivity}"
        )

    return VoiceConfig(
        project=raw["project"],
        stt=STTConfig(
            model=raw["stt"]["model"],
            variant=raw["stt"].get("variant", "small"),
        ),
        tts=TTSConfig(
            model=raw["tts"]["model"],
            voice=raw["tts"].get("voice", "default"),
        ),
        vad=VADConfig(
            enabled=bool(vad_raw.get("enabled", True)),
            sensitivity=sensitivity,
        ),
        llm=LLMConfig(
            provider=llm_raw.get("provider", "simulated"),
            model=llm_raw.get("model", "simulated"),
            api_key=api_key,
        ),
        system_prompt=raw.get(
            "system_prompt",
            "You are a helpful voice assistant. Keep responses concise and natural."
        ),
    )

def _validate_required_keys(raw: dict, path: Path) -> None:
    required = ["project", "stt", "tts", "llm"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(
            f"voice.config.yaml at {path} is missing required keys: {missing}"
        )

    stt_required = ["model"]
    missing_stt = [k for k in stt_required if k not in raw.get("stt", {})]
    if missing_stt:
        raise ValueError(
            f"voice.config.yaml stt section is missing: {missing_stt}"
        )

    tts_required = ["model"]
    missing_tts = [k for k in tts_required if k not in raw.get("tts", {})]
    if missing_tts:
        raise ValueError(
            f"voice.config.yaml tts section is missing: {missing_tts}"
        )