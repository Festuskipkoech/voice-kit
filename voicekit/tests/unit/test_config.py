import pytest
from pathlib import Path
from voicekit.config import load_config

def write_config(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "voice.config.yaml"
    p.write_text(content)
    return p

def test_valid_config_loads(tmp_path):
    p = write_config(tmp_path, """
project: test-agent
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
""")
    config = load_config(p)
    assert config.project == "test-agent"
    assert config.stt.model == "simulated"
    assert config.stt.variant == "small"
    assert config.tts.model == "simulated"
    assert config.vad.enabled is True
    assert config.vad.sensitivity == 0.5
    assert config.llm.provider == "simulated"

def test_missing_config_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_config(tmp_path / "voice.config.yaml")
    assert "voice.config.yaml not found" in str(exc.value)
    assert "voicekit init" in str(exc.value)

def test_missing_required_key_raises_error(tmp_path):
    p = write_config(tmp_path, """
stt:
  model: simulated
  variant: small
tts:
  model: simulated
llm:
  provider: simulated
  model: simulated
""")
    with pytest.raises(ValueError) as exc:
        load_config(p)
    assert "project" in str(exc.value)

def test_invalid_vad_sensitivity_raises_error(tmp_path):
    p = write_config(tmp_path, """
project: test
stt:
  model: simulated
  variant: small
tts:
  model: simulated
  voice: default
vad:
  enabled: true
  sensitivity: 1.5
llm:
  provider: simulated
  model: simulated
  api_key: ""
""")
    with pytest.raises(ValueError) as exc:
        load_config(p)
    assert "sensitivity" in str(exc.value)

def test_env_variable_in_api_key_resolved(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-test-12345")
    p = write_config(tmp_path, """
project: test
stt:
  model: simulated
  variant: small
tts:
  model: simulated
  voice: default
llm:
  provider: simulated
  model: simulated
  api_key: ${TEST_API_KEY}
""")
    config = load_config(p)
    assert config.llm.api_key == "sk-test-12345"

def test_missing_env_variable_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    p = write_config(tmp_path, """
project: test
stt:
  model: simulated
  variant: small
tts:
  model: simulated
  voice: default
llm:
  provider: simulated
  model: simulated
  api_key: ${MISSING_KEY}
""")
    with pytest.raises(EnvironmentError) as exc:
        load_config(p)
    assert "MISSING_KEY" in str(exc.value)

def test_default_system_prompt_applied_when_not_set(tmp_path):
    p = write_config(tmp_path, """
project: test
stt:
  model: simulated
  variant: small
tts:
  model: simulated
  voice: default
llm:
  provider: simulated
  model: simulated
  api_key: ""
""")
    config = load_config(p)
    assert len(config.system_prompt) > 0