import os
from voicekit.config import VoiceConfig, STTConfig, TTSConfig, VADConfig, LLMConfig


def config_from_env() -> VoiceConfig:
    return VoiceConfig(
        project=os.environ.get("VOICEKIT_PROJECT", "voicekit"),
        stt=STTConfig(
            model=os.environ.get("VOICEKIT_STT_MODEL", "whisper"),
            variant=os.environ.get("VOICEKIT_STT_VARIANT", "small"),
        ),
        tts=TTSConfig(
            model=os.environ.get("VOICEKIT_TTS_MODEL", "chatterbox-turbo"),
            voice=os.environ.get("VOICEKIT_TTS_VOICE", "default"),
        ),
        vad=VADConfig(
            enabled=os.environ.get("VOICEKIT_VAD_ENABLED", "true").lower() == "true",
            sensitivity=float(os.environ.get("VOICEKIT_VAD_SENSITIVITY", "0.5")),
        ),
        llm=LLMConfig(
            provider=os.environ.get("VOICEKIT_LLM_PROVIDER", "anthropic"),
            model=os.environ.get("VOICEKIT_LLM_MODEL", "claude-haiku-4-5"),
            api_key=os.environ.get("VOICEKIT_LLM_API_KEY", ""),
        ),
        system_prompt=os.environ.get(
            "VOICEKIT_SYSTEM_PROMPT",
            "You are a helpful voice assistant. "
            "Keep responses concise and natural. "
            "Never use markdown — you are speaking out loud.",
        ),
    )


def configure_llm_key(config: VoiceConfig) -> None:
    key = config.llm.api_key
    if not key:
        return
    if config.llm.provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = key
    elif config.llm.provider == "openai":
        os.environ["OPENAI_API_KEY"] = key