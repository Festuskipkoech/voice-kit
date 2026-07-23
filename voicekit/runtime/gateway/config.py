"""
Gateway configuration.

Reads all runtime settings from environment variables.
API keys are routed to the correct provider env vars by configure_llm_key().

System prompt is defined here in code and never read from the environment.
It contains structured markers (<|BREAK|>) and precise formatting rules that
must not be accidentally overridden by an env var — a misconfigured prompt
breaks the entire streaming architecture.
"""
import os

from voicekit.config import VoiceConfig, STTConfig, TTSConfig, VADConfig, LLMConfig

SYSTEM_PROMPT = (
    "You are a voice assistant. Everything you say is immediately converted "
    "to speech and played as audio. The user cannot see text.\n\n"
    "MARKER RULE — THIS IS CRITICAL:\n"
    "After every sentence or natural speech pause, you MUST insert the exact "
    "marker <|BREAK|> with no spaces around it.\n"
    "Example: \"Sure,<|BREAK|> I am doing well.<|BREAK|> How can I help you?<|BREAK|>\"\n"
    "Every response MUST contain at least one <|BREAK|> marker.\n"
    "Missing markers cause audio delay — treat this as a hard requirement.\n\n"
    "CONTENT RULES:\n"
    "Plain spoken English only. No emojis, no asterisks, no markdown, "
    "no bullet points, no lists, no headers. Maximum 2 to 3 sentences per "
    "response. Never repeat yourself. Never use filler phrases."
)

def config_from_env() -> VoiceConfig:
    return VoiceConfig(
        project=os.environ.get("VOICEKIT_PROJECT"),
        stt=STTConfig(
            model=os.environ.get("VOICEKIT_STT_MODEL"),
            variant=os.environ.get("VOICEKIT_STT_VARIANT"),
        ),
        tts=TTSConfig(
            model=os.environ.get("VOICEKIT_TTS_MODEL"),
            voice=os.environ.get("VOICEKIT_TTS_VOICE"),
        ),
        vad=VADConfig(
            enabled=os.environ.get("VOICEKIT_VAD_ENABLED", "true").lower() == "true",
            sensitivity=float(os.environ.get("VOICEKIT_VAD_SENSITIVITY")),
        ),
        llm=LLMConfig(
            provider=os.environ.get("VOICEKIT_LLM_PROVIDER"),
            model=os.environ.get("VOICEKIT_LLM_MODEL"),
            api_key=os.environ.get("VOICEKIT_LLM_API_KEY"),
        ),
        system_prompt=SYSTEM_PROMPT,
    )

def configure_llm_key(config: VoiceConfig) -> None:
    key = config.llm.api_key
    if not key:
        return
    if config.llm.provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = key
    elif config.llm.provider == "openai":
        os.environ["OPENAI_API_KEY"] = key