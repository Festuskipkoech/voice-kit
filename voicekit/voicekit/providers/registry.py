"""
Provider registry.

Maps model name strings from config to provider classes.
Adding a new model: write one class, add one line here.
Nothing else changes.
"""

from voicekit.providers.base import STTProvider, TTSProvider, LLMProvider


def get_stt(config) -> STTProvider:
    model = config.stt.model

    if model == "whisper":
        from voicekit.providers.stt.whisper import WhisperSTT
        return WhisperSTT(variant=config.stt.variant)

    else:
        raise ValueError(
            f"Unknown STT model: '{model}'. "
            f"Available: whisper, simulated"
        )


def get_tts(config) -> TTSProvider:
    model = config.tts.model

    if model == "chatterbox-turbo":
        from voicekit.providers.tts.chatterbox import ChatterboxTTS
        return ChatterboxTTS(voice=config.tts.voice)

    elif model == "kokoro":
        from voicekit.providers.tts.kokoro import KokoroTTS
        return KokoroTTS(voice=config.tts.voice)

    else:
        raise ValueError(
            f"Unknown TTS model: '{model}'. "
            f"Available: chatterbox-turbo, kokoro, simulated"
        )


def get_llm(config) -> LLMProvider:
    provider = config.llm.provider

    if provider == "anthropic":
        from voicekit.providers.llm.claude import ClaudeProvider
        return ClaudeProvider(model=config.llm.model)

    # elif provider == "openai":
    #     from voicekit.providers.llm.openai_provider import OpenAIProvider
    #     return OpenAIProvider(model=config.llm.model)
    
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Available: anthropic, openai, simulated"
        )