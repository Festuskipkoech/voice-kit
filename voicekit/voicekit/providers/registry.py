from voicekit.providers.base import STTProvider, TTSProvider, LLMProvider
from voicekit.providers.stt.whisper import WhisperSTT
from voicekit.providers.tts.chatterbox import ChatterboxTTS
from voicekit.providers.llm.claude import ClaudeProvider
 
# adding a new model means two things only:
# 1. write a class in providers/stt/, providers/tts/, or providers/llm/
#    implementing the relevant abstract base class from providers/base.py
# 2. add one entry to the relevant registry dict below
# nothing else in the codebase changes
 
STT_REGISTRY: dict[str, type[STTProvider]] = {
    "whisper": WhisperSTT,
    # "parakeet": ParakeetSTT,    <- add in future
}
 
TTS_REGISTRY: dict[str, type[TTSProvider]] = {
    "chatterbox-turbo": ChatterboxTTS,
    # "kokoro": KokoroTTS,        <- Phase 2
}
 
LLM_REGISTRY: dict[str, type[LLMProvider]] = {
    "anthropic": ClaudeProvider,
    # "openai": OpenAIProvider,   <- Phase 2
}

def get_stt(config) -> STTProvider:
    cls = STT_REGISTRY.get(config.stt.model)
    if not cls:
        available = list(STT_REGISTRY.keys())
        raise ValueError(
            f"Unknown STT model '{config.stt.model}'. "
            f"Available: {available}"
        )
    return cls(variant=config.stt.variant)

def get_tts(config) -> TTSProvider:
    cls = TTS_REGISTRY.get(config.tts.model)
    if not cls:
        available = list(TTS_REGISTRY.keys())
        raise ValueError(
            f"Unknown TTS model '{config.tts.model}'. "
            f"Available: {available}"
        )
    return cls(voice=config.tts.voice) 
 
def get_llm(config) -> LLMProvider:
    cls = LLM_REGISTRY.get(config.llm.provider)
    if not cls:
        available = list(LLM_REGISTRY.keys())
        raise ValueError(
            f"Unknown LLM provider '{config.llm.provider}'. "
            f"Available: {available}"
        )
    return cls(model=config.llm.model)
 