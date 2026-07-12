from voicekit.providers.base import STTProvider, TTSProvider, LLMProvider
from voicekit.providers.stt.simulated import SimulatedSTT
from voicekit.providers.tts.simulated import SimulatedTTS
from voicekit.providers.llm.simulated import SimulatedLLM

# adding a new model means two things only:
# 1. write a class in providers/stt/, providers/tts/, or providers/llm/
# 2. add it to the relevant dict below
# nothing else in the codebase changes

STT_REGISTRY: dict[str, type[STTProvider]] = {
    "simulated": SimulatedSTT,
    # "whisper": WhisperSTT,      <- uncomment when real model added
    # "parakeet": ParakeetSTT,
}

TTS_REGISTRY: dict[str, type[TTSProvider]] = {
    "simulated": SimulatedTTS,
    # "chatterbox-turbo": ChatterboxTTS,
    # "kokoro": KokoroTTS,
}
 
LLM_REGISTRY: dict[str, type[LLMProvider]] = {
    "simulated": SimulatedLLM,
    # "anthropic": ClaudeProvider,
    # "openai": OpenAIProvider,
}
 
def get_stt(config) -> STTProvider:
    cls= STT_REGISTRY.get(config.stt.model)
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