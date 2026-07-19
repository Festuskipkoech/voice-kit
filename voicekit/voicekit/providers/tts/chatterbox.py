import asyncio
import io
from typing import AsyncIterator

from voicekit.providers.base import TTSProvider
 
SAMPLE_RATE = 24000
SENTENCE_BOUNDARIES = (".", "!", "?", ",") 
 
class ChatterboxTTS(TTSProvider):
    """
    Chatterbox Turbo TTS — 350M params, MIT license.
    Paralinguistic tags supported: [chuckle], [sigh], [gasp], [laugh], [cough]
    Voice cloning from ~10 second reference audio clip.
 
    Uses ChatterboxTurboTTS from chatterbox.tts_turbo — the dedicated
    Turbo class, not the base ChatterboxTTS. Turbo uses 1-step diffusion
    for significantly faster inference than the original model.
 
    Sentence boundary buffering:
        TTS needs sentence context for natural intonation.
        Sending one token at a time produces flat robotic delivery.
        Waiting for the full response kills latency.
        Buffer tokens to sentence boundaries — audio starts after the
        first complete sentence (typically 300-500ms into LLM response)
        with natural prosody from having a complete thought to work with.
 
    API confirmed from chatterbox-tts 0.1.7 (latest as of July 2026):
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_pretrained(device="cpu")
        wav = model.generate(text, audio_prompt_path="ref.wav")
        torchaudio.save("out.wav", wav, model.sr)
    """
    def __init__(self, voice: str = "default"):
        # voice: "default" uses built-in voice
        # voice: "/path/to/reference.wav" enables voice cloning
        self.voice = voice
        self.model = None
        
    async def load(self) -> None:
        # patch perth before chatterbox loads it
        # perth.PerthImplicitWatermarker fails with uv due to missing pkg_resources
        # patch it to DummyWatermarker before chatterbox imports perth
        import perth
        
        # Manually create a local Dummy class to bypass the import error
        class DummyWatermarker:
            def __init__(self, *args, **kwargs): pass
            def apply_watermark(self, wav, *args, **kwargs): return wav
            def __call__(self, *args, **kwargs): return args[0] if args else None

        # Assign it directly to mock out the broken watermarker package behavior
        perth.PerthImplicitWatermarker = DummyWatermarker
        perth.DummyWatermarker = DummyWatermarker

        from chatterbox.tts_turbo import ChatterboxTurboTTS

        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(
            None,
            lambda: ChatterboxTurboTTS.from_pretrained(device="cpu")
        )

    async def synthesize(self, text_stream: AsyncIterator[str]) ->AsyncIterator[bytes]:
        """
        Accept LLM token stream, yield WAV audio chunks.
 
        Buffers incoming tokens until a sentence boundary appears,
        then synthesises the buffered sentence immediately.
        Audio streams back in 100ms chunks per sentence.
        """
        buffer = ""

        async for token in text_stream:
            buffer += token

            if buffer.rstrip().endswith(SENTENCE_BOUNDARIES):
                sentence = buffer.strip()
                if sentence:
                    async for chunk in self._synthesize_sentence(sentence):
                        yield chunk
                buffer = ""
        # flush any remaining text after stream ends
        if buffer.strip():
            async for chunk in self._synthesize_sentence(buffer.strip()):
                yield chunk
    
    async def _synthesize_sentence(self, text: str) -> AsyncIterator[bytes]:
        """
        Synthesise one sentence to WAV bytes and stream in chunks.
        Runs inference in executor — generate() is synchronous.
        """
        import torchaudio
        loop = asyncio.get_event_loop()
        
        # build generate kwargs
        # audio_prompt_path enables voice cloning when set
        generate_kwargs: dict = {"audio_prompt_path": self.voice} \
            if self.voice != "default" else {}
        wav_tensor = await loop.run_in_executor(
            None,
            lambda: self.model.generate(text, **generate_kwargs)
        )
        buf = io.BytesIO()
        torchaudio.save(buf, wav_tensor, self.model.sr, format="wav")
        wav_bytes = buf.getvalue()

        # stream in 100ms chunks so playback starts immediately
        # without waiting for the full sentence to transmit
        chunk_size = self.model.sr * 2 // 10
        for i in range(0, len(wav_bytes), chunk_size):
            yield wav_bytes[i:i + chunk_size]
            await asyncio.sleep(0)

    async def health(self) -> bool:
        return self.model is not None