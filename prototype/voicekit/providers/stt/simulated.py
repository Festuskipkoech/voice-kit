import asyncio
from typing import AsyncIterator
import numpy as np
 
from voicekit.providers.base import STTProvider
 
TRANSCRIPTS_BY_DURATION = [
    (1.0, "hello"),
    (3.0, "hello how are you doing today"),
    (6.0, "hello this is a longer utterance that tests multi sentence handling"),
    (float("inf"), "hello this is a very long utterance testing the pipeline under extended audio input conditions"),
]

class SimulatedSTT(STTProvider):
    """
    Simulates a real STT model without loading anything.
 
    Realistic behaviour:
    - Waits configurable latency before yielding (simulates inference)
    - Returns predictable transcripts based on audio duration
      so tests can assert on specific strings
    - Generates load time delay on startup so startup sequencing
      is tested realistically
    """

    def __init__(self, variant: str = "small", latency_ms: int = 150):
        self.variant = variant
        self.latency_ms = latency_ms
        self._loaded = False
    
    async def load(self) -> None:
        # simulate model load time — whisper small takes ~3 seconds
        await asyncio.sleep(0.5)
        self._loaded = True

    async def transcribe(
        self,
        audio_stream: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[str]:
        chunks = []
        async for chunk in  audio_stream:
            chunks.append(chunk)
        
        if not chunks:
            return
        
        # simulate inference latency
        await asyncio.sleep(self.latency_ms / 1000)

        total_samples = sum(len(c) for c in chunks)
        duration_seconds = total_samples / 1600

        transcript = TRANSCRIPTS_BY_DURATION[-1][1]
        for threshold, text in TRANSCRIPTS_BY_DURATION:
            if duration_seconds <= threshold:
                transcript = text
                break

        # yield tokens word by word to simulate streaming
        for word in transcript.split():
            yield word + " "
            await asyncio.sleep(0.02)

    async def health(self) -> bool:
        return self._loaded