from abc import ABC, abstractmethod
from typing import AsyncIterator
import numpy as np

class STTProvider(ABC):
    """
    Contract every STT model must satisfy.
    The pipeline only ever calls these methods — never the
    concrete implementation directly.
    """

    @abstractmethod
    async def load(self) -> None:
        """
        Load model into memory.
        Called once at service startup, not per request.
        Heavy work goes here so requests stay fast.
        """
    
    @abstractmethod
    async def transcribe(
        self,
        audio_stream: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[str]:
        """
        Accept a stream of audio chunks (float32, 16kHz mono).
        Yield transcript tokens as they are recognised.
        Must not wait for the full audio before yielding.
        """

    @abstractmethod
    async def health(self) -> bool:
        """
        Return True if the model is loaded and ready to serve.
        Called by the /health endpoint every 10 seconds.
        """

class TTSProvider(ABC):
    """
    Contract every TTS model must satisfy.
    """
    @abstractmethod
    async def load(self) -> None:
        ...

    @abstractmethod
    async def synthesize(
        self,
        tex_stream: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """
        Accept a stream of text tokens from the LLM.
        Yield WAV audio chunks as they are generated.
        First audio chunk must arrive before text stream ends —
        that is what makes the pipeline feel real-time.
        """
    
    @abstractmethod
    async def health(self) -> bool:
        ...

class LLMProvider(ABC):
    """
    Contract every LLM must satisfy.
    """

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int = 300
    ) -> AsyncIterator[str]:
        """
        Accept conversation history and system prompt.
        Yield response tokens as they are generated.
        Never return the full response at once.
        """
    