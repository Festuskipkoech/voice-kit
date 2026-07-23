"""
Provider base classes.

Every STT, TTS, and LLM implementation must satisfy the contract
defined here. The pipeline only ever calls these methods — never
the concrete implementation directly.

Adding a new model:
    1. Write one class that inherits the relevant base class below
    2. Add one line to voicekit/providers/registry.py
    3. Change one config value
    Nothing else changes.
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator

import numpy as np

class STTProvider(ABC):
    """
    Contract every STT model must satisfy.
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
        audio_stream: AsyncIterator[np.ndarray],
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

    synthesize() accepts a single clean phrase string a complete sentence
    or natural speech unit. Splitting and cleaning happen upstream in
    PhraseStream before this method is ever called. No buffering needed here.
    """

    @abstractmethod
    async def load(self) -> None:
        ...

    @abstractmethod
    async def synthesize(
        self,
        phrase: str,
    ) -> AsyncIterator[bytes]:
        """
        Accept a single clean phrase string from the LLM.
        Yield WAV audio chunks as they are generated.

        Chunk 1: complete WAV file (44-byte header + float32 PCM)
        Chunk 2+ raw float32 PCM bytes, no header, 24000 Hz mono

        First audio chunk must arrive before the phrase stream ends —
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
        max_tokens: int = 300,
    ) -> AsyncIterator[str]:
        """
        Accept conversation history and system prompt.
        Yield response tokens as they are generated.
        Never return the full response at once.

        Tokens will include <|BREAK|> markers inserted by the system prompt.
        These are consumed by PhraseStream and never reach the TTS model.
        """