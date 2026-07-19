import asyncio
 
import anthropic
 
from voicekit.providers.base import LLMProvider 
 
class ClaudeProvider(LLMProvider):
    """
    LLM provider via Anthropic Claude API.
 
    Reads ANTHROPIC_API_KEY from environment automatically.
    The gateway sets this from VOICEKIT_LLM_API_KEY before instantiating
    the pipeline.
 
    Executor + queue pattern:
        The Anthropic Python SDK is synchronous. Calling it directly on
        the event loop blocks all other coroutines — including TTS audio
        streaming — for the duration of the LLM response. This would
        cause audio to stutter and sessions to appear frozen.
 
        The solution: run the synchronous SDK in a thread executor.
        Tokens are passed from the thread to the event loop via an
        asyncio.Queue using loop.call_soon_threadsafe(), which is the
        only thread-safe way to interact with a running event loop.
 
    Model selection:
        claude-haiku-4-5   fastest, cheapest, first token <150ms  recommended
        claude-sonnet-4-6  better reasoning, slower
        claude-opus-4-6    best reasoning, slowest
    """

    def __init__(self, model: str ="claude-haiku-4.5"):
        self.model = model
        self.client = anthropic.Anthropic()
    
    async def stream(self, messages: list[dict], system: str, max_tokens: int = 300):
        """
        Stream LLM response tokens.
 
        Accepts conversation history in OpenAI-compatible format:
            [{"role": "user", "content": "..."}, ...]
 
        Yields tokens as they arrive from the API.
        Never buffers the full response.
        """
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _stream_sync() -> None:
            try:
                with self.client.messages.stream(
                    model = self.model,
                    max_tokens = max_tokens,
                    system=system,
                    messages=messages
                ) as stream:
                    for token in stream.text_stream:
                        loop.call_soon_threadsafe(queue.put_nowait, token)
            
            finally:
                # always send sentinel even if an exception occurred
                # without this the consumer hangs forever
                loop.call_soon_threadsafe(queue.put_nowait, None)

        executor_task = loop.run_in_executor(None, _stream_sync)

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token
        
         # wait for executor to finish cleanly
        await executor_task
