"""
ClaudeProvider — Anthropic Claude LLM provider.

Reads ANTHROPIC_API_KEY from environment automatically.
The gateway sets this from VOICEKIT_LLM_API_KEY before instantiating
the pipeline.

Executor + queue pattern:
    The Anthropic Python SDK is synchronous. Calling it directly on the event
    loop blocks all other coroutines — including the TTS audio streaming — for
    the duration of the LLM response. This would cause audio to stutter and
    sessions to appear frozen.

    Solution: run the synchronous SDK in a thread executor. Tokens are passed
    from the thread to the event loop via an asyncio.Queue using
    loop.call_soon_threadsafe(), which is the only thread-safe way to interact
    with a running event loop.

    The None sentinel is always sent in the finally block. Without it, if the
    SDK raises an exception mid-stream, the consumer (split_tokens in
    remote_pipeline) hangs forever waiting for a token that never arrives.

Temperature 0.1:
    Low temperature makes <|BREAK|> marker insertion deterministic. At 0.3 or
    higher, Claude occasionally skips markers or inserts them inconsistently,
    forcing PhraseStream into fallback mode and degrading streaming latency.
    0.1 gives consistent, reliable marker placement on every response.

Model selection:
    claude-haiku-4-5   fastest, cheapest, first token < 150ms  recommended
    claude-sonnet-4-6  better reasoning, slower
    claude-opus-4-6    best reasoning, slowest
"""
import asyncio

import anthropic

from voicekit.providers.base import LLMProvider

class ClaudeProvider(LLMProvider):

    def __init__(self, model: str = "claude-haiku-4-5") -> None:
        self.model  = model
        self.client = anthropic.Anthropic()

    async def stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int = 300,
    ):
        """
        Stream LLM response tokens.

        Accepts conversation history in OpenAI-compatible format:
            [{"role": "user", "content": "..."}, ...]

        Yields tokens as they arrive from the API, including <|BREAK|> markers
        inserted by the system prompt. PhraseStream in remote_pipeline consumes
        these markers for phrase splitting — they never reach the TTS model or
        the client.
        """
        loop  = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _stream_sync() -> None:
            try:
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    system=system,
                    messages=messages,
                ) as stream:
                    for token in stream.text_stream:
                        loop.call_soon_threadsafe(queue.put_nowait, token)
            finally:
                # always send sentinel — prevents split_tokens() from hanging
                # if an exception occurs mid-stream
                loop.call_soon_threadsafe(queue.put_nowait, None)

        executor_task = loop.run_in_executor(None, _stream_sync)

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

        await executor_task