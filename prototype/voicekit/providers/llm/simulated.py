import asyncio
from typing import AsyncIterator
 
from voicekit.providers.base import LLMProvider

RESPONSES = [
    "I understand your question. Let me help you with that.",
    "That is a great point. Here is what I think about it.",
    "Thank you for asking. The answer is straightforward.",
    "I can help you with that. Here is what you need to know.",
    "Good question. There are a few things worth considering here.",
]

class SimulatedLLM(LLMProvider):
    """
    Simulates a real LLM without making any API calls.
 
    Realistic behaviour:
    - Initial delay before first token (simulates network + inference)
    - Streams tokens one word at a time with realistic inter-token delay
    - Cycles through predictable responses so tests can assert on output
    - Respects conversation history length to simulate growing context
    """
    
    def __init__(self, model: str = "simulated", token_delay_ms: int = 30):
        self.model = model
        self.token_delay_ms = token_delay_ms
        self._turn_count = 0

    async def stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int = 300
    ) -> AsyncIterator[str]:
        response = RESPONSES[self._turn_count % len(RESPONSES)]
        self._turn_count += 1

        # simulate network latency to first token
        await asyncio.sleep(0.1)

        words = response.split()
        for i,word in enumerate(words):
            is_last = 1 == len(words) - 1
            yield word + ("" if is_last else " ")
            await asyncio.sleep(self.token_delay_ms / 1000)
