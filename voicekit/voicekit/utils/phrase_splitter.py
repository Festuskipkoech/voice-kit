"""
PhraseStream — splits an LLM token stream into clean, complete phrases.

Three operating modes:

    MODE_DETECTING watches the first DETECTION_THRESHOLD tokens for <|BREAK|>
                if found  MODE_MARKER  (reliable, fast, natural boundaries)
                if timeout MODE_FALLBACK (punctuation splitting, safety net)

    MODE_MARKER primary path — LLM correctly inserting <|BREAK|> markers
                     split on marker, strip it, clean the phrase, yield

    MODE_FALLBACK LLM failed to insert markers (rare, < 1% with correct prompt)
                split on strong sentence boundaries only: . ! ?
                minimum MIN_PHRASE_CHARS before allowing a split
                never commas or semicolons — prevents mid-thought cuts

Mode is set once and never reversed.

Cleaning is done here, once, on each complete phrase before it is yielded.
Nothing downstream needs to clean anything — phrases leave this class ready
for g2p and TTS synthesis.

Rate adaptation:
    PhraseStream yields phrases into phrase_queue(maxsize=2) in remote_pipeline.
    When Kokoro is busy, the queue fills and phrase_queue.put() blocks naturally.
    Tokens from the LLM accumulate in self.buffer until space opens.
    The bounded queue is the rate controller — no timers, no polling needed.
"""
import re
from typing import AsyncIterator

from voicekit.utils.text_cleaner import clean_for_tts

MARKER = "<|BREAK|>"

MODE_DETECTING = "detecting"
MODE_MARKER    = "marker"
MODE_FALLBACK  = "fallback"

DETECTION_THRESHOLD = 30
MIN_PHRASE_CHARS    = 20
SENTENCE_END        = re.compile(r"[.!?](\s|$)")


class PhraseStream:
    """
    Consumes raw LLM tokens (including <|BREAK|> markers).
    Yields clean, complete phrases ready for TTS synthesis.

    Usage in remote_pipeline.py:

        splitter = PhraseStream()
        async for token in self._llm.stream(...):
            async for phrase in splitter.feed(token):
                await phrase_queue.put(phrase)
        async for phrase in splitter.flush():
            await phrase_queue.put(phrase)
        await phrase_queue.put(None)
    """

    def __init__(self) -> None:
        self.mode        = MODE_DETECTING
        self.buffer      = ""
        self.token_count = 0

    async def feed(self, token: str) -> AsyncIterator[str]:
        """
        Feed one raw token from the LLM.
        Yields zero or more clean phrases when a split point is detected.
        """
        self.token_count += 1
        self.buffer += token

        if self.mode == MODE_DETECTING:
            if MARKER in self.buffer:
                self.mode = MODE_MARKER
                async for phrase in self._split_on_marker():
                    yield phrase
            elif self.token_count >= DETECTION_THRESHOLD:
                self.mode = MODE_FALLBACK
                async for phrase in self._split_on_punctuation():
                    yield phrase

        elif self.mode == MODE_MARKER:
            async for phrase in self._split_on_marker():
                yield phrase

        elif self.mode == MODE_FALLBACK:
            async for phrase in self._split_on_punctuation():
                yield phrase

    async def flush(self) -> AsyncIterator[str]:
        """
        Call once after the LLM token stream ends.
        Drains any remaining buffer as a final phrase.
        """
        remaining = clean_for_tts(self.buffer)
        if remaining:
            yield remaining
        self.buffer = ""

    async def _split_on_marker(self) -> AsyncIterator[str]:
        """
        Split buffer on every <|BREAK|> occurrence.
        Strips the marker and cleans each phrase before yielding.
        Leaves any incomplete suffix (after the last marker) in self.buffer.
        """
        while MARKER in self.buffer:
            raw_phrase, self.buffer = self.buffer.split(MARKER, 1)
            phrase = clean_for_tts(raw_phrase)
            if phrase:
                yield phrase

    async def _split_on_punctuation(self) -> AsyncIterator[str]:
        """
        Fallback: split on strong sentence-ending punctuation only (.!?)
        Requires MIN_PHRASE_CHARS before allowing any split.
        Cleans each phrase before yielding.
        """
        while True:
            match = SENTENCE_END.search(self.buffer)
            if not match:
                break
            end = match.end()
            raw_phrase = self.buffer[:end]
            self.buffer = self.buffer[end:]
            phrase = clean_for_tts(raw_phrase)
            if phrase and len(phrase) >= MIN_PHRASE_CHARS:
                yield phrase