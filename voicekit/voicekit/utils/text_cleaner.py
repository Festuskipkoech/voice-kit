"""
single source of truth for cleaning text before TTS synthesis.

Called by PhraseStream after splitting, once per complete phrase.
Nothing else in the pipeline should clean text — this is the only cleaning point.

Strips in order:
    1. <|BREAK|> markers — LLM-inserted split markers, must not reach TTS
    2. Non-ASCII / emojis — TTS models expect plain ASCII
    3. Markdown bold/italic — asterisks and underscores corrupt prosody
    4. Markdown headers — hash prefixes on spoken lines sound wrong
    5. Bullet points — dashes and bullets are not spoken
    6. Backticks — code fences have no spoken equivalent
    7. Whitespace normalisation — collapse newlines and multiple spaces
"""
import re

def clean_for_tts(text: str) -> str:
    # remove <|BREAK|> markers — must be stripped before anything reaches TTS or client
    text = text.replace("<|BREAK|>", "")
    # remove emojis and non-ascii characters
    text = text.encode("ascii", "ignore").decode("ascii")
    # remove markdown bold and italic markers
    text = re.sub(r"\*+|_+", "", text)
    # remove markdown headers
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # remove bullet points
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)
    # remove backticks
    text = re.sub(r"`+", "", text)
    # collapse newlines into spaces
    text = re.sub(r"\n+", " ", text)
    # collapse multiple spaces into one
    text = re.sub(r" +", " ", text)
    return text.strip()