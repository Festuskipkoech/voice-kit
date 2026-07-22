import re

def clean_for_tts(text: str) -> str:
    """
    Strip markdown and non-ASCII characters from LLM output
    before sending to TTS.

    This is a safety net — the system prompt is the primary fix.
    This catches anything that slips through.
    """
    # remove emojis and non-ascii
    text = text.encode("ascii", "ignore").decode("ascii")
    # remove markdown bold/italic
    text = re.sub(r"\*+|_+", "", text)
    # remove markdown headers
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # remove bullet points
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)
    # remove backticks
    text = re.sub(r"`+", "", text)
    # collapse newlines and spaces
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()