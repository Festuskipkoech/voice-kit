import numpy as np

class VADProcessor:
    """
    Voice Activity Detection.
 
    Determines whether an audio chunk contains speech or silence.
    Runs before STT so the model never processes silence — this
    reduces cost and latency significantly under real conditions.
 
    Prototype implementation uses energy-based detection.
    Production implementation swaps this for Silero VAD without
    changing any other code — the interface is identical.
    """

    def __init__(self, sensitivity: float = 0.5):
        if not 0.0 <= sensitivity <= 1.0:
            raise ValueError(
                f"VAD sensitivity must be between 0.0 and 1.0, got {sensitivity}"
            )
        self.sensitivity = sensitivity
        
        # energy threshold scales inversely with sensitivity
        # high sensitivity (0.9) = very low threshold = catches quiet speech
        # low sensitivity (0.1)  = high threshold    = only loud speech passes
        self._threshhold = 0.05 * (1.0 - sensitivity) + 0.001

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Return True if the chunk contains speech above the threshold.
        Audio must be float32, 16kHz mono, values in [-1.0, 1.0].
        """
        if len(audio_chunk) == 0:
            return False
        
        raws = float(np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2)))
        return raws > self._threshhold
    
    def filter_stream(self, chunks: list[np.ndarray]) -> list[np.ndarray]:
        """
        Filter a list of audio chunks, returning only those containing speech.
        Convenience method for batch processing.
        """
        return [chunk for chunk in chunks if self.is_speech(chunk)]