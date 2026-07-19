import numpy as np

class VADProcessor:
    """
    Voice Activity Detection using Silero VAD.

    Silero is a streaming RNN model. It maintains hidden state across
    512-sample windows (32ms at 16kHz). Each call to the model updates
    internal hidden layers — it builds context over time rather than
    making stateless per-window decisions.

    Correct usage for streaming audio pipeline:
        - Feed 512-sample windows sequentially without resetting state
        - State accumulates within a single utterance — this is intentional
        - Reset state after each complete utterance (end_of_speech signal)
        - Each VADProcessor instance owns its own model state

    Why NOT reset before each chunk:
        Resetting kills the RNN context the model needs to detect speech.
        A fresh RNN has no context and gives unreliable low confidence
        even on real speech audio.

    Why NOT split and call independently:
        Independent calls lose the sequential context that makes Silero
        accurate. The model was designed to see a continuous audio stream.

    Production pattern:
        On each 512-sample window from the microphone:
            confidence = model(window, 16000)
            if confidence > threshold: speech detected
        At end of utterance (user stops speaking):
            model.reset_states()  ← reset for next utterance

    Model loads at class level — shared across all instances.
    Each voice session creates its own VADProcessor with its own
    isolated model state via per-instance state management.
    """

    _model = None
    _loaded = False

    WINDOW_SIZE = 512       # required by Silero at 16kHz — do not change
    SAMPLE_RATE = 16000

    def __init__(self, sensitivity: float = 0.5):
        if not 0.0 <= sensitivity <= 1.0:
            raise ValueError(
                f"VAD sensitivity must be between 0.0 and 1.0, got {sensitivity}"
            )
        self.sensitivity = sensitivity
        self._threshold = sensitivity
        self._ensure_loaded()

    @classmethod
    def _ensure_loaded(cls) -> None:
        if not cls._loaded:
            from silero_vad import load_silero_vad
            cls._model = load_silero_vad()
            cls._loaded = True

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Return True if the chunk contains speech.

        Feeds the chunk as sequential 512-sample windows to the Silero RNN.
        State accumulates across windows within the chunk — this is correct
        and intentional, matching how Silero was designed to work.

        Audio requirements:
            dtype:       float32
            sample rate: 16000 Hz
            channels:    1 (mono)
            values:      normalised to [-1.0, 1.0]

        Returns True if any window in the chunk exceeds the threshold.
        Caller should invoke reset_states() after each complete utterance.
        """
        import torch

        if len(audio_chunk) == 0:
            return False

        # pad to nearest multiple of WINDOW_SIZE
        remainder = len(audio_chunk) % self.WINDOW_SIZE
        if remainder != 0:
            audio_chunk = np.pad(
                audio_chunk,
                (0, self.WINDOW_SIZE - remainder)
            )

        # feed windows sequentially — state accumulates across them
        # this is the correct streaming pattern for Silero
        for i in range(0, len(audio_chunk), self.WINDOW_SIZE):
            window = audio_chunk[i:i + self.WINDOW_SIZE]
            tensor = torch.from_numpy(window.astype(np.float32))
            with torch.no_grad():
                confidence = self._model(tensor, self.SAMPLE_RATE).item()
            if confidence >= self._threshold:
                return True

        return False

    def reset_states(self) -> None:
        """
        Reset Silero's internal RNN hidden state.

        Call this after each complete utterance — when the user has finished
        speaking and the pipeline has processed the turn. This prepares the
        model for the next independent utterance.

        Do NOT call between chunks within a single utterance — that would
        destroy the context the model needs to detect speech reliably.
        """
        self._model.reset_states()

    def filter_stream(self, chunks: list[np.ndarray]) -> list[np.ndarray]:
        """
        Filter a list of audio chunks, returning only those containing speech.

        State accumulates across chunks — this matches real streaming behaviour
        where audio arrives continuously. Silence chunks naturally score low
        confidence, speech chunks score high.

        Call reset_states() after the full stream ends if needed.
        """
        return [chunk for chunk in chunks if self.is_speech(chunk)]