"""
Exceptions raised by the Voicekit SDK.
"""

class VoicekitError(Exception):
    """Base class for all Voicekit SDK errors."""


class HandshakeError(VoicekitError):
    """
    Raised when the server rejects the session.config handshake.
    The `message` attribute contains the human-readable reason from the server.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class PolicyError(HandshakeError):
    """
    Raised when the server rejects a session override (e.g. voice or VAD)
    because the server policy forbids it.  WebSocket close code 1008.
    """


class ProtocolError(VoicekitError):
    """
    Raised when the server sends a frame that violates the protocol spec,
    or when the client attempts an operation out of order.
    """


class NotConnectedError(VoicekitError):
    """Raised when a method is called before connect() / start_session()."""
